# A Wavenet For Source Separation - Francesc Lluis - 25.10.2018
# Models.py

import keras
import util
import os
import numpy as np
import layers
import math
import logging


#Singing Voice Separation Wavenet Model

class SingingVoiceSeparationWavenet():

    def __init__(self, config, load_checkpoint=None, input_length=None, target_field_length=None, print_model_summary=False):

        self.config = config
        self.verbosity = config['training']['verbosity']

        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']

        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)

        if input_length is not None:
            self.input_length = int(input_length)
            self.target_field_length = self.input_length - (self.receptive_field_length - 1)
        if target_field_length is not None:
            self.target_field_length = target_field_length
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))
        else:
            self.target_field_length = config['model']['target_field_length']
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))

        self.target_padding = int(config['model']['target_padding'])
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        self.half_target_field_length = int(self.target_field_length // 2)
        self.half_receptive_field_length = int(self.receptive_field_length // 2)
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.activation = keras.layers.Activation('relu')
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        self.target_sample_indices = self.get_target_field_indices()

        self.optimizer = self.get_optimizer()
        self.out_1_loss = self.get_out_1_loss()
        self.out_2_loss = self.get_out_2_loss()
        self.metrics = self.get_metrics()
        self.epoch_num = 0
        self.checkpoints_path = ''
        self.samples_path = ''
        self.history_filename = ''

        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = self.input_length
        self.config['model']['target_field_length'] = self.target_field_length
        self.config['model']['type'] = 'singing-voice'

        self.model = self.setup_model(load_checkpoint, print_model_summary)

    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
        self.samples_path = os.path.join(self.config['training']['path'], 'samples')
        self.history_filename = 'history_' + self.config['training']['path'][
                                             self.config['training']['path'].rindex('/') + 1:] + '.csv'

        model = self.build_model()

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.epoch_num = 0
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                self.epoch_num = int(last_checkpoint[11:16])
            print('Loading model from epoch: %d' % self.epoch_num)
            model.load_weights(last_checkpoint_path)

        else:
            print('Building new model...')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            self.epoch_num = 0

        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)

        if print_model_summary:
            model.summary()

        model.compile(optimizer=self.optimizer,
                      loss={'data_output_1': self.out_1_loss, 'data_output_2': self.out_2_loss}, metrics=self.metrics)
        self.config['model']['num_params'] = model.count_params()

        config_path = os.path.join(self.config['training']['path'], 'config.json')
        if not os.path.exists(config_path):
            util.pretty_json_dump(self.config, config_path)

        if print_model_summary:
            util.pretty_json_dump(self.config)
        return model

    def get_optimizer(self):

        return keras.optimizers.Adam(lr=self.config['optimizer']['lr'], decay=self.config['optimizer']['decay'],
                                     epsilon=self.config['optimizer']['epsilon'])

    def get_out_1_loss(self):

        if self.config['training']['loss']['out_1']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_1']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_1']['l1'],
            self.config['training']['loss']['out_1']['l2'])

    def get_out_2_loss(self):

        if self.config['training']['loss']['out_2']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_2']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_2']['l1'],
            self.config['training']['loss']['out_2']['l2'])

    def get_callbacks(self):

        return [
            keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
                                          monitor='loss'),
            keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path,
                                                         'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5')),
            keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True)
        ]

    def fit_model(self, train_set_generator, num_steps_train, test_set_generator, num_steps_test, num_epochs):

        print('Fitting model with %d training num steps and %d test num steps...' % (num_steps_train, num_steps_test))

        self.model.fit_generator(train_set_generator,
                                 num_steps_train,
                                 epochs=num_epochs,
                                 validation_data=test_set_generator,
                                 validation_steps=num_steps_test,
                                 callbacks=self.get_callbacks(),
                                 verbose=self.verbosity,
                                 initial_epoch=self.epoch_num)

    def separate_batch(self, inputs):
        return self.model.predict_on_batch(inputs)

    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length // 2.0))

    def get_metrics(self):

        return [
            keras.metrics.mean_absolute_error,
            self.valid_mean_absolute_error
        ]

    def valid_mean_absolute_error(self, y_true, y_pred):
        return keras.backend.mean(
            keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))

    def build_model(self):

        data_input = keras.engine.Input(
                shape=(self.input_length,),
                name='data_input')

        data_expanded = layers.AddSingletonDepth()(data_input)

        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                              self.config['model']['filters']['lengths']['res'], padding='same',
                                              use_bias=False,
                                              name='initial_causal_conv')(data_expanded)

        skip_connections = []
        res_block_i = 0
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            for dilation in self.dilations:
                res_block_i += 1
                data_out, skip_out = self.dilated_residual_block(data_out, res_block_i, layer_in_stack, dilation, stack_i)
                if skip_out is not None:
                    skip_connections.append(skip_out)
                layer_in_stack += 1

        data_out = keras.layers.Add()(skip_connections)

        data_out = self.activation(data_out)

        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                              self.config['model']['filters']['lengths']['final'][0],
                                              padding='same',
                                              use_bias=False)(data_out)

        data_out = self.activation(data_out)
        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                              self.config['model']['filters']['lengths']['final'][1], padding='same',
                                              use_bias=False)(data_out)

        data_out = keras.layers.Convolution1D(1, 1)(data_out)

        data_out_vocals_1 = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_1')(
            data_out)

        data_out_vocals_2 = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                                output_shape=lambda shape: (shape[0], shape[1]), name='data_output_2')(
            data_out)

        return keras.engine.Model(inputs=[data_input], outputs=[data_out_vocals_1, data_out_vocals_2])

    def dilated_residual_block(self, data_x, res_block_i, layer_i, dilation, stack_i):

        original_x = data_x

        data_out = keras.layers.Conv1D(2 * self.config['model']['filters']['depths']['res'],
                                                    self.config['model']['filters']['lengths']['res'],
                                                    dilation_rate=dilation, padding='same',
                                                    use_bias=False,
                                                    name='res_%d_dilated_conv_d%d_s%d' % (
                                                    res_block_i, dilation, stack_i),
                                                    activation=None)(data_x)

        data_out_1 = layers.Slice(
            (Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        data_out_2 = layers.Slice(
            (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                             2 * self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        tanh_out = keras.layers.Activation('tanh')(data_out_1)
        sigm_out = keras.layers.Activation('sigmoid')(data_out_2)

        data_x = keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
            [tanh_out, sigm_out])

        data_x = keras.layers.Convolution1D(
            self.config['model']['filters']['depths']['res'] + self.config['model']['filters']['depths']['skip'], 1,
            padding='same', use_bias=False)(data_x)

        res_x = layers.Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                             (self.input_length, self.config['model']['filters']['depths']['res']),
                             name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                               self.config['model']['filters']['depths']['res'] +
                                               self.config['model']['filters']['depths']['skip'])),
                              (self.input_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                               Ellipsis), (self.padded_target_field_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

        res_x = keras.layers.Add()([original_x, res_x])

        return res_x, skip_x


# Multi-Instrument Separation Wavenet Model

class MultiInstrumentSeparationWavenet():

    def __init__(self, config, load_checkpoint=None, input_length=None, target_field_length=None, print_model_summary=False):

        self.config = config
        self.verbosity = config['training']['verbosity']

        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']

        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)

        if input_length is not None:
            self.input_length = input_length
            self.target_field_length = self.input_length - (self.receptive_field_length - 1)
        if target_field_length is not None:
            self.target_field_length = target_field_length
            self.input_length = self.receptive_field_length + (self.target_field_length - 1)
        else:
            self.target_field_length = config['model']['target_field_length']
            self.input_length = self.receptive_field_length + (self.target_field_length - 1)

        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        self.half_target_field_length = self.target_field_length / 2
        self.half_receptive_field_length = self.receptive_field_length / 2
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.activation = keras.layers.Activation('relu')
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        self.target_sample_indices = self.get_target_field_indices()

        self.optimizer = self.get_optimizer()
        self.out_1_loss = self.get_out_1_loss()
        self.out_2_loss = self.get_out_2_loss()
        self.out_3_loss = self.get_out_3_loss()
        self.metrics = self.get_metrics()
        self.epoch_num = 0
        self.checkpoints_path = ''
        self.samples_path = ''
        self.history_filename = ''

        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = self.input_length
        self.config['model']['target_field_length'] = self.target_field_length
        self.config['model']['type'] = 'multi-instrument'

        self.model = self.setup_model(load_checkpoint, print_model_summary)

    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
        self.samples_path = os.path.join(self.config['training']['path'], 'samples')
        self.history_filename = 'history_' + self.config['training']['path'][
                                             self.config['training']['path'].rindex('/') + 1:] + '.csv'

        model = self.build_model()

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.epoch_num = 0
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                self.epoch_num = int(last_checkpoint[11:16])
            print('Loading model from epoch: %d' % self.epoch_num)
            model.load_weights(last_checkpoint_path)

        else:
            print ('Building new model...')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            self.epoch_num = 0

        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)

        if print_model_summary:
            model.summary()

        model.compile(optimizer=self.optimizer,
                      loss={'data_output_1': self.out_1_loss, 'data_output_2': self.out_2_loss,
                            'data_output_3': self.out_3_loss}, metrics=self.metrics)
        self.config['model']['num_params'] = model.count_params()

        config_path = os.path.join(self.config['training']['path'], 'config.json')
        if not os.path.exists(config_path):
            util.pretty_json_dump(self.config, config_path)

        if print_model_summary:
            util.pretty_json_dump(self.config)
        return model

    def get_optimizer(self):

        return keras.optimizers.Adam(lr=self.config['optimizer']['lr'], decay=self.config['optimizer']['decay'],
                                     epsilon=self.config['optimizer']['epsilon'])

    def get_out_1_loss(self):

        if self.config['training']['loss']['out_1']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_1']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_1']['l1'],
            self.config['training']['loss']['out_1']['l2'])

    def get_out_2_loss(self):

        if self.config['training']['loss']['out_2']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_2']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_2']['l1'],
            self.config['training']['loss']['out_2']['l2'])

    def get_out_3_loss(self):
        if self.config['training']['loss']['out_3']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_3']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_3']['l1'],
            self.config['training']['loss']['out_3']['l2'])

    def get_callbacks(self):

        return [
            keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
                                          monitor='loss'),
            keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path,
                                                         'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5')),
            keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True)
        ]

    def fit_model(self, train_set_generator, num_steps_train, test_set_generator, num_steps_test, num_epochs):

        print('Fitting model with %d training num steps and %d test num steps...' % (num_steps_train, num_steps_test))

        self.model.fit_generator(train_set_generator,
                                 num_steps_train,
                                 epochs=num_epochs,
                                 validation_data=test_set_generator,
                                 validation_steps=num_steps_test,
                                 callbacks=self.get_callbacks(),
                                 verbose=self.verbosity,
                                 initial_epoch=self.epoch_num)

    def separate_batch(self, inputs):
        return self.model.predict_on_batch(inputs)

    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_metrics(self):

        return [
            keras.metrics.mean_absolute_error,
            self.valid_mean_absolute_error
        ]

    def valid_mean_absolute_error(self, y_true, y_pred):
        return keras.backend.mean(
            keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))

    def build_model(self):

        data_input = keras.engine.Input(
                shape=(self.input_length,),
                name='data_input')

        data_expanded = layers.AddSingletonDepth()(data_input)

        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                              self.config['model']['filters']['lengths']['res'], padding='same',
                                              use_bias=False,
                                              name='initial_causal_conv')(data_expanded)

        skip_connections = []
        res_block_i = 0
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            for dilation in self.dilations:
                res_block_i += 1
                data_out, skip_out = self.dilated_residual_block(data_out, res_block_i, layer_in_stack, dilation, stack_i)
                if skip_out is not None:
                    skip_connections.append(skip_out)
                layer_in_stack += 1

        data_out = keras.layers.Add()(skip_connections)

        data_out = self.activation(data_out)

        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                              self.config['model']['filters']['lengths']['final'][0],
                                              padding='same',
                                              use_bias=False)(data_out)

        data_out = self.activation(data_out)
        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                              self.config['model']['filters']['lengths']['final'][1], padding='same',
                                              use_bias=False)(data_out)

        data_out = keras.layers.Convolution1D(3, 1)(data_out)

        data_out_vocals = layers.Slice((Ellipsis, slice(0, 1)), (self.padded_target_field_length, 1),
                                       name='slice_data_output_1')(data_out)
        data_out_drums = layers.Slice((Ellipsis, slice(1, 2)), (self.padded_target_field_length, 1),
                                              name='slice_data_output_2')(data_out)
        data_out_bass = layers.Slice((Ellipsis, slice(2, 3)), (self.padded_target_field_length, 1),
                                      name='slice_data_output_3')(data_out)

        data_out_vocals = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_1')(
            data_out_vocals)

        data_out_drums = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_2')(
            data_out_drums)

        data_out_bass = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_3')(
            data_out_bass)

        return keras.engine.Model(inputs=[data_input], outputs=[data_out_vocals, data_out_drums, data_out_bass])

    def dilated_residual_block(self, data_x, res_block_i, layer_i, dilation, stack_i):

        original_x = data_x

        data_out = keras.layers.Conv1D(2 * self.config['model']['filters']['depths']['res'],
                                                    self.config['model']['filters']['lengths']['res'],
                                                    dilation_rate=dilation, padding='same',
                                                    use_bias=False,
                                                    name='res_%d_dilated_conv_d%d_s%d' % (
                                                    res_block_i, dilation, stack_i),
                                                    activation=None)(data_x)

        data_out_1 = layers.Slice(
            (Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        data_out_2 = layers.Slice(
            (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                             2 * self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        tanh_out = keras.layers.Activation('tanh')(data_out_1)
        sigm_out = keras.layers.Activation('sigmoid')(data_out_2)

        data_x = keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
            [tanh_out, sigm_out])

        data_x = keras.layers.Convolution1D(
            self.config['model']['filters']['depths']['res'] + self.config['model']['filters']['depths']['skip'], 1,
            padding='same', use_bias=False)(data_x)

        res_x = layers.Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                             (self.input_length, self.config['model']['filters']['depths']['res']),
                             name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                               self.config['model']['filters']['depths']['res'] +
                                               self.config['model']['filters']['depths']['skip'])),
                              (self.input_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                               Ellipsis), (self.padded_target_field_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

        res_x = keras.layers.Add()([original_x, res_x])

        return res_x, skip_x


class SingingVoiceSeparationAdversarial():

    def __init__(self, config, load_checkpoint=None, input_length=None, target_field_length=None, print_model_summary=True):

        self.config = config
        self.verbosity = config['training']['verbosity']

        self.num_stacks = self.config['model']['num_stacks']
        if type(self.config['model']['dilations']) is int:
            self.dilations = [2 ** i for i in range(0, self.config['model']['dilations'] + 1)]
        elif type(self.config['model']['dilations']) is list:
            self.dilations = self.config['model']['dilations']

        self.receptive_field_length = util.compute_receptive_field_length(config['model']['num_stacks'], self.dilations,
                                                                          config['model']['filters']['lengths']['res'],
                                                                          1)
    
        if input_length is not None:
            self.input_length = int(input_length)
            self.target_field_length = self.input_length - (self.receptive_field_length - 1)
        if target_field_length is not None:
            self.target_field_length = target_field_length
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))
        else:
            self.target_field_length = config['model']['target_field_length']
            self.input_length = int(self.receptive_field_length + (self.target_field_length - 1))

        self.target_padding = config['model']['target_padding']
        self.padded_target_field_length = self.target_field_length + 2 * self.target_padding
        self.half_target_field_length = int(self.target_field_length // 2)
        self.half_receptive_field_length = int(self.receptive_field_length // 2)
        self.num_residual_blocks = len(self.dilations) * self.num_stacks
        self.activation = keras.layers.Activation('relu')
        self.samples_of_interest_indices = self.get_padded_target_field_indices()
        self.target_sample_indices = self.get_target_field_indices()

        self.optimizer = self.get_optimizer()
        self.out_1_loss = self.get_out_1_loss()
        self.out_2_loss = self.get_out_2_loss()
        self.metrics = self.get_metrics()
        self.epoch_num = 0
        self.checkpoints_path = 'sessions/adversarial/checkpoints'
        self.samples_path = ''
        self.history_filename = ''

        self.config['model']['num_residual_blocks'] = self.num_residual_blocks
        self.config['model']['receptive_field_length'] = self.receptive_field_length
        self.config['model']['input_length'] = int(self.input_length)
        self.config['model']['target_field_length'] = self.target_field_length
        self.config['model']['type'] = 'singing-voice-adversarial'

        # Edits made here #####################################
        self.generator = self.build_generator()
        self.discriminator_vocals = self.build_discriminator()
        self.discriminator_acc = self.build_discriminator()
        self.current_loss_value = 0
        self.adv_weight = 0.00001 * 0.25
        #######################################################
        self.model = self.setup_model(load_checkpoint, print_model_summary)

        # Used for early stopping callback
        self.model.stop_training = False


    def setup_model(self, load_checkpoint=None, print_model_summary=False):

        self.checkpoints_path = os.path.join(self.config['training']['path'], 'checkpoints')
        self.samples_path = os.path.join(self.config['training']['path'], 'samples')
        self.history_filename = 'history_' + self.config['training']['path'][
                                             self.config['training']['path'].rindex('/') + 1:] + '.csv'

        model = self.build_model()

        if os.path.exists(self.checkpoints_path) and util.dir_contains_files(self.checkpoints_path):

            if load_checkpoint is not None:
                last_checkpoint_path = load_checkpoint
                self.epoch_num = 0
            else:
                checkpoints = os.listdir(self.checkpoints_path)
                checkpoints.sort(key=lambda x: os.stat(os.path.join(self.checkpoints_path, x)).st_mtime)
                last_checkpoint = checkpoints[-1]
                last_checkpoint_path = os.path.join(self.checkpoints_path, last_checkpoint)
                self.epoch_num = int(last_checkpoint[11:16])
            print('Loading model from epoch: %d' % self.epoch_num)
            model.load_weights(last_checkpoint_path)

        else:
            print ('Building new model...')

            if not os.path.exists(self.config['training']['path']):
                os.mkdir(self.config['training']['path'])

            if not os.path.exists(self.checkpoints_path):
                os.mkdir(self.checkpoints_path)

            self.epoch_num = 0

        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)

        if print_model_summary:
            model.summary()
            self.generator.summary()
            self.discriminator_vocals.summary()
            self.discriminator_acc.summary()

        # model.compile(optimizer=self.optimizer,
        #                 #util.segmentation loss
        #             # loss='binary_crossentropy'
        #             loss=['binary_crossentropy', util.adversarial_loss],
        #             loss_weights=[1, self.adv_weight]
        #               #metrics=self.metrics
        #               )
        model.compile(optimizer=self.optimizer,
                      loss=[self.out_1_loss, self.out_2_loss,util.adversarial_loss,util.adversarial_loss],
                      loss_weights=[1, 1,self.adv_weight,self.adv_weight])
        # metrics=self.metrics)
        # self.config['model']['num_params'] = model.count_params()

        config_path = os.path.join(self.config['training']['path'], 'config.json')
        if not os.path.exists(config_path):
            util.pretty_json_dump(self.config, config_path)

        if print_model_summary:
            util.pretty_json_dump(self.config)
        return model


    #  TRAIN HERE
    # Adapted from https://github.com/eriklindernoren/Keras-GAN
    #
    def fit_model(self, train_set_generator, num_steps_train, test_set_generator, num_steps_test, num_epochs):
        """

        :param train_set_generator:
            batch_inputs = accompaniment_fragment + vocals_fragment
            batch_outputs_1 = vocals_fragment
            batch_outputs_2 = accompaniment_fragment

            Strucure:
            ({'data_input': batch_inputs}, {'data_output_1': batch_outputs_1, 'data_output_2': batch_outputs_2})

        :param num_steps_train:
        :param test_set_generator:
            Same as train_set_generator
        :param num_steps_test:
        :param num_epochs:
        :return:
        """
        print('Fitting model with %d training num steps and %d test num steps...' % (num_steps_train, num_steps_test))

        def custom_metric(y_true, y_pred):
            return self.current_loss_value
        #  Early stopper may prematurely end training (given how GANs train)
        # earlyStopper = keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1, monitor='custom_metric', mode='min')
        modelCheckpoint = keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path, 'checkpoint.{epoch:05d}-{g_loss:.3f}-{d_loss:.3f}.hdf5'),
                                                          monitor='custom_metric', mode='min',period=10),
        csvLogger = keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True)
        # modelCheckpoint returned as 1 element tuple, unpacking done here
        callbacks = [
            # earlyStopper, 
            *modelCheckpoint, 
            csvLogger
            ]
        for c in callbacks:
            c.set_model(self.model)
            c.on_train_begin()

        valid = np.ones((self.config['training']['batch_size'],1603))
        fake_gen = np.zeros((self.config['training']['batch_size'],1603))
            
        # Start from one more than checkpoint and run for num_epochs
        for epoch in range(self.epoch_num + 1, num_epochs + self.epoch_num + 1):
                    
            (mixed_dict, sep_dict) = next(train_set_generator)
            mixed = mixed_dict['data_input']
            vocals = np.array(sep_dict['data_output_1'])
            accompaniments = np.array(sep_dict['data_output_2'])
            
            # Get generator result
            [generated_vocals, generated_accompaniments] = self.generator.predict_on_batch(mixed)

            # Train discriminator
            d_loss = 0
            d_loss_vocals_real = sum(self.discriminator_vocals.train_on_batch(vocals, valid))
            d_loss_vocals_gen = sum(self.discriminator_vocals.train_on_batch(generated_vocals, fake_gen))
            d_loss_acc_real = sum(self.discriminator_acc.train_on_batch(accompaniments, valid))
            d_loss_acc_gen = sum(self.discriminator_acc.train_on_batch(generated_accompaniments, fake_gen))
      
            d_loss = np.add(d_loss_vocals_real,d_loss_vocals_gen)
            d_loss = np.add(d_loss,d_loss_acc_real)
            d_loss = np.add(d_loss,d_loss_acc_gen)
            d_loss *= 0.25
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_gen)

            # Train generator
            g_loss = self.model.train_on_batch(mixed, [vocals,accompaniments,valid, valid])

            self.current_loss_value = g_loss

            
            # 
            callbacks[0].on_epoch_end(epoch-1, {'g_loss':g_loss[0],'d_loss':d_loss})
            callbacks[1].on_epoch_end(epoch, {'g_loss':g_loss[0],'d_loss':d_loss})

            # Plot the progress
            print("Epoch: ", epoch)
            print("D Loss: ", d_loss)
            print(self.model.metrics_names)
            print("G_loss: ", g_loss)
            print()

        for c in callbacks:
                c.on_train_end()



    def separate_batch(self, inputs):
        return self.generator.predict_on_batch(inputs)


    def build_model(self):
        
        # semi_weight = 1
        self.discriminator_vocals.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.discriminator_acc.compile(loss='binary_crossentropy',
                                   optimizer=self.optimizer,
                                   metrics=['accuracy'])
        self.discriminator_vocals.trainable = False
        self.discriminator_acc.trainable = False


        #self.generator.compile(loss='binary_crossentropy',optimizer=self.optimizer)

        # Get generator result
        mixed_source = keras.layers.Input(shape=(self.input_length,))
        [gen_output_vocals, gen_output_acc] = self.generator(mixed_source)
    

        # Feed to discriminator to get confidence maps
        confidence_map_vocals = self.discriminator_vocals(gen_output_vocals)
        confidence_map_acc = self.discriminator_acc(gen_output_acc)

        # output_model = Model(input=[inputs], output=[intermediate_layer])
        # return keras.models.Model( inputs=mixed_source, outputs=[confidence_map_vocals, confidence_map_acc])
        # Using original loss
        return keras.models.Model( inputs=mixed_source, outputs=[gen_output_vocals, gen_output_acc, confidence_map_vocals, confidence_map_acc])

        # loss={'data_output_1': self.out_1_loss, 'data_output_2': self.out_2_loss}

        # Binary cross entropy and adversarial loss, all data is labeled
        

    def build_generator(self):
        
        data_input = keras.engine.Input(
                shape=(self.input_length,),
                name='data_input')

        data_expanded = layers.AddSingletonDepth()(data_input)

        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['res'],
                                              self.config['model']['filters']['lengths']['res'], padding='same',
                                              use_bias=False,
                                              name='initial_causal_conv')(data_expanded)

        skip_connections = []
        res_block_i = 0
        for stack_i in range(self.num_stacks):
            layer_in_stack = 0
            for dilation in self.dilations:
                res_block_i += 1
                data_out, skip_out = self.dilated_residual_block(data_out, res_block_i, layer_in_stack, dilation, stack_i)
                if skip_out is not None:
                    skip_connections.append(skip_out)
                layer_in_stack += 1

        data_out = keras.layers.Add()(skip_connections)

        data_out = self.activation(data_out)

        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                              self.config['model']['filters']['lengths']['final'][0],
                                              padding='same',
                                              use_bias=False)(data_out)

        data_out = self.activation(data_out)
        data_out = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                              self.config['model']['filters']['lengths']['final'][1], padding='same',
                                              use_bias=False)(data_out)

        data_out = keras.layers.Convolution1D(1, 1)(data_out)

        data_out_vocals = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]), name='data_output_1')(
            data_out)

        data_out_acc = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                                output_shape=lambda shape: (shape[0], shape[1]), name='data_output_2')(
            data_out)

        return keras.engine.Model(inputs=[data_input], outputs=[data_out_vocals, data_out_acc])

    def build_discriminator(self):
        inp = keras.engine.Input(shape=(1603,))
        data_out = self.discriminator_body(inp)
        return keras.engine.Model(inp,data_out)

    def build_multi_discriminator(self):
        vocals = keras.engine.Input(shape=(1603,), name='vocals')
        accompaniment = keras.engine.Input(shape=(1603,), name='accompaniment')
        data_out = self.discriminator_body(vocals,'vocals_out')
        data_out_acc = self.discriminator_body(accompaniment,'acc_out')
        
        return keras.engine.Model(inputs=[vocals, accompaniment], outputs=[data_out, data_out_acc])


    def discriminator_body(self, inp,name='inp'):
        inp = layers.AddSingletonDepth()(inp)
        inp = keras.layers.Convolution1D(
                                            self.config['model']['filters']['depths']['res'],
                                            self.config['model']['filters']['lengths']['res'], 
                                            padding='same',
                                            use_bias=False,
                                            
                                            )(inp)
        inp = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(inp)
        inp = keras.layers.Dropout(0.25)(inp)
        inp = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][0],
                                            self.config['model']['filters']['lengths']['final'][0],
                                            padding='same',
                                            use_bias=False
                                            )(inp)
        inp = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(inp)
        inp = keras.layers.Dropout(0.25)(inp)
        inp = keras.layers.Convolution1D(self.config['model']['filters']['depths']['final'][1],
                                            self.config['model']['filters']['lengths']['final'][1], padding='same',
                                            use_bias=False
                                            )(inp)
        inp = keras.layers.advanced_activations.LeakyReLU(alpha=0.2)(inp)
        inp = keras.layers.Dropout(0.25)(inp)
        inp = keras.layers.Convolution1D(1, 1)(inp)
        # inp = keras.layers.UpSampling1D(size=12)(inp)
        
        
        inp = keras.layers.Lambda(lambda x: keras.backend.squeeze(x, 2),
                                              output_shape=lambda shape: (shape[0], shape[1]),
                                              name=name)(inp)
        # inp = keras.layers.Flatten()(inp)
        # inp = keras.layers.Dense(1603, activation='sigmoid')(inp)
        inp = keras.layers.Activation(activation='sigmoid')(inp)

        return inp
        


    def dilated_residual_block(self, data_x, res_block_i, layer_i, dilation, stack_i):

        original_x = data_x

        data_out = keras.layers.Conv1D(2 * self.config['model']['filters']['depths']['res'],
                                                    self.config['model']['filters']['lengths']['res'],
                                                    dilation_rate=dilation, padding='same',
                                                    use_bias=False,
                                                    name='res_%d_dilated_conv_d%d_s%d' % (
                                                    res_block_i, dilation, stack_i),
                                                    activation=None)(data_x)

        data_out_1 = layers.Slice(
            (Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_1_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        data_out_2 = layers.Slice(
            (Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                             2 * self.config['model']['filters']['depths']['res'])),
            (self.input_length, self.config['model']['filters']['depths']['res']),
            name='res_%d_data_slice_2_d%d_s%d' % (self.num_residual_blocks, dilation, stack_i))(data_out)

        tanh_out = keras.layers.Activation('tanh')(data_out_1)
        sigm_out = keras.layers.Activation('sigmoid')(data_out_2)

        data_x = keras.layers.Multiply(name='res_%d_gated_activation_%d_s%d' % (res_block_i, layer_i, stack_i))(
            [tanh_out, sigm_out])

        data_x = keras.layers.Convolution1D(
            self.config['model']['filters']['depths']['res'] + self.config['model']['filters']['depths']['skip'], 1,
            padding='same', use_bias=False)(data_x)

        res_x = layers.Slice((Ellipsis, slice(0, self.config['model']['filters']['depths']['res'])),
                             (self.input_length, self.config['model']['filters']['depths']['res']),
                             name='res_%d_data_slice_3_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((Ellipsis, slice(self.config['model']['filters']['depths']['res'],
                                               self.config['model']['filters']['depths']['res'] +
                                               self.config['model']['filters']['depths']['skip'])),
                              (self.input_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_data_slice_4_d%d_s%d' % (res_block_i, dilation, stack_i))(data_x)

        skip_x = layers.Slice((slice(self.samples_of_interest_indices[0], self.samples_of_interest_indices[-1] + 1, 1),
                               Ellipsis), (self.padded_target_field_length, self.config['model']['filters']['depths']['skip']),
                              name='res_%d_keep_samples_of_interest_d%d_s%d' % (res_block_i, dilation, stack_i))(skip_x)

        res_x = keras.layers.Add()([original_x, res_x])

        return res_x, skip_x

    
    def get_optimizer(self):

        return keras.optimizers.Adam(lr=self.config['optimizer']['lr'], decay=self.config['optimizer']['decay'],
                                     epsilon=self.config['optimizer']['epsilon'])

    def get_out_1_loss(self):

        if self.config['training']['loss']['out_1']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_1']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_1']['l1'],
            self.config['training']['loss']['out_1']['l2'])

    def get_out_2_loss(self):

        if self.config['training']['loss']['out_2']['weight'] == 0:
            return lambda y_true, y_pred: y_true * 0

        return lambda y_true, y_pred: self.config['training']['loss']['out_2']['weight'] * util.l1_l2_loss(
            y_true, y_pred, self.config['training']['loss']['out_2']['l1'],
            self.config['training']['loss']['out_2']['l2'])

    def get_callbacks(self):

        return [
            keras.callbacks.EarlyStopping(patience=self.config['training']['early_stopping_patience'], verbose=1,
                                          monitor='loss'),
            keras.callbacks.ModelCheckpoint(os.path.join(self.checkpoints_path,
                                                         'checkpoint.{epoch:05d}-{val_loss:.3f}.hdf5')),
            keras.callbacks.CSVLogger(os.path.join(self.config['training']['path'], self.history_filename), append=True)
        ]
    def get_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length,
                     target_sample_index + self.half_target_field_length + 1)

    def get_padded_target_field_indices(self):

        target_sample_index = self.get_target_sample_index()

        return range(target_sample_index - self.half_target_field_length - self.target_padding,
                     target_sample_index + self.half_target_field_length + self.target_padding + 1)

    def get_target_sample_index(self):
        return int(np.floor(self.input_length / 2.0))

    def get_metrics(self):

        return [
            keras.metrics.mean_absolute_error,
            self.valid_mean_absolute_error
        ]
    

    def valid_mean_absolute_error(self, y_true, y_pred):
        return keras.backend.mean(
            keras.backend.abs(y_true[:, 1:-2] - y_pred[:, 1:-2]))
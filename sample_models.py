from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Merge, MaxPooling1D,
                          Maximum, Add)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = LSTM(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the modelfrom keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Merge, MaxPooling1D,
                          Maximum, Add)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn =  BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def multi_layer_cnn_output_length(input_length, cnn_shapes):
    in_len = input_length
    for cnn_shape in cnn_shapes:
        in_len = cnn_output_length(in_len, cnn_shape["filter_size"],
                                    cnn_shape["border_mode"], cnn_shape["stride"],
                                    cnn_shape["dilation"])


    return in_len

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    bn_rnn = input_data
    for i in range(recur_layers):
        simp_rnn = GRU(units, return_sequences=True,
                   implementation=2, name='rnn_{}'.format(i))(bn_rnn)
        bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True,
                                  implementation=2, name="bidirectional_rnn"))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, cnn_layer, filters, kernel_size, conv_stride,
                conv_border_mode, cnn_pool_size, cnn_dilation_rate,
                rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                output_dim=29):
    """ Build a deep network for speech 
    """
    # TODO: Specify the layers in your network
    # Main acoustic input
    dnece_cnn = lambda x: Conv1D(x.filters[1])
    maxout_function = lambda x: Maximum()([Dense(K.int_shape(x)[2])(x), Dense(K.int_shape(x)[2])(x)])
    cnn_shapes = []

    input_data = Input(name='the_input', shape=(None, input_dim))

    # initial_maxout = maxout_function(initial_conv_1d)

    maxout = input_data
    # maxout = initial_maxout
    for i in range(0, cnn_layer):

        original_input = maxout

        conv_1d = Conv1D(filters, kernel_size,
                         strides=conv_stride,
                         padding=conv_border_mode,
                         dilation_rate=2 ** i,
                         name='res_conv1d_layer{}'.format(i))(original_input)

        maxout = conv_1d
        # res
        # res = Add()([conv_1d, original_input])

        # maxout
        # maxout = maxout_function(res)
        # maxout = maxout_function(conv_1d)
        # if i == 0:
        #     maxout = MaxPooling1D(pool_size=cnn_pool_size)(maxout)

        cnn_shapes.append({"filter_size": filters, "border_mode": conv_border_mode,
                           "stride": conv_stride, "dilation": 2 ** i})

    bn = maxout
    for i in range(rnn_layer):
        bidir_rnn = Bidirectional(GRU(rnn_units, return_sequences=True,
                                      implementation=2,
                                      name="bidirectional_rnn_layer{}".format(i),
                                      dropout=rnn_dropout,
                                      recurrent_dropout=rnn_recurrent_dropout
                                      ))(bn)
        bn = BatchNormalization()(bidir_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length(
    #    x, kernel_size, conv_border_mode, conv_stride, dilation=cnn_dilation_rate)
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

# for debug
if __name__ == "__main__":
    from keras.backend.tensorflow_backend import set_session
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    set_session(tf.Session(config=config))

    # watch for any changes in the sample_models module, and reload it automatically
    # import NN architectures for speech recognition
    # import function for training acoustic model
    from train_utils import train_model

    model_end = final_model(input_dim=161, cnn_layer=1, filters=256, kernel_size=11, conv_stride=1,
                            conv_border_mode='valid', cnn_pool_size=2, cnn_dilation_rate=2,
                            rnn_layer=2, rnn_units=200, rnn_dropout=0.3, rnn_recurrent_dropout=0.3,
                            output_dim=29)
    train_model(input_to_softmax=model_end,
                pickle_path='model_end.pickle',
                save_model_path='model_end.h5',
                minibatch_size=20,
                epochs=20,
                spectrogram=True)  # change to False if you would like to use MFCC features
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn =  BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    bn_rnn = input_data
    for i in range(recur_layers):
        simp_rnn = GRU(units, return_sequences=True,
                   implementation=2, name='rnn_{}'.format(i))(bn_rnn)
        bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, return_sequences=True,
                                  implementation=2, name="bidirectional_rnn"))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, cnn_layer, filters, kernel_size, conv_stride,
                conv_border_mode, cnn_pool_size, cnn_dilation_rate,
                rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                output_dim=29):
    """ Build a deep network for speech 
    """
    # TODO: Specify the layers in your network
    # Main acoustic input
    dnece_cnn = lambda x: Conv1D(x.filters[1])
    maxout_function = lambda x: Maximum()([Dense(K.int_shape(x)[2])(x), Dense(K.int_shape(x)[2])(x)])

    input_data = Input(name='the_input', shape=(None, input_dim))
    initial_conv_1d = Conv1D(filters, kernel_size,
                             strides=conv_stride,
                             padding=conv_border_mode,
                             dilation_rate=1,
                             name='conv1d_initial_layer')(input_data)

    #print(K.int_shape(initial_conv_1d))
    # initial_maxout = maxout_function(initial_conv_1d)

    maxout = initial_conv_1d
    # maxout = initial_maxout
    for i in range(cnn_layer):

        original_input = maxout

        conv_1d = Conv1D(filters, kernel_size,
                         strides=conv_stride,
                         padding=conv_border_mode,
                         dilation_rate=2 ** i,
                         name='res_conv1d_layer{}'.format(i))(original_input)

        maxout = conv_1d
        # res
        # res = Add()([conv_1d, original_input])

        # maxout
        # maxout = maxout_function(res)
        # maxout = maxout_function(conv_1d)
        # if i == 0:
        maxout = MaxPooling1D(pool_size=cnn_pool_size)(maxout)

    bn = maxout
    for i in range(rnn_layer):
        bidir_rnn = Bidirectional(GRU(rnn_units, return_sequences=True,
                                      implementation=2,
                                      name="bidirectional_rnn_layer{}".format(i),
                                      dropout=rnn_dropout,
                                      recurrent_dropout=rnn_recurrent_dropout
                                      ))(bn)
        bn = BatchNormalization()(bidir_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, dilation=cnn_dilation_rate)
    print(model.summary())
    return model
from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Merge, MaxPooling1D,
                          Maximum, Add, Multiply, ZeroPadding3D, Dropout, Conv2D)
from keras.regularizers import l2


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
    l = (output_length + stride - 1) // stride
    return l

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


def final_model(input_dim, cnn_layer, filters, 
                rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                output_dim=29):
    """ Build a deep network for speech
    """
    cnn_shapes = []
    conv_stride = 1
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, 2,
                     strides=conv_stride,
                     padding="valid",
                     use_bias=True,
                     name='initial_conv1d_layer')(input_data)

    cnn_shapes.append({"filter_size": 2, "border_mode": "valid",
                       "stride": conv_stride, "dilation": 1})

    res = conv_1d
    for i in range(1, cnn_layer):
        original = res

        conv_1d = Conv1D(filters, 2,
                         strides=conv_stride,
                         padding="same",
                         dilation_rate=2 ** i,
                         use_bias=True,
                         name='res_conv1d_layer{}'.format(i))(res)
        sigmoid_actitvated = Activation("sigmoid")(conv_1d)
        tanh_activated = Activation("tanh")(conv_1d)
        
        cnn_shapes.append({"filter_size": 2, "border_mode": "same",
                           "stride": conv_stride, "dilation": 2 ** i})
        
        merged = Multiply()([sigmoid_actitvated, tanh_activated])

        x = Conv1D(filters, 1, strides=1, padding="same")(merged)
        
        cnn_shapes.append({"filter_size": 1, "border_mode": "same",
                           "stride": conv_stride, "dilation": 1})
        
        res = Add()([x, original])
    activated = Activation("relu")(res)

    x = Conv1D(filters, 1, strides=1, padding="same",
                    use_bias=True,)(activated)
    activated = Activation("relu")(x)

    cnn_shapes.append({"filter_size": 1, "border_mode": "same",
                       "stride": conv_stride, "dilation": 1})

    bn = activated
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
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model
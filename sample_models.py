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


# todo remove unused argument
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

    conv_1d = input_data
    for i in range(0, cnn_layer):

        conv_1d = Conv1D(filters, kernel_size,
                         strides=conv_stride,
                         padding=conv_border_mode,
                         dilation_rate=2 ** i,
                         name='res_conv1d_layer{}'.format(i))(conv_1d)

        cnn_shapes.append({"filter_size": kernel_size, "border_mode": conv_border_mode,
                           "stride": conv_stride, "dilation": 2 ** i})

    bn = conv_1d
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
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

def final_model_1(input_dim, cnn_layer, filters, kernel_size, conv_stride,
                conv_border_mode, cnn_pool_size,
                rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                output_dim=29):
    """ Build a deep network for speech
    """
    # dnece_cnn = lambda x: Conv1D(x.filters[1])
    # maxout_function = lambda x: Maximum()([Dense(K.int_shape(x)[2])(x), Dense(K.int_shape(x)[2])(x)])
    cnn_shapes = []
    conv_stride = 1
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, 2,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     name='initial_conv1d_layer')(input_data)

    cnn_shapes.append({"filter_size": 2, "border_mode": "valid",
                       "stride": conv_stride, "dilation": 1})

    skipped = []
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

        merged = Multiply()([sigmoid_actitvated, tanh_activated])

        padded = Conv1D(filters, 1, strides=1, padding="same",
                        name='padding_layer_{}'.format(i))(merged)
        skipped.append(padded)

        # pad_shape = ((0, 0),
        #            (2, 0),
        #             (0, 0))
        # padded = ZeroPadding3D(padding=pad_shape)(padded)
        res = Add()([padded, original])

        # maxout = maxout_function(res)
        # maxout = maxout_function(conv_1d)
        # if i == 0:
        #     maxout = MaxPooling1D(pool_size=cnn_pool_size)(maxout)

        cnn_shapes.append({"filter_size": 2, "border_mode": "same",
                           "stride": conv_stride, "dilation": 2 ** i})

    merged = Add()(skipped)

    activated = Activation("relu")(merged)

    padded = Conv1D(filters, 1, strides=1, padding="same",
                    name='padding_layer_')(activated)
    activated = Activation("relu")(padded)

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
    #model.output_length = lambda x: cnn_output_length(
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

def final_model_2(input_dim, cnn_layer, filters, kernel_size, conv_stride,
                conv_border_mode, cnn_pool_size,
                rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                output_dim=29):
    """ Build a deep network for speech
    """
    cnn_shapes = []
    conv_stride = 1
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, 2,
                     strides=conv_stride,
                     padding=conv_border_mode,
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

        merged = Multiply()([sigmoid_actitvated, tanh_activated])

        padded = Conv1D(filters, 1, strides=1, padding="same",
                        name='padding_layer_{}'.format(i))(merged)
        
        res = Add()([padded, original])

        # maxout = maxout_function(res)
        # maxout = maxout_function(conv_1d)
        # if i == 0:
        #     maxout = MaxPooling1D(pool_size=cnn_pool_size)(maxout)

        cnn_shapes.append({"filter_size": 2, "border_mode": "same",
                           "stride": conv_stride, "dilation": 2 ** i})

    activated = Activation("relu")(res)

    padded = Conv1D(filters, 1, strides=1, padding="same",
                    use_bias=True,)(activated)
    activated = Activation("relu")(padded)

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
    #model.output_length = lambda x: cnn_output_length(
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

def final_model_3(input_dim, res_layers, res_stack, filters, conv_border_mode,
                  rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                  output_dim=29):
    """ Build a deep network for speech
    """
    filter_size = 3
    res_conv_w_reg = 1e-6
    conv_stride = 1
    kernel_initializer = "he_normal"

    def res_dilatted_conv(input, dilation_rate):

        conv_1d = Conv1D(filters, filter_size,
                         strides=conv_stride,
                         padding="same",
                         dilation_rate=dilation_rate,
                         use_bias=True,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=l2(res_conv_w_reg)
                         #name="dilation({})_conv_in_res".format(dilation_rate)
                         )(input)
        sigmoid_actitvated = Activation("sigmoid")(conv_1d)
        tanh_activated = Activation("tanh")(conv_1d)

        merged = Multiply()([sigmoid_actitvated, tanh_activated])

        x = Conv1D(filters,
                        1,
                        strides=1,
                        padding="same",
                        use_bias=True,
                        kernel_regularizer=l2(res_conv_w_reg),
                        kernel_initializer=kernel_initializer
                        )(merged)

        res = Add()([x, input])
        return res

    cnn_shapes = []
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, 2,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     use_bias=True,
                     kernel_regularizer=l2(res_conv_w_reg),
                     kernel_initializer=kernel_initializer,
                     name='initial_conv1d_layer')(input_data)

    cnn_shapes.append({"filter_size": filter_size, "border_mode": "valid",
                       "stride": conv_stride, "dilation": 1})

    res = conv_1d
    for i in range(res_stack):
        for j in range(res_layers):
            if i == 0 and j == 0:
                continue

            res = res_dilatted_conv(res, dilation_rate=2 ** j)

            cnn_shapes.append({"filter_size": filter_size, "border_mode": "same",
                               "stride": conv_stride, "dilation": 2 ** j})

    activated = Activation("relu")(res)

    cnn_shapes.append({"filter_size": 1, "border_mode": "same",
                       "stride": conv_stride, "dilation": 1})

    bn = activated
    for j in range(rnn_layer):
        bidir_rnn = Bidirectional(GRU(rnn_units, return_sequences=True,
                                      implementation=2,
                                      name="bidirectional_rnn_layer{}".format(j),
                                      dropout=rnn_dropout,
                                      recurrent_dropout=rnn_recurrent_dropout
                                      ))(bn)
        bn = BatchNormalization()(bidir_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn)
    #dropouted = Dropout(0.2)(time_dense)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length(
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

def final_model_4(input_dim, res_layers, res_stack, filters, conv_border_mode,
                  output_dim=29):
    """ Build a deep network for speech
    """
    filter_size = 2
    res_conv_w_reg = 1e-6
    conv_stride = 1
    kernel_initializer = "he_normal"

    def res_dilatted_conv(input, dilation_rate):

        conv_1d = Conv1D(filters, filter_size,
                         strides=conv_stride,
                         padding="same",
                         dilation_rate=dilation_rate,
                         use_bias=True,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=l2(res_conv_w_reg)
                         #name="dilation({})_conv_in_res".format(dilation_rate)
                         )(input)
        sigmoid_actitvated = Activation("sigmoid")(conv_1d)
        tanh_activated = Activation("tanh")(conv_1d)

        merged = Multiply()([sigmoid_actitvated, tanh_activated])

        x = Conv1D(filters,
                        1,
                        strides=1,
                        padding="same",
                        use_bias=True,
                        kernel_regularizer=l2(res_conv_w_reg),
                        kernel_initializer=kernel_initializer
                        )(merged)

        res = Add()([x, input])
        return res

    cnn_shapes = []
    input_data = Input(name='the_input', shape=(None, input_dim))

    res = input_data
    for i in range(res_stack):
        for j in range(res_layers):
            if i == 0:
                continue

            res = res_dilatted_conv(res, dilation_rate=2 ** j)

            cnn_shapes.append({"filter_size": filter_size, "border_mode": "same",
                               "stride": conv_stride, "dilation": 2 ** j})

    activated = Activation("relu")(res)

    x = Conv1D(output_dim, 1, strides=1, padding="same", use_bias=True,
                    kernel_regularizer=l2(res_conv_w_reg),
                    kernel_initializer=kernel_initializer)(activated)
    activated = Activation("relu")(x)
    cnn_shapes.append({"filter_size": 1, "border_mode": "same",
                       "stride": conv_stride, "dilation": 1})

    x = Conv1D(output_dim, 1, strides=1, padding="same", use_bias=True,
               kernel_regularizer=l2(res_conv_w_reg),
               kernel_initializer=kernel_initializer)(activated)
    cnn_shapes.append({"filter_size": 1, "border_mode": "same",
                       "stride": conv_stride, "dilation": 1})

    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(x)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    #model.output_length = lambda x: cnn_output_length(
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

def final_model_7(input_dim, cnn_layer, filters, kernel_size, conv_stride,
                conv_border_mode, cnn_pool_size,
                rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                output_dim=29):
    """ Build a deep network for speech
    """
    cnn_shapes = []
    conv_stride = 1
    input_data = Input(name='the_input', shape=(None, input_dim))

    conv_1d = Conv1D(filters, 2,
                     strides=conv_stride,
                     padding=conv_border_mode,
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

        merged = Multiply()([sigmoid_actitvated, tanh_activated])

        padded = Conv1D(filters, 1, strides=1, padding="same",
                        name='padding_layer_{}'.format(i))(merged)
        
        res = Add()([padded, original])

        # maxout = maxout_function(res)
        # maxout = maxout_function(conv_1d)
        # if i == 0:
        #     maxout = MaxPooling1D(pool_size=cnn_pool_size)(maxout)

        cnn_shapes.append({"filter_size": 2, "border_mode": "same",
                           "stride": conv_stride, "dilation": 2 ** i})

    activated = Activation("relu")(res)

    padded = Conv1D(filters, 1, strides=1, padding="same",
                    use_bias=True,)(activated)
    activated = Activation("relu")(padded)

    cnn_shapes.append({"filter_size": 1, "border_mode": "same",
                       "stride": conv_stride, "dilation": 1})

    bn = activated
    for i in range(rnn_layer):
        bidir_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True,
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
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
    model.output_length = lambda x: multi_layer_cnn_output_length(x, cnn_shapes)
    print(model.summary())
    return model

def final_model_8(input_dim, res_layers, num_res_stack,
                  rnn_layer, rnn_units, rnn_dropout, rnn_recurrent_dropout,
                  output_dim=29):
    """ Build a deep network for speech
    """
    filter_size = 2
    filters_unit = 128
    res_conv_w_reg = 1e-6
    conv_stride = 1
    kernel_initializer = "he_normal"

    def res_stack(input, filters, res_layers):
        x = input
        for j in range(res_layers):
            conv_1d = Conv1D(filters, filter_size,
                             strides=conv_stride,
                             padding="same",
                             use_bias=True,
                             kernel_initializer=kernel_initializer,
                             kernel_regularizer=l2(res_conv_w_reg)
                             #name="dilation({})_conv_in_res".format(dilation_rate)
                             )(x)
            x = BatchNormalization()(conv_1d)
            x  = Activation("relu")(x)

        res = Add()([x, input])
        return res

    cnn_shapes = []
    input_data = Input(name='the_input', shape=(None, input_dim))

    filter_size = 128
    #
    # x = Conv1D(128, 3, strides=1, padding="same", use_bias=True,
    #                 kernel_regularizer=l2(res_conv_w_reg),
    #                 kernel_initializer=kernel_initializer)(input_data)

    cnn_shapes.append({"filter_size": filter_size, "border_mode": "same",
                       "stride": conv_stride, "dilation": 1})

    res = input_data
    for i in range(num_res_stack):
        filters = filters_unit * (2 ** i)
        conv_1d = Conv1D(filters, filter_size,
                         strides=conv_stride,
                         padding="same",
                         use_bias=True,
                         kernel_regularizer=l2(res_conv_w_reg),
                         kernel_initializer=kernel_initializer,
                         name='initial_res_stack_{}_layer'.format(i))(res)

        cnn_shapes.append({"filter_size": filter_size, "border_mode": "same",
                           "stride": conv_stride, "dilation": 1})

        res = res_stack(conv_1d, filters, res_layers)

        for _ in range(res_layers):
            cnn_shapes.append({"filter_size": filter_size, "border_mode": "same",
                           "stride": conv_stride, "dilation": 1})

        x = BatchNormalization()(res)
        x = Activation("relu")(x)

    bn = x
    for j in range(rnn_layer):
        bidir_rnn = Bidirectional(GRU(rnn_units, return_sequences=True,
                                      implementation=2,
                                      name="bidirectional_rnn_layer{}".format(j),
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
    #    x, cnn_shapes[0]["filter_size"], cnn_shapes[0]["border_mode"], cnn_shapes[0]["stride"],
    #    dilation=cnn_shapes[0]["dilation"])
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

    # model_end = final_model_3(input_dim=161, res_layers=4, res_stack=2, filters=256,
    #                         conv_border_mode='valid',
    #                         rnn_layer=1, rnn_units=200, rnn_dropout=0.3, rnn_recurrent_dropout=0.3,
    #                         output_dim=29)
    model_end = final_model_8(input_dim=161, res_layers=3, num_res_stack=3,
                                rnn_layer=1, rnn_units=256, rnn_dropout=0.3,
                              rnn_recurrent_dropout=0.3, output_dim=29)

    train_model(input_to_softmax=model_end,
                pickle_path='model_end.pickle',
                save_model_path='model_end.h5',
                minibatch_size=20,
                epochs=20,
                spectrogram=True)  # change to False if you would like to use MFCC features

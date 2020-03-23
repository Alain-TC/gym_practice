import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K


"""Huber loss for Q Learning
References: https://en.wikipedia.org/wiki/Huber_loss
            https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
"""

def _huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


def SmallModel(input_shape, output_shape, learning_rate, dueling=False):
    # Neural Net for Deep-Q learning Model
    input_shape = (input_shape,)
    X_input = Input(input_shape)
    X = X_input
    X = Dense(64, input_shape=input_shape, activation="relu")(X)
    X = Dense(64, activation="relu")(X)
    X = Dense(64, activation="relu")(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)

        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(output_shape,))(state_value)

        action_advantage = Dense(output_shape, input_dim=32, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(output_shape,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(output_shape, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs = X_input, outputs = X, name='CartPole D3QN model')
    #model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
    model.compile(loss=_huber_loss, optimizer=Adam(lr=learning_rate))

    model.summary()
    return model


def OurModel(input_shape, output_shape, learning_rate, dueling=False):
    X_input = Input(input_shape)
    X = X_input

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(output_shape,))(state_value)

        action_advantage = Dense(output_shape, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(output_shape,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(output_shape, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole D3QN model')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model


def CNNModel(input_shape, action_space, learning_rate, dueling):
    X_input = Input(input_shape)
    X = X_input

    X = Conv2D(64, 5, strides=(3, 3), padding="valid", input_shape=input_shape, activation="relu",
               data_format="channels_first")(X)
    X = Conv2D(64, 4, strides=(2, 2), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Conv2D(64, 3, strides=(1, 1), padding="valid", activation="relu", data_format="channels_first")(X)
    X = Flatten()(X)
    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole PER D3QN CNN model')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=learning_rate, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model
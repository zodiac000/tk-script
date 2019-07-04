from tensorflow.keras import datasets, layers, models
import tensorflow.keras.backend as K
# from tensorflow.losses import mean_squared_error
import tensorflow as tf
import pdb

# input_shape = 324
input_shape = 224

# Define custom loss
def custom_loss():

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
        x_true, y_true, f_true, x_delta, y_delta = y_true[0], y_true[1], y_true[2], y_true[3], y_true[4]
        x_pred, y_pred, f_pred = y_pred[0], y_pred[1], y_pred[2]
        if (x_true - x_delta).eval() < 0:
            print("000000000000000000000000")
        # return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)
        # return K.mean(K.square(x_true - x_pred) + K.square(y_true - y_pred), axis=-1)
        return tf.losses.mean_squared_error(x_true, x_pred) + tf.losses.mean_squared_error(y_true, y_pred)
   
    # Return a function
    return loss


def create_model():
    model = models.Sequential()
    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, (3, 3), input_shape=(input_shape, input_shape, 1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    # model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(optimizer='adam',
              # loss='mse',
              loss=custom_loss(),
              metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = create_model()


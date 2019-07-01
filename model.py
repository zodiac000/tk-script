from tensorflow.keras import datasets, layers, models

def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 1)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Dropout(0.5))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(2, activation='sigmoid'))

    model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

    return model


if __name__ == "__main__":
    model = create_model()


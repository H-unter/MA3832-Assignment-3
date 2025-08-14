from tensorflow import keras


def build_model(input_shape=(512, 512, 3), num_classes=2, dense=128, dropout=0.2):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1./255)(inputs)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(dense, activation="relu")(x)
    x = keras.layers.Dropout(dropout)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)
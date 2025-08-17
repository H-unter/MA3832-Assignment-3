from tensorflow import keras


def build_model( # below are default values, but these are all overritable/tunable
    input_shape=(512, 512, 3),
    conv1_filters=32,
    conv2_filters=64,
    dense_units=128,
    use_dropout=True,
    dropout_rate=0.2,
    pooling="max", # "max" or "avg" for local pooling and final global pooling
):
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Rescaling(1.0 / 255)(inputs)

    # conv block 1
    x = keras.layers.Conv2D(int(conv1_filters), 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D()(x) if pooling == "max" else keras.layers.AveragePooling2D()(x)

    # conv block 2
    x = keras.layers.Conv2D(int(conv2_filters), 3, activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D()(x) if pooling == "max" else keras.layers.AveragePooling2D()(x)
    if pooling == "max":
        x = keras.layers.GlobalMaxPooling2D()(x)
    else:
        x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(int(dense_units), activation="relu")(x)
    if use_dropout:
        x = keras.layers.Dropout(float(dropout_rate))(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)
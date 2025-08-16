from tensorflow import keras
from tensorflow.keras import layers, models

def build_transfer_model(input_shape=(224, 224, 3), dropout_rate=0.2, unfreeze_fraction=0.0):
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape, pooling="avg"
    )
    base.trainable = False
    if unfreeze_fraction > 0:
        n = len(base.layers)
        k = max(1, int(round(n * float(unfreeze_fraction))))
        for layer in base.layers[-k:]:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    inputs = keras.Input(shape=input_shape)
    x = inputs                         # keep values in [0,255] float
    x = base(x)                        # EfficientNet applies its own Rescaling inside
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return models.Model(inputs, outputs)
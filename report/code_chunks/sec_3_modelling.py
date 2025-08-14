
def model_builder(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(512, 512, 3)))  # Full-size image input

    # input downscaling for faster training
    model.add(layers.Resizing(256, 256, interpolation="bilinear"))


    # Conv Layer 1
    model.add(layers.Conv2D(
        filters=hp.Int('filters1', 32, 128, step=16),
        kernel_size=hp.Choice('kernel1', [3, 5]),
        activation='relu',
        padding='same'
    ))
    layer_1_pool_type = hp.Choice("layer_1_pool_type", ["max", "avg"])
    if layer_1_pool_type == "max":
        model.add(layers.MaxPooling2D())
    elif layer_1_pool_type == "avg":
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    
    if hp.Boolean('use_dropout1'):
        model.add(layers.Dropout(hp.Float('dropout1', 0.1, 0.5)))



    # Conv Layer 2
    model.add(layers.Conv2D(
        filters=hp.Int('filters2', 32, 128, step=16),
        kernel_size=hp.Choice('kernel2', [3, 5]),
        activation='relu',
        padding='same'
    ))
    layer_2_pool_type = hp.Choice("layer_2_pool_type", ["max", "avg"])
    if layer_2_pool_type == "max":
        model.add(layers.MaxPooling2D())
    elif layer_2_pool_type == "avg":
        model.add(layers.AveragePooling2D(pool_size=(2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(
        units=hp.Int('dense_units', 64, 256, step=32),
        activation='relu'
    ))

    if hp.Boolean('use_dropout2'):
        model.add(layers.Dropout(hp.Float('dropout2', 0.1, 0.5)))

    model.add(layers.Dense(1, activation='sigmoid'))

    # Optimiser and learning rate
    learning_rate = hp.Float('lr', 1e-5, 1e-2, sampling='log')
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    optimizer_classes = {'adam': keras.optimizers.Adam, 'rmsprop': keras.optimizers.RMSprop}

    model.compile(
        optimizer=optimizer_classes[optimizer](learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=50,
    factor=3, # how many models to try at each stage how aggressively to reduce the number of models
    directory='cnn_tuning',
    project_name='hyperband_test'
)
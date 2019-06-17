from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping


def get_callbacks(config):
    callbacks = [
        ModelCheckpoint(config.chkpnt_weights_path,
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='min',
                        period=5),

        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=max(5, config.patience / 10),
                          verbose=1, mode='min', min_delta=1e-5, cooldown=0, min_lr=0),

        TensorBoard(log_dir=config.log_path, histogram_freq=0,
                    batch_size=config.batch_size,
                    write_graph=True,
                    write_grads=True,
                    write_images=True)]

    if config.patience > 0:
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0.0001,
                                       patience=config.patience,
                                       verbose=1))

    return callbacks

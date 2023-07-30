import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, Nadam
from numpy.random import seed

# !/usr/bin/env python
# Needed to set seed for random generators for making reproducible experiments

seed(1)
tf.compat.v1.set_random_seed(1)


class Unet(object):
    def __init__(self, params, should_train=True):
        # Seed for the random generators
        self.seed = 1
        # Find the model you would like
        # model_name = get_model_name(params)

        # Find the number of classes and bands
        if params["collapse_cls"]:
            n_cls = 1
        else:
            n_cls = np.size(params["cls"])
        n_bands = np.size(params["bands"])

        # Create the model in keras
        # model_name= this.
        if params["num_gpus"] == 1 and should_train:
            self.model = self.__create_inference__(n_bands, n_cls, params)  # initialize the model

        else:
            # with tf.device("/cpu:0"):
            self.model = self.__create_inference__(n_bands, n_cls, params)  # initialize the model on the CPU

    def __create_inference__(self, n_bands, n_cls, params):
        # Note about BN and dropout: https://stackoverflow.com/questions/46316687/how-to-include-batch-normalization-in-non-sequential-keras-model
        # get_custom_objects().update({'swish': Activation(swish)})
        inputs = tf.keras.layers.Input((params["patch_size"], params["patch_size"], n_bands))
        # -----------------------------------------------------------------------
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(inputs)
        conv1 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv1) if params["use_batch_norm"] else conv1
        conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv1)
        conv1 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv1) if params["use_batch_norm"] else conv1
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
        # -----------------------------------------------------------------------
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(pool1)
        conv2 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv2) if params["use_batch_norm"] else conv2
        conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv2)
        conv2 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv2) if params["use_batch_norm"] else conv2
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        # -----------------------------------------------------------------------
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(pool2)
        conv3 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv3) if params["use_batch_norm"] else conv3
        conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv3)
        conv3 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv3) if params["use_batch_norm"] else conv3
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
        # -----------------------------------------------------------------------
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(pool3)
        conv4 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv4) if params["use_batch_norm"] else conv4
        conv4 = tf.keras.layers.Conv2D(256, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv4)
        conv4 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv4) if params["use_batch_norm"] else conv4
        pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
        # -----------------------------------------------------------------------
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(pool4)
        conv5 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv5) if params["use_batch_norm"] else conv5
        conv5 = tf.keras.layers.Conv2D(512, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv5)
        conv5 = BatchNormalization(momentum=params["batch_norm_momentum"])(conv5) if params["use_batch_norm"] else conv5
        # -----------------------------------------------------------------------
        up6 = tf.keras.layers.Concatenate(axis=3)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv5), conv4])
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(up6)
        conv6 = tf.keras.layers.Dropout(params["dropout"])(conv6) if not params["dropout_on_last_layer_only"] else conv6
        conv6 = tf.keras.layers.Conv2D(256, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv6)
        conv6 = tf.keras.layers.Dropout(params["dropout"])(conv6) if not params["dropout_on_last_layer_only"] else conv6
        # -----------------------------------------------------------------------
        up7 = tf.keras.layers.Concatenate(axis=3)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv6), conv3])
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(up7)
        conv7 = tf.keras.layers.Dropout(params["dropout"])(conv7) if not params["dropout_on_last_layer_only"] else conv7
        conv7 = tf.keras.layers.Conv2D(128, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv7)
        conv7 = tf.keras.layers.Dropout(params["dropout"])(conv7) if not params["dropout_on_last_layer_only"] else conv7
        # -----------------------------------------------------------------------
        up8 = tf.keras.layers.Concatenate(axis=3)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv7), conv2])
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(up8)
        conv8 = tf.keras.layers.Dropout(params["dropout"])(conv8) if not params["dropout_on_last_layer_only"] else conv8
        conv8 = tf.keras.layers.Conv2D(64, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv8)
        conv8 = tf.keras.layers.Dropout(params["dropout"])(conv8) if not params["dropout_on_last_layer_only"] else conv8
        # -----------------------------------------------------------------------
        up9 = tf.keras.layers.Concatenate(axis=3)([tf.keras.layers.UpSampling2D(size=(2, 2))(conv8), conv1])
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(up9)
        conv9 = tf.keras.layers.Dropout(params["dropout"])(conv9) if not params["dropout_on_last_layer_only"] else conv9
        conv9 = tf.keras.layers.Conv2D(32, (3, 3), activation=params["activation_func"], padding='same',
                                       kernel_regularizer=regularizers.l2(params["L2reg"]))(conv9)
        conv9 = tf.keras.layers.Dropout(params["dropout"])(conv9)
        # -----------------------------------------------------------------------
        clip_pixels = np.int32(params["overlap"] / 2)  # Only used for input in Cropping2D function on next line
        crop9 = tf.keras.layers.Cropping2D(cropping=((clip_pixels, clip_pixels), (clip_pixels, clip_pixels)))(conv9)
        # -----------------------------------------------------------------------
        conv10 = tf.keras.layers.Conv2D(n_cls, (1, 1), activation='sigmoid')(crop9)
        # -----------------------------------------------------------------------
        model = Model(inputs=inputs, outputs=conv10)

        return model

    def train(self, params):
        #     # Define callbacks
        csv_logger, model_checkpoint, reduce_lr, tensorboard, early_stopping = get_callbacks(params)
        used_callbacks = [csv_logger, model_checkpoint, tensorboard]
        if params["reduce_lr"]:
            used_callbacks.append(reduce_lr)
        if params["early_stopping"]:
            used_callbacks.append(early_stopping)
        # Configure optimizer (use Nadam or Adam and 'binary_crossentropy' or jaccard_coef_loss)
        if params["optimizer"] == 'Adam':
            if params["loss_func"] == 'binary_crossentropy':
                self.model.compile(optimizer=Adam(lr=params["learning_rate"], decay=params["decay"], amsgrad=True),
                                   loss='binary_crossentropy',
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
            elif params["loss_func"] == 'jaccard_coef_loss':
                self.model.compile(optimizer=Adam(lr=params["learning_rate"], decay=params["decay"], amsgrad=True),
                                   loss=jaccard_coef_loss,
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
        elif params["optimizer"] == 'Nadam':
            if params["loss_func"] == 'binary_crossentropy':
                self.model.compile(optimizer=Nadam(lr=params["learning_rate"]),
                                   loss='binary_crossentropy',
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])
            elif params["loss_func"] == 'jaccard_coef_loss':
                self.model.compile(optimizer=Nadam(lr=params["learning_rate"]),
                                   loss=jaccard_coef_loss,
                                   metrics=['binary_crossentropy', jaccard_coef_loss, jaccard_coef,
                                            jaccard_coef_thresholded, 'accuracy'])

        # Create generators
        image_generator = ImageSequence(params, shuffle=True, seed=self.seed,
                                        augment_data=params["affine_transformation"])
        val_generator = ImageSequence(params, shuffle=True, seed=self.seed,
                                      augment_data=params["affine_transformation"],
                                      validation_generator=True)

        # Do the training
        print('------------------------------------------')
        print('Start training:')
        history = self.model.fit_generator(image_generator,
                                           epochs=params["epochs"],
                                           # steps_per_epoch=params["steps_per_epoch"],
                                           verbose=1,
                                           workers=4,
                                           max_queue_size=16,
                                           use_multiprocessing=True,
                                           shuffle=True,
                                           callbacks=used_callbacks,
                                           validation_data=val_generator,
                                           validation_steps=None)

        # Save the weights (append the val score in the name)
        # There is a bug with multi_gpu_model (https://github.com/kuza55/keras-extras/issues/3), hence model.layers[-2]
        model_name = get_model_name(params)
        params["num_gpus"] = 1
        if params["num_gpus"] != 1:
            self.model = self.model.layers[-2]
            self.model.save_weights(params.project_path + 'models/Unet/' + model_name)
            # self.model = multi_gpu_model(self.model, gpus=params.num_gpus)  # Make it run on multiple GPUs
        else:
            model.model.save_weights(params["project_path"] + 'models/Unet/' + get_model_name(params))
        return history

    def predict(self, img, n_bands, n_cls, num_gpus, params):
        # Predict batches of patches
        print("image shape:", np.shape(img))
        patches = np.shape(img)[0]  # Total number of patches
        patch_batch_size = 128

        # Do the prediction
        predicted = np.zeros(
            (patches, params["patch_size"] - params["overlap"], params["patch_size"] - params["overlap"], n_cls))
        print("patches:", patches)
        for i in range(0, patches, patch_batch_size):
            print("patches loop", patches)
            tf.config.run_functions_eagerly(True)
            predicted[i:i + patch_batch_size, :, :, :] = self.model.predict(img[i:i + patch_batch_size, :, :, :])

        return predicted


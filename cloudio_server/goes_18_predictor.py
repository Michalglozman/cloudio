import os
import os.path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tifffile as tiff
from PIL import Image
from keras import regularizers
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from numpy.random import seed
from sklearn.metrics import confusion_matrix
from dotenv import load_dotenv
load_dotenv('.env')

mpl.use('agg')


def get_params():

    HParams = {
        "modelID": '180609113138',
        "num_gpus": 1,
        "optimizer": 'Adam',
        "loss_func": 'binary_crossentropy',
        "activation_func": 'elu',  # https://keras.io/activations/
        "initialization": 'glorot_uniform',  # Initialization of layers
        "use_batch_norm": True,
        "dropout_on_last_layer_only": False,
        "early_stopping": True,  # Use early stopping in optimizer
        "reduce_lr": False,  # Reduce learning rate during training
        "save_best_only": False,  # Save only best step in each training epoch
        "use_ensemble_learning": False,  # Not implemented at the moment
        "ensemble_method": 'Bagging',
        "learning_rate": 1e-4,
        "dropout": 0.2,  # Must be written as float for parser to work
        "L1reg": 0.,  # Must be written as float for parser to work
        "L2reg": 1e-4,  # Must be written as float for parser to work
        "L1L2reg": 0.,  # Must be written as float for parser to work
        "decay": 0.,  # Must be written as float for parser to work
        "batch_norm_momentum": 0.7,  # Momentum in batch normalization layers
        "threshold": 0.46,  # Threshold to create binary cloud mask
        "patch_size": 256,  # Width and height of the patches the img is divided into
        "overlap": 64,  # Overlap in pixels when predicting (to avoid border effects)
        "overlap_train_set": 0,  # Overlap in training data patches (must be even)
        "batch_size": 10,
        "steps_per_epoch": None,  # = batches per epoch
        "epochs": 5,
        "norm_method": 'landsat8_biome_normalization',
        "norm_threshold": 65535,  # Threshold for the contrast enhancement
        "cls": ['cloud', 'fill'],
        "collapse_cls": True,
        "affine_transformation": True,  # Regular data augmentation
        "brightness_augmentation": False,  # Experimental data augmentation
        # Collapse classes to one binary mask (False => multi_cls model)
        # TODO: IF YOU CHOOSE BAND 8, IT DOES NOT MATCH THE .npy TRAINING DATA
        "bands": [0, 1, 2],  # Band 8 is the panchromatic band Get
        # absolute path of the project (https://stackoverflow.com/questions/50499
        # /how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing)
        "project_path": f"{os.getenv('STORAGE_PATH')}/train_again_try/",
        "project_mounted_path": os.getenv('STORAGE_PATH'),
        "satellite": 'Landsat8',
    }
    return HParams


seed(1)
tf.compat.v1.set_random_seed(1)


class Unet(object):
    def __init__(self, params, should_train=False):
        # Seed for the random generators
        self.seed = 1

        # Find the number of classes and bands
        if params["collapse_cls"]:
            n_cls = 1
        else:
            n_cls = np.size(params["cls"])
        n_bands = np.size(params["bands"])

        # Create the model in keras
        self.model = self.__create_inference__(n_bands, n_cls, params)  # initialize the model on the CPU
        self.model.load_weights(params["project_path"] + 'reports/Unet/unet_tmp_lr_of_jacob.hdf5')

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

        self.model.compile(optimizer=Adam(lr=params["learning_rate"], decay=params["decay"], amsgrad=True),
                           loss='binary_crossentropy',
                           metrics=['binary_crossentropy', 'accuracy'])

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

        model.model.save_weights(params["project_path"] + 'models/Unet/' + get_model_name(params))
        return history

    def predict(self, img, n_bands, n_cls, num_gpus, params):
        # Predict batches of patches
        patches = np.shape(img)[0]  # Total number of patches
        patch_batch_size = 128

        # Do the prediction
        predicted = np.zeros(
            (patches, params["patch_size"] - params["overlap"], params["patch_size"] - params["overlap"], n_cls))
        # print("patches:" , patches)
        for i in range(0, patches, patch_batch_size):
            # print("patches loop" , patches)
            tf.config.run_functions_eagerly(True)
            predicted[i:i + patch_batch_size, :, :, :] = self.model.predict(img[i:i + patch_batch_size, :, :, :])

        return predicted


def stitch_image(img_patched, n_height, n_width, patch_size, overlap):
    """
    Stitch the overlapping patches together to one large image (the original format)
    """
    isz_overlap = patch_size - overlap  # i.e. remove the overlap

    n_bands = np.size(img_patched, axis=3)

    img = np.zeros((n_width * isz_overlap, n_height * isz_overlap, n_bands))

    # Stitch the patches together
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j

            # Cut out the interior of the patch
            interior_patch = img_patched[id, :, :, :]

            # Define "pixel coordinates" of the patches in the whole image
            xmin = isz_overlap * i
            xmax = isz_overlap * i + isz_overlap
            ymin = isz_overlap * j
            ymax = isz_overlap * j + isz_overlap

            # Insert the patch into the stitched image
            img[xmin:xmax, ymin:ymax, :] = interior_patch

    return img


def get_cls(satellite, dataset, cls_string):
    """
    Get the classes from the integer values in the true masks (i.e. 'water' in sen2cor has integer value 3)
    """
    # if satellite == 'Landsat8':
    #   if dataset == 'Biome_gt':
    cls_int = []
    for c in cls_string:
        if c == 'fill':
            cls_int.append(0)
        elif c == 'cloud':
            cls_int.append(255)

    return cls_int


def patch_image(img, patch_size, overlap):
    """
    Split up an image into smaller overlapping patches
    """
    # Add zeropadding around the image (has to match the overlap)
    img_shape = np.shape(img)
    img_padded = np.zeros((img_shape[0] + 2 * patch_size, img_shape[1] + 2 * patch_size, img_shape[2]))

    img_padded[overlap:overlap + img_shape[0], overlap:overlap + img_shape[1], :] = img

    # Find number of patches
    n_width = int((np.size(img_padded, axis=0) - patch_size) / (patch_size - overlap))
    n_height = int((np.size(img_padded, axis=1) - patch_size) / (patch_size - overlap))

    # Now cut into patches
    n_bands = np.size(img_padded, axis=2)
    img_patched = np.zeros((n_height * n_width, patch_size, patch_size, n_bands), dtype=img.dtype)
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j
            # Define "pixel coordinates" of the patches in the whole image
            xmin = patch_size * i - i * overlap
            xmax = patch_size * i + patch_size - i * overlap
            ymin = patch_size * j - j * overlap
            ymax = patch_size * j + patch_size - j * overlap

            # img_patched[id, width , height, depth]
            img_patched[id, :, :, :] = img_padded[xmin:xmax, ymin:ymax, :]

    return img_patched, n_height, n_width  # n_height and n_width are necessary for stitching image back together


def load_product(file_path):
    im = tiff.imread(file_path)

    width, height, _ = im.shape
    # Create RGB image
    img_rgb = np.zeros((width, height, 4))
    # Load the img to be used for prediction
    img = np.zeros((width, height, 3), dtype=np.float32)
    img[:, :, :] = np.nan_to_num(tiff.imread(file_path)[:, :, :3])
    # Create the overlapping pixels mask (Band 1-8 are overlapping, but 9, 10, 11 vary at the border regions)
    return img, img_rgb


def get_model_name(params):
    '''
    Combine the parameters for the model into a string (to name the model file)
    '''
    if params["modelID"]:
        model_name = 'Unet_' + params["satellite"] + '_' + params["modelID"]

    return model_name


def predict_img(model, params, img, n_bands, n_cls, num_gpus):
    # """
    # Run prediction on an full image
    # """
    # Find dimensions
    img_shape = np.shape(img)

    # Normalize the product

    img_patched, n_height, n_width = patch_image(img, patch_size=params["patch_size"], overlap=params["overlap"])

    # Now find all completely black patches and inpaint partly black patches
    indices = []  # Used to ignore completely black patches during prediction
    for i in range(0, np.shape(img_patched)[0]):  # For all patches
        if np.any(img_patched[i, :, :, :] == 0):  # If any black pixels
            if np.mean(img_patched[i, :, :, :] != 0):  # Ignore completely black patches
                indices.append(i)  # Use the patch for prediction
                # Fill in zero pixels using the non-zero pixels in the patch
                for j in range(0, np.shape(img_patched)[3]):  # Loop over each spectral band in the patch
                    # Bands do not always overlap. Use mean of all bands if zero-slice is found, otherwise use
                    # mean of the specific band
                    if np.mean(img_patched[i, :, :, j]) == 0:
                        mean_value = np.mean(img_patched[i, img_patched[i, :, :, :] != 0])
                    else:
                        mean_value = np.mean(img_patched[i, img_patched[i, :, :, j] != 0, j])

                    img_patched[i, img_patched[i, :, :, j] == 0, j] = mean_value
        else:
            indices.append(i)  # Use the patch for prediction

    predicted_patches = np.zeros((np.shape(img_patched)[0],
                                  params["patch_size"] - params["overlap"], params["patch_size"] - params["overlap"],
                                  n_cls))
    predicted_patches[indices, :, :, :] = model.predict(img_patched[indices, :, :, :], n_bands, n_cls, num_gpus, params)

    # Stitch the patches back together
    predicted_stitched = stitch_image(predicted_patches, n_height, n_width, patch_size=params["patch_size"],
                                      overlap=params["overlap"])

    # Now throw away the padded sections from the overlap
    padding = int(params["overlap"] / 2)  # The overlap is over 2 patches, so you need to throw away overlap/2 on each
    predicted_mask = predicted_stitched[padding - 1:padding - 1 + img_shape[0],
                     # padding-1 because it is index in array
                     padding - 1:padding - 1 + img_shape[1],
                     :]

    # Throw away the inpainting of the zero pixels in the individual patches
    # The summation is done to ensure that all pixels are included. The bands do not perfectly overlap (!)
    predicted_mask[np.sum(img, axis=2) == 0] = 0

    return predicted_mask

def save_confusion_matrix(y_test, y_pred, path):
    plt.clf()
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)

    # Add titles to the axis
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticklabels(['Not Cloud', 'Cloud'])
    ax.set_yticklabels(['Not Cloud', 'Cloud'])

    plt.savefig(path)


def visualize_geos_tile(model, file_path, data_path, num_gpus, params):
    # Measure the time it takes to load data

    # Find the number of classes
    if params["collapse_cls"]:
        n_cls = 1
    else:
        n_cls = np.size(params["cls"])
    n_bands = np.size(params["bands"])

    # Load the RGB data for the scene
    img, img_rgb = load_product(data_path + file_path)

    # Get the masks
    cls = get_cls('Landsat8', 'Biome_gt', params["cls"])
    # Get the predicted mask
    predicted_mask = predict_img(model, params, img, n_bands, n_cls, num_gpus)

    predicted_binary_mask = predicted_mask >= np.float32(0.5)
    y_predicted = predicted_binary_mask.flatten()

    filename = os.path.splitext(file_path)[0]
    name_parts = filename.split("_")

    result = "_".join(name_parts[:-1])
    print(result)
    # Load the true classification mask
    mask_true = tiff.imread(
        f'{data_path}/' + result + '_dqf_mask.tif')  # The 30 m is the native resolution
    mask_true[mask_true == 2] = 1
    y_true = mask_true.flatten()

    # report = classification_report(y_true, y_predicted, output_dict=True)
    # df_report = pd.DataFrame(report).transpose()
    data_output_path = f'{data_path}'
    os.makedirs(data_output_path, exist_ok=True)

    # print("++===================================================++")
    # report = classification_report(y_true, y_predicted)
    # print(report)
    # print("++===================================================++")
    # df_report.to_csv(f"{data_output_path}/output/confussion_{result}.csv")
    #save_confusion_matrix(y_true, y_predicted, f"{params['project_mounted_path']}/train_again_try/output/confussion_{result}.jpg")

    predicted_mask_rgb = np.zeros(np.shape(img_rgb))
    model_name = get_model_name(params)
    for i, c in enumerate(params["cls"]):
        # Convert predicted mask to RGB and save all masks (use PIL to save as it is much faster than matplotlib)

        # UNCOMMENT BELOW TO SAVE THRESHOLDED MASK
        threshold = 0.6
        predicted_mask_rgb[:, :, 0] = (predicted_mask[:, :, 0])  # Color coding for the mask
        predicted_mask_rgb[:, :, 1] = predicted_mask[:, :, i]
        predicted_mask_rgb[:, :, 2] = (predicted_mask[:, :, i] >= np.float32(0.5)) * 255

        res_path = data_output_path + file_path.split("_")[0] + '_predicted.tif'
        Image.fromarray(np.uint8(predicted_mask_rgb)).save(res_path)
        overlay_path = overlay_images(res_path, data_path, file_path)
        predicted_path = print_predicted_image(res_path, filename, data_path)
        return predicted_path, overlay_path


def overlay_images(res_path,data_path,file_path,color=(255,0,0)):
  img_test = tiff.imread(data_path + file_path)
  img_test = (img_test * 255).astype(np.uint8)  # convert to 8-bit unsigned integer
  img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
  img_test1 = cv2.cvtColor(img_test, cv2.COLOR_RGB2GRAY)
  img_prediction = tiff.imread(res_path)
  # Convert images to numpy arrays if they are not already
  if isinstance(img_prediction, Image.Image):
    rgb_array = np.array(img_prediction)
  else:
    rgb_array = img_prediction
  if len(img_test1.shape) == 2 or img_test1.shape[2] == 1:
    # Convert image to 8-bit depth
    img_8bit = cv2.convertScaleAbs(img_test1)
    # Convert grayscale image to RGB image
    grayscale_array_rgb  = cv2.cvtColor(img_8bit, cv2.COLOR_GRAY2RGB)

    overlaid_image = grayscale_array_rgb.copy()
  else:
    overlaid_image = img_test1.copy()

  mask = (rgb_array > 0).any(axis=-1)
  overlaid_image[mask] = color

  overlaid_image[mask] = color
  plt.figure(figsize=(6, 6))
  plt.imshow(overlaid_image, cmap='gray')
  plt.axis('off')
  plt.title('Image Prediction Overlay')
  plot_path = f'{data_path}/{file_path.split("_")[0]}_mask_plot.png'
  plt.savefig(plot_path)
  plt.close()
  return plot_path

def print_predicted_image(res_path, file_path, data_output_path):
    img_prediction = tiff.imread(res_path)
    img_prediction = cv2.cvtColor(img_prediction, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_prediction, cmap='gray')
    plt.axis('off')
    plt.title('Predicted Image')
    plot_path = os.path.join(data_output_path, f'{file_path.split("_")[0]}_predicted_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def predict(data_path, filename):
    params = get_params()
    model = Unet(params)

    predicted_mask_path = visualize_geos_tile(model, filename + "_4bands.tif", data_path, 1, params)
    return predicted_mask_path

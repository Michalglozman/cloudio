import tensorflow as tf
import inspect
import time
import datetime
import os
import random
import numpy as np
import tifffile as tiff
import cv2
from PIL import Image
from numpy.random import seed
import os.path
import threading
from keras import backend as K
from keras.backend import binary_crossentropy
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.utils import Sequence
import shutil
from keras.models import Model
from keras.layers import Layer
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras import regularizers
from keras.optimizers import Adam, Nadam
import matplotlib.pyplot as plt
import tifffile as tiff
import matplotlib as mpl

mpl.use('agg')
def get_params():
    HParams = {
        "modelID": '180609113138',
        "num_gpus": 1,
        "optimizer": 'Adam',
        "loss_func": 'binary_crossentropy',
        "activation_func": 'elu',  # https://keras.io/activations/
        "initialization": 'glorot_uniform',  # Initialization of layers
        "use_batch_norm": True, "dropout_on_last_layer_only": True,
        "early_stopping": True,  # Use early stopping in optimizer
        "reduce_lr": False,  # Reduce learning rate during training
        "save_best_only": False,  # Save only best step in each training epoch
        "use_ensemble_learning": False,  # Not implemented at the moment
        "ensemble_method": 'Bagging', "learning_rate": 1e-4,
        "dropout": 0.2,  # Must be written as float for parser to work
        "L1reg": 0.,  # Must be written as float for parser to work
        "L2reg": 1e-4,  # Must be written as float for parser to work
        "L1L2reg": 0.,  # Must be written as float for parser to work
        "decay": 0.,  # Must be written as float for parser to work
        "batch_norm_momentum": 0.7,  # Momentum in batch normalization layers
        "threshold": 0.46,  # Threshold to create binary cloud mask
        "patch_size": 256,  # Width and height of the patches the img is divided into
        "overlap": 40,  # Overlap in pixels when predicting (to avoid border effects)
        "overlap_train_set": 0,  # Overlap in training data patches (must be even)
        "batch_size": 10, "steps_per_epoch": None,  # = batches per epoch
        "epochs": 5, "norm_method": 'landsat8_biome_normalization',
        "norm_threshold": 65535,  # Threshold for the contrast enhancement
        "cls": ['cloud', 'thin'],
        "collapse_cls": True,
        "affine_transformation": True,  # Regular data augmentation
        "brightness_augmentation": False,  # Experimental data augmentation
        # Collapse classes to one binary mask (False => multi_cls model)
        # TODO: IF YOU CHOOSE BAND 8, IT DOES NOT MATCH THE .npy TRAINING DATA
        "bands": [0, 1, 2],  # Band 8 is the panchromatic band Get
        # absolute path of the project (https://stackoverflow.com/questions/50499
        # /how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing)
        # project_path=os.path.dirname(os.path.abspath(inspect.stack()[-1][1])) +"/")
        "project_path": f"{os.getenv('STORAGE_PATH')}/FinalProject/Cloudio/data/",
        "project_mounted_path": os.getenv('STORAGE_PATH'),
        "satellite": 'Landsat8',
        "train_dataset": 'Biome_gt',  # Training dataset (gt/fmask/sen2cor)
        "test_dataset": 'Biome_gt',  # Test dataset (gt/fmask/sen2cor)
        "split_dataset": True,  # Not used at the moment.
        "train_tiles": __data_split__('Biome_gt')[0],
        "test_tiles": __data_split__('Biome_gt')[1],
    }  # Used for testing if dataset is split,
    return HParams


def __data_split__(dataset):
    if 'Biome' in dataset:
        # For each biome, the top two tiles are 'Clear', then two 'MidClouds', and then two 'Cloudy'
        # NOTE: IT IS IMPORTANT TO KEEP THE ORDER THE SAME, AS IT IS USED WHEN EVALUATING THE 'MIDCLOUDS',
        #       'CLOUDY', AND 'CLEAR' GROUPS
        train_tiles = ['LC80420082013220LGN00',  # Barren
                       'LC81640502013179LGN01',
                       'LC81330312013202LGN00',
                       'LC81570452014213LGN00',
                       'LC81360302014162LGN00',
                       'LC81550082014263LGN00',
                       'LC80070662014234LGN00',  # Forest
                       'LC81310182013108LGN01',
                       'LC80160502014041LGN00',
                       'LC82290572014141LGN00',
                       'LC81170272014189LGN00',
                       'LC81800662014230LGN00',
                       'LC81220312014208LGN00',  # Grass/Crops
                       'LC81490432014141LGN00',
                       'LC80290372013257LGN00',
                       'LC81750512013208LGN00',
                       'LC81220422014096LGN00',
                       'LC81510262014139LGN00',
                       'LC80010732013109LGN00',  # Shrubland
                       'LC80750172013163LGN00',
                       'LC80350192014190LGN00',
                       'LC80760182013170LGN00',
                       'LC81020802014100LGN00',
                       'LC81600462013215LGN00',
                       'LC80841202014309LGN00',  # Snow/Ice
                       'LC82271192014287LGN00',
                       'LC80060102014147LGN00',
                       'LC82171112014297LGN00',
                       'LC80250022014232LGN00',
                       'LC82320072014226LGN00',
                       'LC80410372013357LGN00',  # Urban
                       'LC81770262013254LGN00',
                       'LC80460282014171LGN00',
                       'LC81620432014072LGN00',
                       'LC80170312013157LGN00',
                       'LC81920192013103LGN01',
                       'LC80180082014215LGN00',  # Water
                       'LC81130632014241LGN00',
                       'LC80430122014214LGN00',
                       'LC82150712013152LGN00',
                       'LC80120552013202LGN00',
                       'LC81240462014238LGN00',
                       'LC80340192014167LGN00',  # Wetlands
                       'LC81030162014107LGN00',
                       'LC80310202013223LGN00',
                       'LC81080182014238LGN00',
                       'LC81080162013171LGN00',
                       'LC81020152014036LGN00']

        test_tiles = ['LC80530022014156LGN00',  # Barren
                      'LC81750432013144LGN00',
                      'LC81390292014135LGN00',
                      'LC81990402014267LGN00',
                      'LC80500092014231LGN00',
                      'LC81930452013126LGN01',
                      'LC80200462014005LGN00',  # Forest
                      'LC81750622013304LGN00',
                      'LC80500172014247LGN00',
                      'LC81330182013186LGN00',
                      'LC81720192013331LGN00',
                      'LC82310592014139LGN00',
                      'LC81820302014180LGN00',  # Grass/Crops
                      'LC82020522013141LGN01',
                      'LC80980712014024LGN00',
                      'LC81320352013243LGN00',
                      'LC80290292014132LGN00',
                      'LC81440462014250LGN00',
                      'LC80320382013278LGN00',  # Shrubland
                      'LC80980762014216LGN00',
                      'LC80630152013207LGN00',
                      'LC81590362014051LGN00',
                      'LC80670172014206LGN00',
                      'LC81490122013218LGN00',
                      'LC80441162013330LGN00',  # Snow/Ice
                      'LC81001082014022LGN00',
                      'LC80211222013361LGN00',
                      'LC81321192014054LGN00',
                      'LC80010112014080LGN00',
                      'LC82001192013335LGN00',
                      'LC80640452014041LGN00',  # Urban
                      'LC81660432014020LGN00',
                      'LC80150312014226LGN00',
                      'LC81970242013218LGN00',
                      'LC81180382014244LGN00',
                      'LC81940222013245LGN00',
                      'LC80210072014236LGN00',  # Water
                      'LC81910182013240LGN00',
                      'LC80650182013237LGN00',
                      'LC81620582014104LGN00',
                      'LC81040622014066LGN00',
                      'LC81660032014196LGN00',
                      'LC81460162014168LGN00',  # Wetlands
                      'LC81580172013201LGN00',
                      'LC81010142014189LGN00',
                      'LC81750732014035LGN00',
                      'LC81070152013260LGN00',
                      'LC81500152013225LGN00']

        return [train_tiles, test_tiles]


def stitch_image(img_patched, n_height, n_width, patch_size, overlap):
    """
    Stitch the overlapping patches together to one large image (the original format)
    """
    isz_overlap = patch_size - overlap  # i.e. remove the overlap

    n_bands = np.size(img_patched, axis=3)

    img = np.zeros((n_width * isz_overlap, n_height * isz_overlap, n_bands))

    # Define bbox of the interior of the patch to be stitched (not required if using Cropping2D layer in model)
    # xmin_overlap = int(overlap / 2)
    # xmax_overlap = int(patch_size - overlap / 2)
    # ymin_overlap = int(overlap / 2)
    # ymax_overlap = int(patch_size - overlap / 2)

    # Stitch the patches together
    for i in range(0, n_width):
        for j in range(0, n_height):
            id = n_height * i + j

            # Cut out the interior of the patch
            # interior_path = img_patched[id, xmin_overlap:xmax_overlap, ymin_overlap:ymax_overlap, :]
            interior_patch = img_patched[id, :, :, :]

            # Define "pixel coordinates" of the patches in the whole image
            xmin = isz_overlap * i
            xmax = isz_overlap * i + isz_overlap
            ymin = isz_overlap * j
            ymax = isz_overlap * j + isz_overlap

            # Insert the patch into the stitched image
            img[xmin:xmax, ymin:ymax, :] = interior_patch

    return img


def get_model_name(params):
    '''
    Combine the parameters for the model into a string (to name the model file)
    '''
    if params["modelID"]:
        model_name = 'Unet_' + params["satellite"] + '_' + params["modelID"]
    elif params["satellite"] == 'Sentinel-2':
        model_name = 'sentinel2_unet_cls-' + "".join(str(c) for c in params["cls"]) + \
                     '_initmodel-' + params["initial_model"] + \
                     '_collapse' + str(params["collapse_cls"]) + \
                     '_bands' + "".join(str(b) for b in params["bands"]) + \
                     "_lr" + str(params["learning_rate"]) + \
                     '_decay' + str(params["decay"]) + \
                     '_L2reg' + str(params["L2reg"]) + \
                     '_dropout' + str(params["dropout"]) + '.hdf5'
    elif params["satellite"] == 'Landsat8':
        model_name = 'landsat8_unet_cls-' + "".join(str(c) for c in params["cls"]) + \
                     '_collapse' + str(params["collapse_cls"]) + \
                     '_bands' + "".join(str(b) for b in params["bands"]) + \
                     "_lr" + str(params["learning_rate"]) + \
                     '_decay' + str(params["decay"]) + \
                     '_L2reg' + str(params["L2reg"]) + \
                     '_dropout' + str(params["dropout"]) + '.hdf5'
    return model_name


def predict_img(model, params, img, n_bands, n_cls, num_gpus):
    # """
    # Run prediction on an full image
    # """
    # Find dimensions
    img_shape = np.shape(img)

    # Normalize the product
    # img = image_normalizer(img, params, type=params["norm_method"])

    # Patch the image in patch_size * patch_size pixel patches
    img_patched, n_height, n_width = patch_image(img, patch_size=params["patch_size"],
                                                 overlap=params["overlap"])  # ,patch_image["img"]

    # Now find all completely black patches and inpaint partly black patches
    indices = []  # Used to ignore completely black patches during prediction
    use_inpainting = False
    for i in range(0, np.shape(img_patched)[0]):  # For all patches
        if np.any(img_patched[i, :, :, :] == 0):  # If any black pixels
            if np.mean(img_patched[i, :, :, :] != 0):  # Ignore completely black patches
                indices.append(i)  # Use the patch for prediction
                # Fill in zero pixels using the non-zero pixels in the patch
                for j in range(0, np.shape(img_patched)[3]):  # Loop over each spectral band in the patch
                    # Use more advanced inpainting method
                    if use_inpainting:
                        zero_mask = np.zeros_like(img_patched[i, :, :, j])
                        zero_mask[img_patched[i, :, :, j] == 0] = 1
                        inpainted_patch = cv2.inpaint(np.uint8(img_patched[i, :, :, j] * 255),
                                                      np.uint8(zero_mask),
                                                      inpaintRadius=5,
                                                      flags=cv2.INPAINT_TELEA)

                        img_patched[i, :, :, j] = np.float32(inpainted_patch) / 255
                    # Use very simple inpainting method (fill in the mean value)
                    else:
                        # Bands do not always overlap. Use mean of all bands if zero-slice is found, otherwise use
                        # mean of the specific band
                        if np.mean(img_patched[i, :, :, j]) == 0:
                            mean_value = np.mean(img_patched[i, img_patched[i, :, :, :] != 0])
                        else:
                            mean_value = np.mean(img_patched[i, img_patched[i, :, :, j] != 0, j])

                        img_patched[i, img_patched[i, :, :, j] == 0, j] = mean_value
        else:
            indices.append(i)  # Use the patch for prediction

    # Now do the cloud masking (on non-zero patches according to indices)
    # start_time = time.time()
    #  print("predicted_patches argmax",predicted_patches.argmax())
    predicted_patches = np.zeros((np.shape(img_patched)[0],
                                  params["patch_size"] - params["overlap"], params["patch_size"] - params["overlap"],
                                  n_cls))
    predicted_patches[indices, :, :, :] = model.predict(img_patched[indices, :, :, :], n_bands, n_cls, num_gpus, params)

    # exec_time = str(time.time() - start_time)
    # print("Prediction of patches (not including splitting and stitching) finished in: " + exec_time + "s")

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

    # Threshold the prediction
    predicted_binary_mask = predicted_mask >= np.float32(params["threshold"])

    return predicted_mask, predicted_binary_mask


def get_cls(satellite, dataset, cls_string):
    """
    Get the classes from the integer values in the true masks (i.e. 'water' in sen2cor has integer value 3)
    """

    if satellite == 'Landsat8':
        if dataset == 'Biome_gt':
            cls_int = []
            for c in cls_string:
                if c == 'fill':
                    cls_int.append(0)
                elif c == 'shadow':
                    cls_int.append(64)
                elif c == 'clear':
                    cls_int.append(128)
                elif c == 'thin':
                    cls_int.append(192)
                elif c == 'cloud':
                    cls_int.append(255)

        elif dataset == 'Biome_fmask':
            cls_int = []
            for c in cls_string:
                if c == 'fill':
                    cls_int.append(0)
                elif c == 'clear':
                    cls_int.append(1)
                elif c == 'cloud':
                    cls_int.append(2)
                elif c == 'shadow':
                    cls_int.append(3)
                elif c == 'snow':
                    cls_int.append(4)
                elif c == 'water':
                    cls_int.append(5)

        # elif dataset == 'SPARCS_gt':
        #     cls_int = []
        #     for c in cls_string:
        #         if c == 'shadow':
        #             cls_int.append(0)
        #             cls_int.append(1)
        #         elif c == 'water':
        #             cls_int.append(2)
        #             cls_int.append(6)
        #         elif c == 'snow':
        #             cls_int.append(3)
        #         elif c == 'cloud':
        #             cls_int.append(5)
        #         elif c == 'clear':
        #             cls_int.append(4)

        # elif dataset == 'SPARCS_fmask':
        #     cls_int = []
        #     for c in cls_string:
        #         if c == 'fill':
        #             cls_int.append(0)
        #         elif c == 'clear':
        #             cls_int.append(1)
        #         elif c == 'cloud':
        #             cls_int.append(2)
        #         elif c == 'shadow':
        #             cls_int.append(3)
        #         elif c == 'snow':
        #             cls_int.append(4)
        #         elif c == 'water':
        #             cls_int.append(5)
    return cls_int


def image_normalizer(img, params, type='enhance_contrast'):
    # """
    # Clip an image at certain threshold value, and then normalize to values between 0 and 1.
    # Threshold is used for contrast enhancement.
    # """
    type = "landsat8_biome_normalization"
    if type == 'enhance_contrast':  # Enhance contrast of entire image
        # The Sentinel-2 data has 15 significant bits, but normally maxes out between 10000-20000.
        # Here we clip and normalize to value between 0 and 1
        img_norm = np.clip(img, 0, params["norm_threshold"])
        img_norm = img_norm / params["norm_threshold"]

    elif type == 'running_normalization':  # Normalize each band of each incoming image based on that image
        # Based on stretch_n function found at https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
        min_value = 0
        max_value = 1

        lower_percent = 0.02  # Used to discard lower bound outliers
        higher_percent = 0.98  # Used to discard upper bound outliers

        bands = img.shape[2]
        img_norm = np.zeros_like(img)

        for i in range(bands):
            c = np.percentile(img[:, :, i], lower_percent)
            d = np.percentile(img[:, :, i], higher_percent)
            t = min_value + (img[:, :, i] - c) * (max_value - min_value) / (d - c)
            t[t < min_value] = min_value
            t[t > max_value] = max_value
            img_norm[:, :, i] = t

    elif type == 'landsat8_biome_normalization':  # Normalize each band of each incoming image based on Landsat8 Biome
        # Standard deviations used for standardization
        std_devs = 4

        # Normalizes to zero mean and half standard deviation (find values in 'jhj_InspectLandsat8Data' notebook)
        img_norm = np.zeros_like(img)
        img_norm[:, :, 0] = (img[:, :, 0] - 4435) / (std_devs * 1414)
        img_norm[:, :, 1] = (img[:, :, 1] - 4013) / (std_devs * 1385)
        img_norm[:, :, 2] = (img[:, :, 2] - 4112) / (std_devs * 1488)
        # for i, b in enumerate(params["bands"]):
        #     if b == 1:
        #         img_norm[:, :, i] = (img[:, :, i] - 4654) / (std_devs * 1370)
        #     elif b == 2:
        #         img_norm[:, :, i] = (img[:, :, i] - 4435) / (std_devs * 1414)
        #     elif b == 3:
        #         img_norm[:, :, i] = (img[:, :, i] - 4013) / (std_devs * 1385)
        #     elif b == 4:
        #         img_norm[:, :, i] = (img[:, :, i] - 4112) / (std_devs * 1488)
        #     elif b == 5:
        #         img_norm[:, :, i] = (img[:, :, i] - 4776) / (std_devs * 1522)
        #     elif b == 6:
        #         img_norm[:, :, i] = (img[:, :, i] - 2371) / (std_devs * 998)
        #     elif b == 7:
        #         img_norm[:, :, i] = (img[:, :, i] - 1906) / (std_devs * 821)
        #     elif b == 8:
        #         img_norm[:, :, i] = (img[:, :, i] - 18253) / (std_devs * 4975)
        #     elif b == 9:
        #         img_norm[:, :, i] = (img[:, :, i] - 380) / (std_devs * 292)
        #     elif b == 10:
        #         img_norm[:, :, i] = (img[:, :, i] - 19090) / (std_devs * 2561)
        #     elif b == 11:
        #         img_norm[:, :, i] = (img[:, :, i] - 17607) / (std_devs * 2119)

    return img_norm


def patch_image(img, patch_size, overlap):
    """
    Split up an image into smaller overlapping patches
    """
    # TODO: Get the size of the padding right.
    # Add zeropadding around the image (has to match the overlap)
    img_shape = np.shape(img)
    img_padded = np.zeros((img_shape[0] + 2 * patch_size, img_shape[1] + 2 * patch_size, img_shape[2]))
    print("img argmax", img.argmax())
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
            # print("hi from patcbes",n_height,n_width,patch_size,img_padded[xmin:xmax, ymin:ymax, :])

            # Cut out the patches.
            # img_patched[id, width , height, depth]
            img_patched[id, :, :, :] = img_padded[xmin:xmax, ymin:ymax, :]

    return img_patched, n_height, n_width  # n_height and n_width are necessary for stitching image back together


def extract_collapsed_cls(mask, cls):
    """
    Combine several classes to one binary mask
    """
    # Remember to zeroize class 1
    if 1 not in cls:
        mask[mask == 1] = 0

    # Make a binary mask including all classes in cls
    for c in cls:
        mask[mask == c] = 1
    mask[mask != 1] = 0

    return mask


def extract_cls_mask(mask, c):
    """
    Create a binary mask for a class with integer value c (i.e. if class 'cloud' = 255, then change it to 'cloud' = 1)
    """
    # Copy the mask for every iteration (if you set "y=mask", then mask will be overwritten!
    # https://stackoverflow.com/questions/19951816/python-changes-to-my-copy-variable-affect-the-original-variable
    y = np.copy(mask)
    # Remember to zeroize class 1
    if c != 1:
        y[y == 1] = 0

    # Make a binary mask for the specific class
    y[y == c] = 1
    y[y != 1] = 0
    return y


def load_product(file_path, params):
    im = np.nan_to_num(tiff.imread(file_path + '_B2.tif'))
    width, height = im.shape
    # Create RGB image
    img_rgb = np.zeros((width, height, 3))
    # Load the img to be used for prediction
    img = np.zeros((width, height, np.size(params["bands"])), dtype=np.float32)
    img[:, :, 0] = np.nan_to_num(tiff.imread(file_path + '_B2.tif'))
    img[:, :, 1] = np.nan_to_num(tiff.imread(file_path + '_B3.tif'))
    img[:, :, 2] = np.nan_to_num(tiff.imread(file_path + '_B4.tif'))

    # Create the overlapping pixels mask (Band 1-8 are overlapping, but 9, 10, 11 vary at the border regions)

    return img, img_rgb


def visualize_landsat8_tile(model, file_path, data_path, num_gpus, params):
    # Measure the time it takes to load data
    start_time = time.time()

    # Find the number of classes
    if params["collapse_cls"]:
        n_cls = 1
    else:
        n_cls = np.size(params["cls"])
    n_bands = np.size(params["bands"])

    # Load the RGB data for the scene
    img, img_rgb = load_product(data_path + file_path, params)

    # Load the true classification mask
    mask_true = np.nan_to_num(tiff.imread(data_path + file_path + '_qa.tif'))  # The 30 m is the native resolution

    # Get the masks
    cls = get_cls('Landsat8', 'Biome_gt', params["cls"])

    # Create the binary masks
    if params["collapse_cls"]:
        mask_true = extract_collapsed_cls(mask_true, cls)

    else:
        for i, c in enumerate(params["cls"]):
            y = extract_cls_mask(mask_true, cls)

            # Save the binary masks as one hot representations
            mask_true[:, :, i] = y[:, :, 0]

    exec_time = str(time.time() - start_time)
    print("Data loaded in        : " + exec_time + "s")

    # Get the predicted mask
    start_time = time.time()
    predicted_mask, predicted_binary_mask = predict_img(model, params, img, n_bands, n_cls, num_gpus)
    exec_time = str(time.time() - start_time)
    print("Prediction finished in: " + exec_time + "s")

    # Have at least one pixel for each class to avoid issues with the colors in the figure
    mask_true[0, 0] = 0
    mask_true[0, 1] = 1
    predicted_binary_mask[0, 0] = 0
    predicted_binary_mask[0, 1] = 1

    # Save as images
    data_output_path = data_path
    start_time = time.time()
    img_enhanced_contrast = image_normalizer(img_rgb, params, type='enhanced_contrast')
    if not os.path.isfile(data_output_path + '%s-image.tiff' % file_path):
        print("im here")
        Image.fromarray(np.uint8(img_enhanced_contrast * 255)).save(data_output_path + '%s-image.tiff' % file_path)

    predicted_mask_rgb = np.zeros((np.shape(img_enhanced_contrast)))
    model_name = get_model_name(params)
    # for i, c in enumerate(params.cls):
    for i, c in enumerate(params["cls"]):
        # Convert predicted mask to RGB and save all masks (use PIL to save as it is much faster than matplotlib)

        # UNCOMMENT BELOW TO SAVE THRESHOLDED MASK
        threshold = 0.45
        predicted_mask_rgb[:, :, 0] = (predicted_mask[:, :, 0] > threshold) * 254  # Color coding for the mask
        predicted_mask_rgb[:, :, 1] = predicted_mask[:, :, i]
        predicted_mask_rgb[:, :, 2] = predicted_mask[:, :, i]
        predicted_path = data_output_path+ file_path + '_predicted.tif'
        Image.fromarray(np.uint8(predicted_mask_rgb)).save(predicted_path)

        Image.fromarray(np.uint8(256 - ((predicted_mask[:, :, i] > threshold) * 254))).save(
            data_output_path + '%s_%s.tiff' % (file_path, model_name))
        Image.fromarray(np.uint8(predicted_binary_mask[:, :, i])).save(
            data_output_path + '%s_thresholded_%s.tiff' % (file_path, model_name))
        if not os.path.isfile(data_output_path + '%s_cls-%s.tiff' % (file_path, params["cls"])):
            Image.fromarray(np.uint8(mask_true)).save(data_output_path + '%s_true_cls-%s_collapse%s.tiff' % (
            file_path, "".join(str(c) for c in params["cls"]), params["collapse_cls"]))

        exec_time = str(time.time() - start_time)
        print("Images saved in       : " + exec_time + "s")
        img_prediction = cv2.imread(predicted_path)
        img_prediction = cv2.cvtColor(img_prediction, cv2.COLOR_BGR2RGB)
        mask = (img_prediction > [20, 0, 0]).any(axis=2)
        img_masked = tiff.imread(data_path + file_path + '_B3.tif')
        img_masked = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
        img_masked[mask] = [255, 0, 0]
        plt.figure(figsize=(6, 6))
        plt.imshow(img_masked, cmap='gray')
        plt.axis('off')
        plt.title('Image Prediction Overlay')
        plot_path = os.path.join(data_output_path, f'{file_path}_mask_plot.png')
        plt.savefig(plot_path)
        plt.close()

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(img_prediction, cmap='gray')
        plt.title('Predicted Image')
        plot_path = os.path.join(data_output_path, f'{file_path}_predicted_plot.png')
        plt.savefig(plot_path)
        plt.close()
        return os.path.join(data_output_path, f'{file_path}_predicted_plot.png'), os.path.join(data_output_path, f'{file_path}_mask_plot.png')


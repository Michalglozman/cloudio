from helper_funcs import visualize_landsat8_tile
from unet import Unet
import os

# !/usr/bin/env python
# !tf_upgrade_v2 -h


def get_params():
    HParams = {
        "modelID": 'landset_michal_prod',
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
        "threshold": 0.5,  # Threshold to create binary cloud mask
        "patch_size": 256,  # Width and height of the patches the img is divided into
        "overlap": 40,  # Overlap in pixels when predicting (to avoid border effects)
        "overlap_train_set": 0,  # Overlap in training data patches (must be even)
        "batch_size": 40, "steps_per_epoch": None,  # = batches per epoch
        "epochs": 5, "norm_method": 'landsat8_biome_normalization',
        "norm_threshold": 65535,  # Threshold for the contrast enhancement
        "cls": ['cloud', 'thin'],
        "collapse_cls": True,
        "affine_transformation": True,  # Regular data augmentation
        "brightness_augmentation": False,  # Experimental data augmentation
        # Collapse classes to one binary mask (False => multi_cls model)
        # TODO: IF YOU CHOOSE BAND 8, IT DOES NOT MATCH THE .npy TRAINING DATA
        "bands": [2, 3, 4],  # Band 8 is the panchromatic band Get
        # absolute path of the project (https://stackoverflow.com/questions/50499
        # /how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executing)
        # project_path=os.path.dirname(os.path.abspath(inspect.stack()[-1][1])) +"/")
        "project_path": f"{os.getenv('STORAGE_PATH')}/FinalProject/Cloudio/data/",
        "project_mounted_path": os.getenv('STORAGE_PATH'),
        "train_dataset": 'Biome_gt',  # Training dataset (gt/fmask/sen2cor)
        "test_dataset": 'Biome_gt',  # Test dataset (gt/fmask/sen2cor)
        "split_dataset": True,  # Not used at the moment.
        "satellite": 'Landsat8',
        "test_tiles": __data_split__('Biome_gt')}  # Used for testing if dataset is split,
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

        # test_tiles = ['LC82290572014141LGN00',
        #              'LC81080162013171LGN00']

        return [train_tiles, test_tiles]


def predict(data_path, img_path):
    params = get_params()
    model = Unet(params)
    model.model.load_weights(f"{os.getenv('STORAGE_PATH')}/FinalProject/Cloudio/data/reports/Unet/unet_tmp.hdf5")

    predicted_mask_path, masked_path = visualize_landsat8_tile(model, img_path, data_path, 1, params)
    return predicted_mask_path, masked_path


if __name__ == "__main__":
    predicted_mask_path = predict()
    print(predicted_mask_path)
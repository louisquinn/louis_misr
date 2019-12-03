"""
This script will do the following for test/train sets of PROBAV
- Load LR, HR and QM images
- Pre-process with image-processing
    - Inspect quality maps and fill low quality pixels with mean, median or mode
        of all other LR images that had high quality corresponding pixels
    - Normalize each imageset of LR image by subtracting the channel mean of all clear pixels
- Register the LR images:
    - Use the LR image with highest clearance for reference
    - Use the `cnn_registration` module for registration
- Dump to a new output folder which matches the format of the input. QMs not needed.
"""
import os
import sys
import cmapy
import shutil
import logging
import argparse
from glob import glob

# Third-party
import cv2
import scipy
import skimage
import numpy as np
from tqdm import tqdm
from scipy.interpolate import Rbf

# CNN registration module
import submodules.cnn_registration.src.Registration as CNNRegistration

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help='Description'
)
parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help='Description'
)
parser.add_argument(
    "--do_registration",
    action="store_true",
    default=False,
    required=False,
    help='Description'
)
parser.add_argument(
    "--channel_norm_method",
    type=str,
    choices=["mean", "median", "mode"],
    default="mean",
    required=False,
    help='Description'
)
args = parser.parse_args()

# -------------------------- SETUP PARAMETERS -----------------------------
# Set up the logger
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(os.path.dirname(os.path.realpath(__file__)))
logger.setLevel(logging.INFO)

lr_image_prefix = "LR"
hr_image_prefix = "HR"
qm_image_prefix = "QM"
sm_image_prefix = "SM"


# -------------------------- CLASS DEFINITIONS ----------------------------
class ScenePreprocessor:
    def __init__(self, scene_packet, cnn_registration_module, do_registration=False):
        self.scene_packet = scene_packet
        self.cnn_registration_module = cnn_registration_module
        self.do_registration = do_registration

        # Scene info
        self.set_name = None
        self.scene_path = None
        self._extract_scene_info()

        # Preprocessing ops
        self.channel_norm_method = args.channel_norm_method
        self.agg_ops = {
            'mean': lambda i: np.nanmean(i, axis=0),
            'median': lambda i: np.nanmedian(i, axis=0),
            'mode': lambda i: scipy.stats.mode(i, axis=0, nan_policy='omit').mode[0]
        }

    def _extract_scene_info(self):
        self.set_name = self.scene_packet["set_name"]
        self.scene_path = self.scene_packet["scene_path"]

    def preprocess_image_processing(self):
        """
        Takes inspiration from `def central_tendency` from:
        https://github.com/cedricoeldorf/proba-v-super-resolution-challenge/blob/master/embiggen/aggregate.py

        - Load all images (HR and LR)
        - Fill obscured pixels with mean of all images in those regions
        - Subtract mean of all final images for normalization
        :return:
        """
        def lowres_image_iterator(path, img_as_float=True):
            """
            Iterator over all of a scene's low-resolution images (LR*.png) and their
            corresponding status maps (QM*.png).

            Returns at each iteration a `(l, c)` tuple, where:
            * `l`: matrix with the loaded low-resolution image (values as np.uint16 or
                   np.float64 depending on `img_as_float`),
            * `c`: the image's corresponding "clear pixel?" boolean mask.

            Scenes' image files are described at:
            https://kelvins.esa.int/proba-v-super-resolution/data/
            """
            path = path if path[-1] in {'/', '\\'} else (path + '/')
            for f in glob(path + 'LR*.png'):
                q = f.replace('LR', 'QM')
                l = skimage.io.imread(f).astype(np.uint16)
                c = skimage.io.imread(q).astype(np.bool)
                if img_as_float:
                    l = skimage.img_as_float64(l)
                yield (l, c)

        def create_agg_image():
            lr_images_and_qms = lowres_image_iterator(self.scene_path, img_as_float=False)

            # Predefine some params we will need
            obsc = []

            # Grab the obscured regions
            for (l, c) in lr_images_and_qms:
                l = l.astype(np.float32)

                # Keep track of the values at obscured pixels
                o = l.copy()
                o[c] = np.nan
                obsc.append(o)

                # Replace the values at obscured pixels as NaNs
                l[~c] = np.nan
                lr_images.append(l)

            # Create an aggregated image with filled pixels at non-clear areas
            agg_image = agg_op(lr_images)
            some_clear = np.isnan(obsc).any(axis=0)
            obsc = agg_op(obsc)
            obsc[some_clear] = 0.0
            np.nan_to_num(agg_image, copy=False)
            agg_image += obsc
            return agg_image.astype(np.uint16)

        # Dereference the aggregation op
        agg_op = self.agg_ops[self.channel_norm_method]
        lr_images = []

        # Create the aggregated image
        agg_ref_image = create_agg_image()

        # Now we want to iterate over the lr images and hr images,
        # read the quality map and fill those pixels with the corresponding ones from the agg image
        for scene_packet in self.scene_packet["lr_image_packets"]:
            lr_image = cv2.imread(scene_packet["image_path"], cv2.IMREAD_UNCHANGED)
            qm = cv2.imread(scene_packet["qm_image_path"], cv2.IMREAD_UNCHANGED)
            final_image = np.where(qm == 0, agg_ref_image, lr_image)

            # Store this image in the scene packet dict
            scene_packet["qm_filled_image"] = final_image

            # TODO: for getting image output examples
            # lr_image_disp = cv2.applyColorMap(
            #     cv2.normalize(lr_image, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8),
            #     cmapy.cmap("viridis")
            # )
            # qm_image_disp = qm.copy()
            # hr_image = cv2.applyColorMap(
            #     cv2.normalize(
            #         cv2.imread(self.scene_packet["hr_image_packet"]["image_path"], cv2.IMREAD_UNCHANGED),
            #         None, 0, 255, norm_type=cv2.NORM_MINMAX
            #     ).astype(np.uint8),
            #     cmapy.cmap("viridis")
            # )
            # final_image_disp = cv2.applyColorMap(
            #     cv2.normalize(final_image, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8),
            #     cmapy.cmap("viridis")
            # )
            # agg_image_disp = cv2.applyColorMap(
            #     cv2.normalize(agg_ref_image, None, 0, 255, norm_type=cv2.NORM_MINMAX).astype(np.uint8),
            #     cmapy.cmap("viridis")
            # )
            #
            # cv2.imshow("orig", cv2.resize(lr_image_disp, None, fx=3, fy=3))
            # cv2.imshow("qm", cv2.resize(qm_image_disp, None, fx=3, fy=3))
            # cv2.imshow("agg", cv2.resize(agg_image_disp, None, fx=3, fy=3))
            # cv2.imshow("final", cv2.resize(final_image_disp, None, fx=3, fy=3))
            # cv2.imshow("hr image", hr_image)
            # cv2.waitKey(0)

    def preprocess_image_registration(self):
        """
        Register the lr images using the first image as reference
        :return:
        """
        if self.do_registration:
            # Grab the reference image
            lr_ref_image = self.scene_packet["lr_image_packets"][0]["qm_filled_image"]
            lr_ref_image_rgb = cv2.cvtColor(lr_ref_image, cv2.COLOR_GRAY2RGB)

            # Loop through the other lr images and register
            for scene_packet in self.scene_packet["lr_image_packets"][1:]:
                lr_image = scene_packet["qm_filled_image"]
                lr_image_rgb = cv2.cvtColor(lr_image, cv2.COLOR_GRAY2RGB)

                # Register and warp the lr image
                x, y, z = self.cnn_registration_module.register(lr_ref_image_rgb, lr_image_rgb)
                lr_image_reg = self._tps_warp(y, z, lr_image_rgb, lr_ref_image_rgb.shape)

                # Show the results
                # cv2.namedWindow("ref", cv2.WINDOW_NORMAL)
                # cv2.namedWindow("orig", cv2.WINDOW_NORMAL)
                # cv2.namedWindow("warp", cv2.WINDOW_NORMAL)
                # cv2.imshow("ref", cv2.normalize(lr_ref_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imshow("orig", cv2.normalize(lr_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.imshow("warp", cv2.normalize(lr_image_reg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                # cv2.waitKey(0)
                #
                # Save the warped output image
                scene_packet["final_image"] = lr_image_reg
        else:
            for scene_packet in self.scene_packet["lr_image_packets"]:
                scene_packet["final_image"] = scene_packet["qm_filled_image"]

    def output_images_to_file(self):
        # Dump the LR images first
        for lr_image_packet in self.scene_packet["lr_image_packets"]:
            output_filename = os.path.join(
                self.scene_packet["output_scene_path"],
                os.path.basename(lr_image_packet["image_path"])
            )
            cv2.imwrite(output_filename, lr_image_packet["final_image"])

        # Copy over the HR image
        if self.set_name == "train":
            output_filename = os.path.join(
                self.scene_packet["output_scene_path"],
                os.path.basename(self.scene_packet["hr_image_packet"]["image_path"])
            )
            shutil.copy(self.scene_packet["hr_image_packet"]["image_path"], output_filename)
            shutil.copy(self.scene_packet["hr_image_packet"]["qm_image_path"], self.scene_packet["output_scene_path"])

    @staticmethod
    def _tps_warp(Y, T, Y_image, out_shape):
        """
        From cnn_registration module
        :param T:
        :param Y_image:
        :param out_shape:
        :return:
        """
        Y_height, Y_width = Y_image.shape[:2]
        T_height, T_width = out_shape[:2]

        i_func = Rbf(T[:, 0], T[:, 1], Y[:, 0], function='thin-plate')
        j_func = Rbf(T[:, 0], T[:, 1], Y[:, 1], function='thin-plate')

        iT, jT = np.mgrid[:T_height, :T_width]
        iT = iT.flatten()
        jT = jT.flatten()
        iY = np.int_(i_func(iT, jT))
        jY = np.int_(j_func(iT, jT))

        keep = np.logical_and(iY >= 0, jY >= 0)
        keep = np.logical_and(keep, iY < Y_height)
        keep = np.logical_and(keep, jY < Y_width)
        iY, jY, iT, jT = iY[keep], jY[keep], iT[keep], jT[keep]

        out_image = np.zeros(out_shape, dtype=Y_image.dtype)
        out_image[iT, jT, :] = Y_image[iY, jY, :]

        return out_image


# ------------------------- FUNCTION DEFINITIONS --------------------------
def check_and_create_paths():
    if not os.path.exists(args.dataset_path):
        logger.error("Path does not exist: %s" % args.dataset_path)
        raise AssertionError
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


def log_dataset(dataset_path, output_path):
    dataset_dict_list = []
    for subdir, dirs, files in os.walk(dataset_path, followlinks=True):
        if files:
            # Ignore the 'norm.csv'
            if len(files) == 1 and "norm.csv" in files[0]:
                # Copy over the norm.csv file
                shutil.copy(os.path.join(subdir, files[0]), output_path)

            # Create the output subdirectories
            output_subdir = subdir.replace(dataset_path, output_path)
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            # Grab the setname
            if "train" in subdir:
                set_name = "train"
            elif "test" in subdir:
                set_name = "test"
            else:
                continue

            # Loop over the files and store accordingly
            dataset_dict = {
                "lr_image_packets": [],
                "hr_image_packet": None,
                "output_scene_path": output_subdir,
                "set_name": set_name,
                "scene_path": subdir
            }

            # Grab all the .png fies
            png_files = glob(os.path.join(subdir, "*.png"))
            for filename in png_files:
                basename = os.path.basename(filename)

                # Find the corresponding map
                if hr_image_prefix in basename:
                    qm_image_path = filename.replace(hr_image_prefix, sm_image_prefix)
                    if not os.path.exists(qm_image_path):
                        logger.error("QM does not exist for image: %s" % filename)
                        qm_image_path = None

                    # Store the paths
                    dataset_dict["hr_image_packet"] = {
                        "image_path": filename,
                        "qm_image_path": qm_image_path
                    }
                elif lr_image_prefix in basename:
                    qm_image_path = filename.replace(lr_image_prefix, qm_image_prefix)
                    if not os.path.exists(qm_image_path):
                        logger.error("QM does not exist for image: %s" % filename)
                        qm_image_path = None

                    # Store the paths
                    dataset_dict["lr_image_packets"].append({
                        "image_path": filename,
                        "qm_image_path": qm_image_path
                    })
                else:
                    continue

            # Store the dataset dict for this scene
            dataset_dict_list.append(dataset_dict)
    return dataset_dict_list


# --------------------------------- MAIN ----------------------------------
def main():
    # Check the input paths
    check_and_create_paths()

    # Iterate over the whole dataset, create output subdirs, and log images/QMs
    logger.info("Reading the dataset...")
    dataset_dict_list = log_dataset(args.dataset_path, args.output_path)

    # Load and build the cnn registration module
    cnn_registration_module = CNNRegistration.CNN()

    # Loop over the dataset dict list and do all processing
    for data_packet in tqdm(dataset_dict_list, desc="PreprocessingDataset"):
        # Create a SceneProcessor class
        scene_preprocessor = ScenePreprocessor(
            scene_packet=data_packet,
            cnn_registration_module=cnn_registration_module,
            do_registration=args.do_registration
        )

        # Preprocess image-processing functions and cnn registration
        scene_preprocessor.preprocess_image_processing()
        scene_preprocessor.preprocess_image_registration()
        scene_preprocessor.output_images_to_file()
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

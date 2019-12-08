import os
import sys
import random
import logging
import argparse
from glob import glob

# MISR imports
from submodules.dcscn_super_resolution import DCSCN as louis_misr
from submodules.dcscn_super_resolution.helper import args as misr_args

# Embiggen module
from submodules.probav.embiggen import io as probav_io

# 3rd-party
import cv2
import cmapy
import scipy
import skimage
import numpy as np
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(os.path.dirname(os.path.realpath(__file__)))
logger.setLevel(logging.INFO)

args = misr_args.get()


# -------------------------- CLASS DEFINITIONS ----------------------------
class ProbavInferenceEngine:
    def __init__(
            self,
            db_path,
            model_path,
            scale_factor=3,
            num_inputs=9,
            create_submission=False
    ):
        self.db_path = db_path
        self.model_path = model_path
        self.scale_factor = scale_factor
        self.num_inputs = num_inputs
        self.create_submission = create_submission

        # Storage params
        self.dataset_list = []
        self.misr_model = None
        self.submission_dict = {"images": [], "filenames": []}

        # Path to dump the submission
        self.submission_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "submission.zip"
        )

        # Pre-processing ops
        self.agg_op = lambda i: np.nanmean(i, axis=0)

        # Read the dataset and store
        logger.info("Reading dataset: %s" % self.db_path)
        self._read_and_log_db()

        # Initialize the MISR model
        self._init_misr_model()

    def _read_and_log_db(self):
        for subdir, dirs, files in os.walk(self.db_path, followlinks=True):
            if files:
                # Create a dict for storing the image packet
                image_data_packet = {
                    "image_path_hr": None,
                    "lr_image_packets": [],
                    "scene_path": subdir
                }

                # We need the HR image, and the LR image/s
                for filename in files:
                    file_fullpath = os.path.join(subdir, filename)
                    if args.hr_filename_prefix in filename:
                        image_data_packet["image_path_hr"] = file_fullpath
                    elif args.lr_filename_prefix in filename:
                        # Find the corresponding quality map
                        qm_path = file_fullpath.replace(args.lr_filename_prefix, "QM")
                        if not os.path.exists(qm_path):
                            raise ValueError("QM image not found for: %s" % file_fullpath)

                        # Store the packet
                        image_data_packet["lr_image_packets"].append({
                            "image_path": file_fullpath,
                            "qm_image_path": qm_path
                        })
                    else:
                        continue

                # Store the image data packet
                self.dataset_list.append(image_data_packet)

    def _init_misr_model(self):
        self.misr_model = louis_misr.SuperResolution(
            flags=misr_args.get(), model_name="misr", is_module_training=False
        )
        self.misr_model.load_graph_misr(frozen_graph_filename=self.model_path, num_input_images=self.num_inputs)
        self.misr_model.init_all_variables()

    @staticmethod
    def _lowres_image_iterator(path, img_as_float=True):
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

    def _create_agg_image(self, scene_path):
        lr_images = []
        lr_images_and_qms = self._lowres_image_iterator(scene_path, img_as_float=False)

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
        agg_image = self.agg_op(lr_images)
        some_clear = np.isnan(obsc).any(axis=0)
        obsc = self.agg_op(obsc)
        obsc[some_clear] = 0.0
        np.nan_to_num(agg_image, copy=False)
        agg_image += obsc
        return agg_image.astype(np.uint16)

    def _preprocess_probav_scene(self, scene_packet):
        scene_path = scene_packet["scene_path"]

        # Get the aggregated image of the lowres samples
        agg_ref_image = self._create_agg_image(scene_path=scene_path)

        # Now iterate over the lr images and preprocess them by reading the quality map and
        # filling the obscured pixels with the corresponding regions from aff_ref_image
        for lr_image_packet in scene_packet["lr_image_packets"]:
            lr_image = cv2.imread(lr_image_packet["image_path"], cv2.IMREAD_UNCHANGED)
            qm = cv2.imread(lr_image_packet["qm_image_path"], cv2.IMREAD_UNCHANGED)
            preprocessed_image = np.where(qm == 0, agg_ref_image, lr_image)
            lr_image_packet["preprocessed_image"] = preprocessed_image

    def run_inference_on_dataset(self):
        # Create some named windows for viewing
        lr_images_window_name = "LR input images"
        output_window_name = "Left: Bicubuc upscaling. Middle: MISR output. Right: HR image"
        # cv2.namedWindow(lr_images_window_name, cv2.WINDOW_NORMAL)
        # cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)

        for image_packet in tqdm(self.dataset_list, desc="ProcessingProbaV"):
            # Pre-process this scene's LR images.
            self._preprocess_probav_scene(scene_packet=image_packet)

            # Choose our images for input
            input_image_packets = image_packet["lr_image_packets"][0:args.num_inputs]
            input_images_list = [x["preprocessed_image"] for x in input_image_packets]
            if args.shuffle_input_images:
                random.shuffle(input_images_list)

            # Load the HR image if it is there, else None
            hr_image = None
            if image_packet["image_path_hr"] is not None:
                hr_image = cv2.imread(image_packet["image_path_hr"], cv2.IMREAD_UNCHANGED)

            # Choose the first image as the reference. This will also be the 'x2' input
            x2_input_image = input_images_list[0]
            x2_input_image = cv2.resize(x2_input_image, None, fx=args.scale_factor, fy=args.scale_factor)
            x2_input_image_norm = cv2.normalize(x2_input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            # Run the inference on the MISR model
            output_image, lr_image_grid, concat_image = self.misr_model.do_for_misr_with_visualisation(
                lr_input_images=input_images_list,
                x2_image=x2_input_image,
                hr_image=hr_image
            )

            # We want to add the artificially upscaled image to the concat image
            new_concat_width = concat_image.shape[1] + x2_input_image.shape[1]
            new_concat_height = concat_image.shape[0]
            new_concat_image = np.zeros((new_concat_height, new_concat_width))
            new_concat_image[0:, 0:x2_input_image.shape[1]] = x2_input_image_norm
            new_concat_image[0:, x2_input_image.shape[1]:] = concat_image

            # Display the output. Images are normalized for viewing in the MISR module.
            # If we are creating a submission, skip showing the output
            if not self.create_submission:
                cv2.imshow(lr_images_window_name, cv2.applyColorMap(
                    lr_image_grid.astype(np.uint8), cmapy.cmap("viridis")))
                cv2.imshow(output_window_name, cv2.applyColorMap(
                    new_concat_image.astype(np.uint8), cmapy.cmap('viridis')))
                cv2.waitKey(0)
            else:
                # If we are creating a submission, store the relevant params
                self.submission_dict["images"].append(output_image.astype(np.uint16))
                self.submission_dict["filenames"].append(os.path.basename(image_packet["scene_path"]))

        # Write the submission, if we must
        probav_io.prepare_submission(
            images=self.submission_dict["images"],
            scenes=self.submission_dict["filenames"],
            subm_fname=self.submission_path
        )


# ------------------------- FUNCTION DEFINITIONS --------------------------


# --------------------------------- MAIN ----------------------------------
def main():
    # Check the given paths, params and report
    if not os.path.exists(args.db_path) or not os.path.isdir(args.db_path):
        raise ValueError("Invalid or missing 'db_path'. Please check: %s" % args.db_path)
    if not os.path.exists(args.model_path) or not os.path.isfile(args.model_path):
        raise ValueError("Invalid or missing 'model_path'. Please check: %s" % args.model_path)
    if args.num_inputs > 9:
        raise ValueError("Number of input images must be <= 9. Otherwise you will run out of GPU memory :)")

    # Construct the inference engine and run
    misr_inference_engine = ProbavInferenceEngine(
        db_path=args.db_path,
        model_path=args.model_path,
        scale_factor=3,
        num_inputs=9,
        create_submission=args.create_submission
    )
    misr_inference_engine.run_inference_on_dataset()
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

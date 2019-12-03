import os
import sys
import random
import logging
import argparse

# CNN image registration imports

# MISR imports
from submodules.dcscn_super_resolution import DCSCN as louis_misr
from submodules.dcscn_super_resolution.helper import args as misr_args

# 3rd-party
import cv2
import cmapy
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--db_path",
#     type=str,
#     required=True,
#     help='Description'
# )
# parser.add_argument(
#     "--model_path",
#     type=str,
#     required=True,
#     help='Description'
# )
# parser.add_argument(
#     "--num_inputs",
#     type=int,
#     default=9,
#     required=False,
#     help='Description'
# )
# parser.add_argument(
#     "--scale_factor",
#     type=int,
#     default=3,
#     required=False,
#     help='Description'
# )
# parser.add_argument(
#     "--shuffle_input_images",
#     action="store_true",
#     default=False,
#     required=False,
#     help='Description'
# )
# parser.add_argument(
#     "--hr_filename_prefix",
#     type=str,
#     default="HR",
#     required=False,
#     help='Description'
# )
# parser.add_argument(
#     "--lr_filename_prefix",
#     type=str,
#     default="LR",
#     required=False,
#     help='Description'
# )
# args = parser.parse_args()

# -------------------------- SETUP PARAMETERS -----------------------------
# Set up the logger
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(os.path.dirname(os.path.realpath(__file__)))
logger.setLevel(logging.INFO)

args = misr_args.get()

# -------------------------- CLASS DEFINITIONS ----------------------------


# ------------------------- FUNCTION DEFINITIONS --------------------------
def read_and_log_db(db_path):
    dataset_list = []
    for subdir, dirs, files in os.walk(db_path, followlinks=True):
        if files:
            # Create a dict for storing the image packet
            image_data_packet = {
                "image_path_hr": None,
                "image_path_lr": [],
                "quality_map_lr": []
            }

            # We need the HR image, and the LR image/s
            for filename in files:
                file_fullpath = os.path.join(subdir, filename)
                if args.hr_filename_prefix in filename:
                    image_data_packet["image_path_hr"] = file_fullpath
                elif args.lr_filename_prefix in filename:
                    image_data_packet["image_path_lr"].append(file_fullpath)
                else:
                    # Probably a QM (quality map)
                    image_data_packet["quality_map_lr"].append(file_fullpath)

            # Store the image data packet
            dataset_list.append(image_data_packet)
    return dataset_list


def run_for_dataset(dataset_list, misr_model):
    """
    For each entry in the dataset:
        - Shuffle the input order if needed
        - Load in the images
        - Apply preprocessing:
            - Assign mean of areas to low quality regions based on QM
            - Do image registration using the first image as the reference
        - Run the entry through MISR
        - Show the results with opencv
    :param dataset_list:
    :param misr_model:
    :return:
    """
    # Create some named windows for viewing
    lr_images_window_name = "LR input images"
    output_window_name = "Left: Bicubuc upscaling. Middle: MISR output. Right: HR image"
    cv2.namedWindow(lr_images_window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)

    for image_packet in dataset_list:
        # Choose our images for input
        input_image_paths = image_packet["image_path_lr"][0:args.num_inputs]
        if args.shuffle_input_images:
            random.shuffle(input_image_paths)

        # Load the HR image if it is there, else None
        if image_packet["image_path_hr"] is None:
            hr_image = None
        else:
            hr_image = cv2.imread(image_packet["image_path_hr"], cv2.IMREAD_UNCHANGED)

        # Load in our input images with opencv
        loaded_lr_images = []
        for image_path in input_image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.error("Corrupted image: %s" % image_path)
                raise AssertionError
            loaded_lr_images.append(image)

        # TODO: Preprocessing here

        # Choose the first image as the reference. This will also be the 'x2' input
        x2_input_image = loaded_lr_images[0]
        x2_input_image = cv2.resize(x2_input_image, None, fx=args.scale_factor, fy=args.scale_factor)
        x2_input_image_norm = cv2.normalize(x2_input_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Run the inference on the MISR model
        lr_image_grid, concat_image = misr_model.do_for_misr_with_visualisation(
            lr_input_images=loaded_lr_images,
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
        cv2.imshow(lr_images_window_name, cv2.applyColorMap(
            lr_image_grid.astype(np.uint8), cmapy.cmap("viridis")))
        cv2.imshow(output_window_name, cv2.applyColorMap(
            new_concat_image.astype(np.uint8), cmapy.cmap('viridis')))
        cv2.waitKey(0)


# --------------------------------- MAIN ----------------------------------
def main():
    # Check the given paths, params and report
    if not os.path.exists(args.db_path) or not os.path.isdir(args.db_path):
        logger.error("Invalid or missing 'db_path'. Please check: %s" % args.db_path)
        raise ValueError
    if not os.path.exists(args.model_path) or not os.path.isfile(args.model_path):
        logger.error("Invalid or missing 'model_path'. Please check: %s" % args.model_path)
        raise ValueError
    if args.num_inputs > 9:
        logger.error("Number of input images must be <= 9. Otherwise you will run out of GPU memory :)")
        raise ValueError

    # Read the given data and store
    dataset_list = read_and_log_db(args.db_path)

    # Load the MISR module and load the frozen graph
    misr_model = louis_misr.SuperResolution(flags=misr_args.get(), model_name="misr", is_module_training=False)
    misr_model.load_graph_misr(
        frozen_graph_filename=args.model_path, num_input_images=args.num_inputs
    )
    misr_model.init_all_variables()

    # Run the MISR on the database
    run_for_dataset(dataset_list=dataset_list, misr_model=misr_model)
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

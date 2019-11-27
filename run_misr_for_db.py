import os
import sys
import logging
import argparse

# MISR imports
from 

# 3rd-party
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--db_path",
    type=str,
    required=True,
    help='Description'
)
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help='Description'
)
parser.add_argument(
    "--num_input_images",
    type=int,
    default=9,
    required=False,
    help='Description'
)
parser.add_argument(
    "--shuffle_input_images",
    type=bool,
    action="store_true",
    default=False,
    required=False,
    help='Description'
)
args = parser.parse_args()

# -------------------------- SETUP PARAMETERS -----------------------------
# Set up the logger
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt='%d-%b-%y %H:%M:%S')
logger = logging.getLogger(os.path.dirname(os.path.realpath(__file__)))
logger.setLevel(logging.INFO)


# -------------------------- CLASS DEFINITIONS ----------------------------


# ------------------------- FUNCTION DEFINITIONS --------------------------


# --------------------------------- MAIN ----------------------------------
def main():
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)

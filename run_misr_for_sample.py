import os
import sys
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--arg0",
    type=str,
    required=True,
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

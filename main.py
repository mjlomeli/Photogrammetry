#!/usr/bin/env python
"""
Starts the Photogrammetry project.

First, calibration needed using the checkerboard images from data folder.
Second, the decoding process is started and the data is stored.
"""

import sys
from decoder import Decoder
from calibrate import Calibrate
from pathlib import Path

__author__ = "Mauricio Lomeli"
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"

CALIB_PATH = Path.cwd() / Path("data")
IMAGES_PATH = Path.cwd() /Path("data")

def main():
    calibrate = Calibrate(CALIB_PATH)

def find_rmv_files(directory: Path):
    """
    Removes all calibration and intrinsic files in the data folders.
    :param directory: Path of the data folder.
    """
    calibration_file = directory / Path('calibration.pickle')
    intrinsics_file = directory / Path('intrinsics.pickle')

    if calibration_file.exists():
        calibration_file.unlink()
    if intrinsics_file.exists():
        intrinsics_file.unlink()

    for path in directory.iterdir():
        if path.is_dir():
            find_rmv_files(path)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        if '-r' in sys.argv:
            find_rmv_files(Path.cwd() / Path('data'))
        if '-f' in sys.argv:
            CALIB_PATH = CALIB_PATH / Path('calib_png_small')
            IMAGES_PATH = IMAGES_PATH / Path('teapot_small')
        else:
            CALIB_PATH = CALIB_PATH / Path('calib_jpg_u')
            IMAGES_PATH = IMAGES_PATH / Path('teapot')
    main()

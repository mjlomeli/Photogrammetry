#!/usr/bin/env python
"""Calibration Intrinsic

Calculates the intrinsic values, like focal length, offsets, and distance, of an image
with a checkerboard.
"""

import sys
import pickle
import numpy as np
import cv2
from pathlib import Path

from scipy.optimize import leastsq

__authors__ = ["Mauricio Lomeli", "Charless Fowlkes"]
__credits__ = ["Benjamin Cordier"]
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"

DATA_FOLDER = Path.cwd() / Path("data")


class Calibrate:
    def __init__(self, directory, imprefix: str, shape=(6, 8), length=2.8):
        glob = "*" + imprefix + "*."
        self.imprefix = imprefix
        self.corners = None
        if directory == None:
            self.path = DATA_FOLDER / Path('calib_jpg_u')
            self.cam_calibfiles = list(self.path.glob(glob + "jpg"))
            self.cam_calibfiles += list(self.path.glob(glob + "png"))
            self.__search_chess(shape, length)
        else:
            if isinstance(directory, list):
                self.path = Path.cwd()
                self.cam_calibfiles = directory
                self.__search_chess(shape, length)
            else:
                self.path = Path(directory)
                self.cam_calibfiles = list(self.path.glob(glob + "jpg"))
                self.cam_calibfiles += list(self.path.glob(glob + "png"))
                self.__search_chess(shape, length)

    def __search_chess(self, shape, length):
        # checkerboard coordinates in 3D
        objp = np.zeros((shape[0] * shape[1], 3), np.float32)  # pts3 of the checkerboard (initialized to zeros)
        objp[:, :2] = length * np.mgrid[0:shape[1], 0:shape[0]].T.reshape(-1, 2)  # we range 0 - 19.6 into x & y

        # arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space.
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        count = 0
        end = len(self.cam_calibfiles) + 1
        img_size = None
        printProgressBar(count, end, prefix='Calibrating', suffix='{}/{} files calibrated. '.format(count, end))
        for idx, fname in enumerate(self.cam_calibfiles):
            img = cv2.imread(str(fname))
            img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners; corners = pixel locations
            ret, self.corners = cv2.findChessboardCorners(gray, (shape[1], shape[0]), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(self.corners)

                # Display image with the corners overlayed
                cv2.drawChessboardCorners(img, (shape[1], shape[0]), self.corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
            count += 1
            printProgressBar(count, end, prefix='Calibrating', suffix='{}/{} files calibrated. '.format(count, end))
        printProgressBar(count, end, prefix='Calibrating', suffix='Saving file.            ')
        cv2.destroyAllWindows()
        # calibrates the camera with the list of measured objpoints and a list of pixel location of the corners.
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.cx = K[0][2]
        self.cy = K[1][2]
        self.dist = dist
        count += 1
        printProgressBar(count, end, prefix='Calibrating', suffix='Finished calibrating. ')

    def __iter__(self):
        keys = ['fx', 'fy', 'cx', 'cy', 'dist']
        for key in keys:
            yield (key, eval('self.' + key))

    def __str__(self):
        # save the results out to a file for later use
        calib = "{\n"
        calib += "\tfx:\t" + str(self.fx) + ",\n"
        calib += "\tfy:\t" + str(self.fy) + ",\n"
        calib += "\tcx:\t" + str(self.cx) + ",\n"
        calib += "\tcy:\t" + str(self.cy) + ",\n"
        calib += "\tdist:\t" + "array(" + str(self.dist) + ")\n"
        calib += "}"
        return calib


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Displays a progress bar for each iteration.
    Title: Progress Bar
    Author: Benjamin Cordier
    Date: 6/10/2019
    Code version: n/a
    Availability: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    if int(iteration % (total / 100)) == 0 or iteration == total or prefix is not '' or suffix is not '':
        # calculated percentage of completeness
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        # modifies the bar
        bar = fill * filledLength + '-' * (length - filledLength)
        # Creates the bar
        print('\r\t\t{} |{}| {}% {}'.format(prefix, bar, percent, suffix), end='\r')
        # Print New Line on Complete
        if iteration == total:
            print()


def find_rmv_files(directory: Path):
    """
    Removes all calibration files in the data folders.
    :param directory: Path of the data folder.
    """
    calib_small_file_C0 = directory / Path('calib_png_small') / Path('calibration_C0.pickle')
    calib_small_file_C1 = directory / Path('calib_png_small') / Path('calibration_C1.pickle')
    calib_large_file_C0 = directory / Path('calib_jpg_u') / Path('calibration_C0.pickle')
    calib_large_file_C1 = directory / Path('calib_jpg_u') / Path('calibration_C1.pickle')

    if calib_small_file_C0.exists():
        calib_small_file_C0.unlink()
    if calib_small_file_C1.exists():
        calib_small_file_C1.unlink()
    if calib_large_file_C0.exists():
        calib_large_file_C0.unlink()
    if calib_large_file_C1.exists():
        calib_large_file_C1.unlink()



if __name__ == "__main__":
    """
    Runs the program:
        python calibrate.py [-r] [-f]
    -r: Erases the previous calibrations.
    -f: Runs a lower resolution of the calibration for faster debugging.
    """
    calib_path = None
    title = "Calibration of {} Resolution"
    if len(sys.argv) > 1:
        if '-r' in sys.argv:
            find_rmv_files(DATA_FOLDER)
        if '-f' in sys.argv:
            calib_path = DATA_FOLDER / Path('calib_png_small')
            title = title.format('Low')
    else:
        calib_path = DATA_FOLDER / Path('calib_jpg_u')
        title = title.format('High')

    calibrate_1 = Calibrate(calib_path, 'C0')
    print(title)
    print("calibrate_1:")
    print(calibrate_1)

    calibrate_2 = Calibrate(calib_path, 'C1')
    print("calibrate_2:")
    print(calibrate_2)
    print()

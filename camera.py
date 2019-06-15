#!/usr/bin/env python
"""Camera

Uses the calibration values, like focal length, offsets, and distance, of an image
with a checkerboard to calibrate a camera.
"""
import pickle
import sys
import numpy as np
from scipy.optimize import leastsq

from calibrate import Calibrate
from pathlib import Path

__authors__ = ["Mauricio Lomeli", "Charless Fowlkes"]
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"

DATA_FOLDER = Path.cwd() / Path('data')


class Camera(Calibrate):
    """
    A simple data structure describing camera parameters

    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation
    """

    def __init__(self, directory: Path, imprefix: str, R, t=np.array([[0, 0, -2]]).T,
                 chess_size=(6, 8), square_length=2.8):
        self.directory = directory
        self.pickle = Path('calibration_' + imprefix + '.pickle')
        self.imprefix = imprefix

        if directory == None:
            self.path = DATA_FOLDER / Path('calib_jpg_u')
            self.pickle = self.path / self.pickle
            if self.pickle.exists():
                self.get_pickle()
            else:
                Calibrate.__init__(self, directory, imprefix, chess_size, square_length)
                self.f = (self.fx + self.fy) / 2
                self.c = np.array([[self.cx, self.cy]]).T
                self.R = makerotation(0, 0, 0) if R is None else R
                self.t = t
                self.align_cameras(chess_size, square_length)
        else:
            if isinstance(directory, list):
                self.path = Path.cwd()
                self.pickle = self.path / self.pickle
                if self.pickle.exists():
                    self.get_pickle()
                else:
                    Calibrate.__init__(self, directory, imprefix, chess_size, square_length)
                    self.f = (self.fx + self.fy) / 2
                    self.c = np.array([[self.cx, self.cy]]).T
                    self.R = makerotation(0, 0, 0) if R is None else R
                    self.t = t
                    self.align_cameras(chess_size, square_length)
            else:
                self.path = Path(directory)
                self.pickle = self.path / self.pickle
                if self.pickle.exists():
                    self.get_pickle()
                else:
                    Calibrate.__init__(self, directory, imprefix, chess_size, square_length)
                    self.f = (self.fx + self.fy) / 2
                    self.c = np.array([[self.cx, self.cy]]).T
                    self.R = makerotation(0, 0, 0) if R is None else R
                    self.t = t
                    self.align_cameras(chess_size, square_length)

        self.write_pickle()

    def align_cameras(self, chess_size, square_length):
        """
        Finds the calibration among the camera and the provided 3D points.
        :param chess_size: n x m number of cross points
        :param square_length: length of each square on the board
        """
        pts2 = self.corners.squeeze().T
        pts3 = np.zeros((3,chess_size[0]*chess_size[1]))
        yy, xx = np.meshgrid(np.arange(chess_size[1]), np.arange(chess_size[0]))
        pts3[0, :] = square_length * xx.reshape(1, -1)
        pts3[1, :] = square_length * yy.reshape(1, -1)
        rt = np.array([0, 0, 0, 0, 0, -2])
        cam = self.calibratePose(pts3, pts2, self, rt)
        self.R = cam.R
        self.t = cam.t

    def __str__(self):
        return f'Camera_{self.imprefix} : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'

    def project(self, pts3):
        """
        Project the given 3D points in world coordinates into the specified camera

        :param pts3: Coordinates of N points stored in a array of shape (3,N)
        :return pts2: Image coordinates of N points stored in an array of shape (2,N)
        """
        assert (pts3.shape[0] == 3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)

        # project
        p = self.f * (pcam / pcam[2, :])

        # offset principal point
        pts2 = p[0:2, :] + self.c

        assert (pts2.shape[1] == pts3.shape[1])
        assert (pts2.shape[0] == 2)

        return pts2

    def update_extrinsics(self, params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.

        :param params: Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation
        """
        self.R = makerotation(params[0], params[1], params[2])
        self.t = np.array([[params[3]], [params[4]], [params[5]]])

    def residuals(self, pts3, pts2, cam, params):
        """
        Compute the difference between the projection of 3D points by the camera
        with the given parameters and the observed 2D locations

        :param pts3: Coordinates of N points stored in a array of shape (3,N)
        :param pts2: Coordinates of N points stored in a array of shape (2,N)
        :param cam: Initial estimate of camera
        :param params: Camera parameters we are optimizing over stored in a vector
        :return residual: Vector of residual 2D projection errors of size 2*N
        """
        cam.update_extrinsics(params)
        residual = pts2 - cam.project(pts3)

        return residual.flatten()

    def calibratePose(self, pts3, pts2, cam_init, params_init):
        """
        Calibrate the provided camera by updating R,t so that pts3 projects
        as close as possible to pts2

        :param pts3: Coordinates of N points stored in a array of shape (3,N)
        :param pts2: Coordinates of N points stored in a array of shape (2,N)
        :param cam_init: Initial estimate of camera
        :param params_init: initial estimate of camera extrinsic parameters
        :return: camera, refined estimate of camera with updated R,t parameters
        """
        # define our error function
        func = lambda params: self.residuals(pts3, pts2, cam_init, params)
        least = leastsq(func, params_init)[0]
        cam_init.update_extrinsics(least)

        return cam_init

    def get_pickle(self):
        """
        Gets the calibrated values onto a pickle file. The file is located in the directory where
        the calibration images are stored.
        :param path: The directory of the checkerboard images.
        """
        if self.pickle.exists():
            with open(self.pickle, 'rb') as f:
                calib = pickle.load(f)
                self.f = calib.f
                self.c = calib.c
                self.R = calib.R
                self.t = calib.t

    def write_pickle(self):
        """
        Saves the calibrated values onto a pickle file. The file is located in the directory where
        the calibration images are stored.
        """
        with open(self.pickle, 'wb') as w:
            pickle.dump(self, w)


def makerotation(rx, ry, rz):
    """
    Generate a rotation matrix
    :param rx: Amount to rotate around x-axes in degrees
    :param ry: Amount to rotate around y-axes in degrees
    :param rz: Amount to rotate around z-axes in degrees
    :return R: Rotation matrix of shape (3,3)
    """
    sin = lambda theta: np.sin(np.deg2rad(theta))
    cos = lambda theta: np.cos(np.deg2rad(theta))

    # rotation matrices of x-rotation, y-rotation, and z-rotation
    rotx = np.array([[1, 0, 0], [0, cos(rx), -sin(rx)], [0, sin(rx), cos(rx)]])
    roty = np.array([[cos(ry), 0, -sin(ry)], [0, 1, 0], [sin(ry), 0, cos(ry)]])
    rotz = np.array([[cos(rz), -sin(rz), 0], [sin(rz), cos(rz), 0], [0, 0, 1]])

    return rotz @ roty @ rotx


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
        python camera.py [-r] [-f]
    -r: Erases the previous calibrations.
    -f: Runs a lower resolution of the calibration for faster debugging.
    """
    all = False
    calib_path = None
    title = "Calibration of {} Resolution"
    if len(sys.argv) > 1:
        if '-r' in sys.argv:
            find_rmv_files(DATA_FOLDER)
        if '-f' in sys.argv:
            calib_path = DATA_FOLDER / Path('calib_png_small')
            title = title.format('Low')
        if '-a' in sys.argv:
            title = title.format('High and Low')

    else:
        calib_path = DATA_FOLDER / Path('calib_jpg_u')
        title = title.format('High')

    if not all:
        print(title)
        camera_1 = Camera(calib_path, 'C0', None)
        print(camera_1)

        camera_2 = Camera(calib_path, 'C1', None)
        print(camera_2)
        print()
    else:
        print(title)
        calib_small_path = DATA_FOLDER / Path('calib_png_small')
        calib_large_path = DATA_FOLDER / Path('calib_jpg_u')
        camera_1 = Camera(calib_small_path, 'C0', None)
        camera_2 = Camera(calib_small_path, 'C1', None)
        print(camera_1)
        print(camera_2)
        camera_1 = Camera(calib_large_path, 'C0', None)
        camera_2 = Camera(calib_large_path, 'C1', None)
        print(camera_1)
        print(camera_2)


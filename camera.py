#!/usr/bin/env python
"""Camera

Uses the calibration values, like focal length, offsets, and distance, of an image
with a checkerboard to calibrate a camera.
"""

import sys
import numpy as np
from calibrate import Calibrate, find_rmv_files
from scipy.optimize import leastsq
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

    def __init__(self, directory: Path, imprefix: str, R, t=np.array([0, 0, -2]).T):
        Calibrate.__init__(self, directory, imprefix)
        self.f = (self.fx + self.fy) / 2
        self.c = np.array([[self.cx, self.cy]]).T
        self.R = makerotation(0,0,0) if R is None else R
        self.t = t

    def __str__(self):
        return f'Camera_{self.imprefix} : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'

    def project(self, pts3):
        """
        Project the given 3D points in world coordinates into the specified camera

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)
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

        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation
        """
        self.R = makerotation(params[0], params[1], params[2])
        self.t = np.array([[params[3]], [params[4]], [params[5]]])


def makerotation(rx, ry, rz):
    """
    Generate a rotation matrix

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi * rx / 180.0
    ry = np.pi * ry / 180.0
    rz = np.pi * rz / 180.0

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, -np.sin(ry)], [0, 1, 0], [np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R = (Rz @ Ry @ Rx)

    return R



def residuals(pts3, pts2, cam, params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing over stored in a vector

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
    """

    cam.update_extrinsics(params)
    residual = pts2 - cam.project(pts3)

    return residual.flatten()


def calibratePose(pts3, pts2, cam_init, params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera

    Returns
    -------
    cam_opt : Camera
        Refined estimate of camera with updated R,t parameters
    """

    # define our error function
    efun = lambda params: residuals(pts3, pts2, cam_init, params)
    popt, _ = leastsq(efun, params_init)
    cam_init.update_extrinsics(popt)

    return cam_init


if __name__ == "__main__":
    """
    Runs the program:
        python camera.py [-r] [-f]
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

    print(title)
    camera_1 = Camera(calib_path, 'C0', None)
    print(camera_1)

    camera_2 = Camera(calib_path, 'C1', None)
    print(camera_2)
    print()


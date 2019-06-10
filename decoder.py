#!/usr/bin/env python
"""This file decodes the images from structured illumination.

A pixel in each camera image is decoded by looking at the sequence of black and white shades across all the frames.
Images are encoded into binary and translated to graycode, then decoded into decimal.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import leastsq
from pathlib import Path


MODELS = Path.cwd() / Path("models")

__authors__ = ["Mauricio Lomeli", "Charless Fowlkes"]
__credits__ = ["Benjamin Cordier"]
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"


class Decoder:
    def __init__(self, imprefixL: str, imprefixR: str, threshold: int, camL, camR):
        """
        :param imprefixL: string, left camera folder prefix
        :param imprefixR: string, right camera folder prefix
        :param threshold: int, the threshold
        :param camL: Camera, left camera object
        :param camR: Camera, right camera object
        """
        pts2L, pts2R, pts3 = self.reconstruct(imprefixL, imprefixR, threshold, camL, camR)

        self.pts3 = pts3
        self.pts2R = pts2R
        self.pts2L = pts2L
        self.mesh_clean()

    def decode(self, imprefix: str, start: int, threshold: int):
        """
        Decode 10bit gray code pattern with the given difference
        threshold.  We assume the images come in consective pairs
        with filenames of the form <prefix><start>.png - <prefix><start+20>.png
        (e.g. a start offset of 20 would yield image20.png, image01.png... image39.png)

        Parameters
        ----------
        imprefix : str
          prefix of where to find the images (assumed to be .png)

        start : int
          image offset.

        threshold : float

        Returns
        -------
        code : 2D numpy.array (dtype=float)

        mask : 2D numpy.array (dtype=float)
        """
        nbits = 10

        imgs = list()
        imgs_inv = list()
        print('loading', end='')
        for i in range(start, start + 2 * nbits, 2):
            fname0 = '%s%2.2d.png' % (imprefix, i)
            fname1 = '%s%2.2d.png' % (imprefix, i + 1)
            print('(', i, i + 1, ')', end='')
            img = plt.imread(fname0)
            img_inv = plt.imread(fname1)
            if (img.dtype == np.uint8):
                img = img.astype(float) / 256
                img_inv = img_inv.astype(float) / 256
            if (len(img.shape) > 2):
                img = np.mean(img, axis=2)
                img_inv = np.mean(img_inv, axis=2)
            imgs.append(img)
            imgs_inv.append(img_inv)

        (h, w) = imgs[0].shape
        print('\n')

        gcd = np.zeros((h, w, nbits))
        mask = np.ones((h, w))
        for i in range(nbits):
            gcd[:, :, i] = imgs[i] > imgs_inv[i]
            mask = mask * (np.abs(imgs[i] - imgs_inv[i]) > threshold)

        bcd = np.zeros((h, w, nbits))
        bcd[:, :, 0] = gcd[:, :, 0]
        for i in range(1, nbits):
            bcd[:, :, i] = np.logical_xor(bcd[:, :, i - 1], gcd[:, :, i])

        code = np.zeros((h, w))
        for i in range(nbits):
            code = code + np.power(2, (nbits - i - 1)) * bcd[:, :, i]

        return code, mask

    def reconstruct(self, imprefixL, imprefixR, threshold, camL, camR):
        """
        Simple reconstruction based on triangulating matched pairs of points
        between to view which have been encoded with a 20bit gray code.

        Parameters
        ----------
        imprefix : str
          prefix for where the images are stored

        threshold : float
          decodability threshold

        camL,camR : Camera
          camera parameters

        Returns
        -------
        pts2L,pts2R : 2D numpy.array (dtype=float)

        pts3 : 2D numpy.array (dtype=float)
        """

        CLh, maskLh = self.decode(imprefixL, 0, threshold)
        CLv, maskLv = self.decode(imprefixL, 20, threshold)
        CRh, maskRh = self.decode(imprefixR, 0, threshold)
        CRv, maskRv = self.decode(imprefixR, 20, threshold)

        CL = CLh + 1024 * CLv
        maskL = maskLh * maskLv
        CR = CRh + 1024 * CRv
        maskR = maskRh * maskRv

        h = CR.shape[0]
        w = CR.shape[1]

        subR = np.nonzero(maskR.flatten())
        subL = np.nonzero(maskL.flatten())

        CRgood = CR.flatten()[subR]
        CLgood = CL.flatten()[subL]

        _, submatchR, submatchL = np.intersect1d(CRgood, CLgood, return_indices=True)

        matchR = subR[0][submatchR]
        matchL = subL[0][submatchL]

        xx, yy = np.meshgrid(range(w), range(h))
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))

        pts2R = np.concatenate((xx[matchR].T, yy[matchR].T), axis=0)
        pts2L = np.concatenate((xx[matchL].T, yy[matchL].T), axis=0)

        pts3 = triangulate(pts2L, camL, pts2R, camR)

        return pts2L, pts2R, pts3

    def mesh_clean(self):
        # TODO: in the selection class, defining the box is placed here.
        # Mesh cleanup parameters

        # Specify limits along the x,y and z axis of a box containing the object
        # we will prune out triangulated points outside these limits
        boxlimits = np.array([-140, 350, -120, 180, -190, 100])

        # Specify a longest allowed edge that can appear in the mesh. Remove triangles
        # from the final mesh that have edges longer than this value
        trithresh = 900
        #
        # bounding box pruning
        #

        prune_boxed_x = np.logical_and(self.pts3[0] > boxlimits[0], self.pts3[0] < boxlimits[1])
        x_trimmed = self.pts3[:, prune_boxed_x]
        pts2L = self.pts2L[:, prune_boxed_x]
        pts2R = self.pts2R[:, prune_boxed_x]

        prune_boxed_y = np.logical_and(x_trimmed[1] > boxlimits[0], x_trimmed[1] < boxlimits[3])
        xy_trimmed = x_trimmed[:, prune_boxed_y]
        pts2L = pts2L[:, prune_boxed_y]
        pts2R = pts2R[:, prune_boxed_y]

        prune_boxed_z = np.logical_and(xy_trimmed[2] > boxlimits[4], xy_trimmed[2] < boxlimits[5])
        xyz_trimmed = xy_trimmed[:, prune_boxed_z]
        pts2L = pts2L[:, prune_boxed_z]
        pts2R = pts2R[:, prune_boxed_z]

        xy_trim = np.bitwise_not(np.logical_and(xyz_trimmed[1] > 160, xyz_trimmed[0] > 218))
        xyz_trimmed = xyz_trimmed[:, xy_trim]
        pts2L = pts2L[:, xy_trim]
        pts2R = pts2R[:, xy_trim]

        xy_trim = np.bitwise_not(np.logical_and(xyz_trimmed[1] > 160, xyz_trimmed[0] < 60))
        xyz_trimmed = xyz_trimmed[:, xy_trim]
        pts2L = pts2L[:, xy_trim]
        pts2R = pts2R[:, xy_trim]

        yz_trim = np.bitwise_not(np.logical_and(xyz_trimmed[1] > 160, xyz_trimmed[2] < -140))
        xyz_trimmed = xyz_trimmed[:, yz_trim]
        pts2L = pts2L[:, yz_trim]
        pts2R = pts2R[:, yz_trim]

        xy_trim = np.bitwise_not(np.logical_and(xyz_trimmed[0] > 170, xyz_trimmed[1] > 175))
        xyz_trimmed = xyz_trimmed[:, xy_trim]
        pts2L = pts2L[:, xy_trim]
        pts2R = pts2R[:, xy_trim]

        xy_trim = xyz_trimmed[0] > 0
        xyz_trimmed = xyz_trimmed[:, xy_trim]
        pts2L = pts2L[:, xy_trim]
        pts2R = pts2R[:, xy_trim]

        #
        # triangulate the 2D points to get the surface mesh
        #
        triL = Delaunay(pts2L.T)
        triR = Delaunay(pts2R.T)

        trianglesL = pts2L.T[triL.simplices]
        trianglesR = pts2R.T[triR.simplices]
        #
        # triangle pruning
        #

        # removes the triangles based on the edge distances of each edge, if one qualifies, it is removed
        #
        # remove any points which are not refenced in any triangle
        #
        simpL = triL.simplices
        simpR = triR.simplices
        edge1L = np.bitwise_not(np.absolute(np.diff(trianglesL.T[:, 0], axis=0)) > trithresh)[0]
        edge1R = np.bitwise_not(np.absolute(np.diff(trianglesR.T[:, 0], axis=0)) > trithresh)[0]
        trianglesL = trianglesL[edge1L]
        trianglesR = trianglesR[edge1R]
        simpL = simpL[edge1L]
        simpR = simpR[edge1R]
        edge2L = np.bitwise_not(np.absolute(np.diff(trianglesL.T[:, 1], axis=0)) > trithresh)[0]
        edge2R = np.bitwise_not(np.absolute(np.diff(trianglesR.T[:, 1], axis=0)) > trithresh)[0]
        trianglesL = trianglesL[edge2L]
        trianglesR = trianglesR[edge2R]
        simpL = simpL[edge2L]
        simpR = simpR[edge2R]
        edge3L = np.bitwise_not(np.absolute(np.diff(trianglesL.T[:, 2], axis=0)) > trithresh)[0]
        edge3R = np.bitwise_not(np.absolute(np.diff(trianglesR.T[:, 2], axis=0)) > trithresh)[0]
        self.trianglesL = trianglesL[edge3L]
        self.trianglesR = trianglesR[edge3R]
        self.simpL = simpL[edge3L]
        self.simpR = simpR[edge3R]

    def plot(self):
        fig = plt.figure()
        x, y, z = self.pts3
        ax = fig.gca(projection="3d")
        ax.plot_trisurf(x, y, z, triangles=self.simpL, cmap=plt.cm.Spectral)
        ax.view_init(azim=0)
        plt.show()

        fig = plt.figure()
        x, y, z = self.pts3
        ax = fig.gca(projection="3d")
        ax.plot_trisurf(x, y, z, triangles=self.simpL, cmap=plt.cm.Spectral)
        ax.view_init(azim=300)
        plt.show()

        fig = plt.figure()
        x, y, z = self.pts3
        ax = fig.gca(projection="3d")
        ax.plot_trisurf(x, y, z, triangles=self.simpR, cmap=plt.cm.Spectral)
        ax.view_init(azim=0)
        plt.show()

        fig = plt.figure()
        x, y, z = self.pts3
        ax = fig.gca(projection="3d")
        ax.plot_trisurf(x, y, z, triangles=self.simpR, cmap=plt.cm.Spectral)
        ax.view_init(azim=300)
        plt.show()

    def __add__(self, other):
        # TODO: add other decoders so that they mesh together
        pass

    def __sub__(self, other):
        # TODO: remove other decoders so that you undo somethings
        pass

    def __str__(self):
        # TODO: print the table, etc.
        pass

    def __getitem__(self, item):
        # TODO: gets the value of a point (ie color, pos, etc)
        pass

    def __setitem__(self, key, value):
        # Todo: sets the value of a point (ie color, pos, etc)
        pass

    def __iter__(self):
        # Todo: get it to be iterable through all the points
        pass

    def __next__(self):
        # Todo: helps to start indexing in iterator i think
        pass

    def __copy__(self):
        # Todo: prevent shallow copies in case copies are required later
        pass






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


class Camera:
    """
    A simple data structure describing camera parameters

    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation
    """

    def __init__(self, f, c, R, t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'

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


def triangulate(pts2L, camL, pts2R, camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system


    Parameters
    ----------
    pts2L : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camL camera

    pts2R : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N) seen from camR camera

    camL : Camera
        The first "left" camera view

    camR : Camera
        The second "right" camera view

    Returns
    -------
    pts3 : 2D numpy.array (dtype=float)
        (3,N) array containing 3D coordinates of the points in global coordinates
    """

    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL, np.ones((1, npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR, np.ones((1, npts))))

    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t - camL.t)

    xL = np.zeros((3, npts))
    xR = np.zeros((3, npts))

    for i in range(npts):
        A = np.vstack((qL[:, i], -R @ qR[:, i])).T
        z, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        xL[:, i] = z[0] * qL[:, i]
        xR[:, i] = z[1] * qR[:, i]

    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = 0.5 * (pts3L + pts3R)

    return pts3


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


def __printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """ 
    Displays a progress bar for each test.
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

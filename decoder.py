#!/usr/bin/env python
"""This file decodes the images from structured illumination.

A pixel in each camera image is decoded by looking at the sequence of black and white shades across all the frames.
Images are encoded into binary and translated to graycode, then decoded into decimal.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import pickle
from pathlib import Path


DATA_FOLDER = Path.cwd() / Path("data")

__authors__ = ["Mauricio Lomeli", "Charless Fowlkes"]
__credits__ = ["Benjamin Cordier"]
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"


class Decoder:
    def __init__(self, imprefixL: str, imprefixR: str, threshold: int, camL, camR, path: Path):
        """
        :param path: Path, where the raw images are located
        :param imprefixL: string, left camera folder prefix
        :param imprefixR: string, right camera folder prefix
        :param threshold: int, the threshold
        :param camL: Camera, left camera object
        :param camR: Camera, right camera object
        """
        self.keys = ['trianglesL', 'trianglesR', 'simpL', 'simpR', 'pts2L', 'pts2R', 'pts3', 'camL', 'camR']
        self.__index = 0
        self.path = path
        self.camL = camL
        self.camR = camR

        if path.exists() and path.is_dir():
            pickle_file = path / Path('intrinsics.pickle')
            if pickle_file.exists():
                self.get_pickle()
        else:
            prefixR = str(path) + imprefixR
            prefixL = str(path) + imprefixL
            self.pts2L, self.pts2R, self.pts3 = self.reconstruct(prefixL, prefixR, threshold, camL, camR)
            self.mesh_clean()
            self.write_pickle()

    def get_pickle(self):
        """
        Loads the decoded values from a pickle file. The file is located in the directory where
        the raw images are stored.
        """
        if self.path.exists():
            with open(self.path, 'rb') as f:
                intrinsics = pickle.load(f)
                self.pts2L = intrinsics['pts2L']
                self.pts2R = intrinsics['pts2R']
                self.pts3 = intrinsics['pts3']
                self.trianglesL = intrinsics['trianglesL']
                self.trianglesR = intrinsics['trianglesR']
                self.simpL = intrinsics['simpL']
                self.simpR = intrinsics['simpR']
                self.camL = intrinsics['camL']
                self.camR = intrinsics['camR']


    def write_pickle(self):
        """
        Saves the decoded values onto a pickle file. The file is located in the directory where
        the raw images are stored.
        """
        if self.path.exists():
            with open(self.path, 'wb') as f:
                intrinsics = {}
                intrinsics['pts2L'] = self.pts2L
                intrinsics['pts2R'] = self.pts2R
                intrinsics['pts3']= self.pts3
                intrinsics['trianglesL'] = self.trianglesL
                intrinsics['trianglesR'] = self.trianglesR
                intrinsics['simpL'] = self.simpL
                intrinsics['simpR'] = self.simpR
                intrinsics['camL'] = self.camL
                intrinsics['camR'] = self.camR
                pickle.dump(intrinsics, f)

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
        self.pts2L = pts2L
        self.pts2R = pts2R

    def __add__(self, other):
        # TODO: add other decoders so that they mesh together
        pass

    def __sub__(self, other):
        # TODO: remove other decoders so that you undo somethings
        pass

    def __str__(self):
        calib = "{\n"
        for key in self.keys:
            calib += "\t" + key + ":\t" + str(eval('self.' + key)) + ",\n"
        calib += "}"
        return calib

    def __iter__(self):
        for key in self.keys:
            yield (key, eval('self.' + key))

    def __next__(self):
        if self.__index >= len(self.keys):
            raise StopIteration
        item = self.keys[self.__index]
        self.__index += 1
        return item


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


def find_rmv_files(directory: Path):
    """
    Removes all intrinsic files in the data folders.
    :param directory: Path of the data folder.
    """
    intrinsic_file = directory / Path('intrinsic.pickle')

    if intrinsic_file.exists():
        intrinsic_file.unlink()

    for path in directory.iterdir():
        if path.is_dir():
            find_rmv_files(path)


if __name__ == "__main__":
    """
    Runs the program:
        python decoder.py [-r] [-f]
    -r: Erases the previous intrinsics.
    -f: Runs a lower resolution of the images for faster calculations (for debugging).
    """
    intrinsic_path = None
    title = "Decoder of {} Resolution"
    if len(sys.argv) > 1:
        if '-r' in sys.argv:
            find_rmv_files(DATA_FOLDER)
        if '-f' in sys.argv:
            intrinsic_path = DATA_FOLDER / Path('teapot_small')
            title = title.format('Low')
    else:
        intrinsic_path = DATA_FOLDER / Path('teapot')
        title = title.format('High')

    decoder = Decoder(intrinsic_path)
    print(title)
    print(decoder)
    print()

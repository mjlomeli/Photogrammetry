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
from camera import Camera
import visutils
from mayavi.mlab import *


DATA_FOLDER = Path.cwd() / Path("data")

__authors__ = ["Mauricio Lomeli", "Charless Fowlkes"]
__credits__ = ["Benjamin Cordier"]
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"


class Decoder:
    def __init__(self, imprefixL='', imprefixR='', threshold=0.02, camL=None, camR=None, path=DATA_FOLDER, testing=False):
        """
        :param path: Path, where the raw images are located
        :param imprefixL: string, left camera folder prefix
        :param imprefixR: string, right camera folder prefix
        :param threshold: int, the threshold
        :param camL: Camera, left camera object
        :param camR: Camera, right camera object
        """
        self.testing = testing
        if not testing:
            self.keys = ['trianglesL', 'trianglesR', 'simpL', 'simpR', 'pts2L', 'pts2R', 'pts3', 'camL', 'camR']
            self.__index = 0
            self.path = path
            self.camL = camL
            self.camR = camR
            self.pickle_file = path / Path('intrinsics.pickle')
            self.mask_C0, self.mask_C1 = self.get_mask(0.1)
            if self.pickle_file.exists():
                self.get_pickle()
            else:
                prefixL = str(path / Path('frame_' + imprefixL + '_'))
                prefixR = str(path / Path('frame_' + imprefixR + '_'))
                self.pts2L, self.pts2R, self.pts3 = self.reconstruct(prefixL, prefixR, threshold, camL, camR)
                self.mesh_clean()
                self.write_pickle()

    def get_mask(self, threshold):
        """
        To apply the mask, multiply the mask to the image
        :param threshold: amount difference to allow in the image
        :return: tuple, numpy array (N,M) masks of boolean values
        """
        background_C0 = plt.imread(str(self.path / Path('color_C0_00.png')))
        background_C1 = plt.imread(str(self.path / Path('color_C1_00.png')))
        foreground_C0 = plt.imread(str(self.path / Path('color_C0_01.png')))
        foreground_C1 = plt.imread(str(self.path / Path('color_C1_01.png')))

        mask_C0 = (np.sum(np.abs(foreground_C0 - background_C0), axis=2) > threshold).astype(bool)
        mask_C1 = (np.sum(np.abs(foreground_C1 - background_C1), axis=2) > threshold).astype(bool)

        return mask_C0, mask_C1

    def get_pickle(self):
        """
        Loads the decoded values from a pickle file. The file is located in the directory where
        the raw images are stored.
        """
        with open(self.pickle_file, 'rb') as f:
            intrinsics = pickle.load(f)
            self.pts2L = intrinsics.pts2L
            self.pts2R = intrinsics.pts2R
            self.pts3 = intrinsics.pts3
            self.camL = intrinsics.camL
            self.camR = intrinsics.camR
            self.simp = intrinsics.simp
            self.triangles = intrinsics.triangles

    def write_pickle(self):
        """
        Saves the decoded values onto a pickle file. The file is located in the directory where
        the raw images are stored.
        """
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self, f)

    def decode(self, imprefix: str, start: int, threshold: float):
        """
        Decode 10bit gray code pattern with the given difference
        threshold.  We assume the images come in consective pairs
        with filenames of the form <prefix><start>.png - <prefix><start+20>.png
        (e.g. a start offset of 20 would yield image20.png, image01.png... image39.png)
        :param imprefix: prefix of where to find the images
        :param start: image offset.
        :param threshold: decodability threshold
        :return: code, mask : 2D numpy.array (dtype=float)
        """
        images = None
        nbits = 10
        count = 0
        file_count = 0
        end = nbits * 3 + 1
        printProgressBar(count, end, 'Decoding {}'.format(self.path.name),
                         '{}/{} images found.'.format(file_count, nbits * 2))
        for i in range(start, start + (2 * nbits)):
            img = plt.imread(imprefix + '{:02d}.png'.format(i))
            if len(img.shape) > 2:
                img = np.average(img, axis=-1)
            if 'frame_C0_' in imprefix:
                img *= self.mask_C0
            elif 'frame_C1_' in imprefix:
                img *= self.mask_C1
            if images is None:
                images = np.array([img])
            else:
                images = np.append(images, [img], axis=0)
            count += 1
            file_count += 1
            printProgressBar(count, end, 'Decoding {}'.format(self.path.name),
                             '{}/{} Images found.'.format(file_count, nbits * 2))
        bit = (images[::2] > images[1::2])
        mask = np.all(np.abs(images[::2] - images[1::2]) > threshold, axis=0).astype(float)

        # we will assume a 10 bit code
        length = len(bit)
        img = bit.copy()
        deci = np.zeros(bit[0].shape)
        file_count = 0
        printProgressBar(count, end, 'Decoding {}'.format(self.path.name),
                         '{}/{} Images decoded.'.format(file_count, nbits))
        for i in range(length - 1):
            img[i + 1] = np.bitwise_xor(img[i], img[i + 1])
            deci += img[i] * (2 ** (length - 1 - i))
            count += 1
            file_count += 1
            printProgressBar(count, end, 'Decoding {}'.format(self.path.name),
                             '{}/{} Images decoded.'.format(file_count, nbits))
        deci += img[length - 1]
        code = deci.astype(float)
        count += 1
        file_count += 1
        printProgressBar(count + 1, end, 'Decoding {}'.format(self.path.name),
                         '{}/{} Finished decoding.'.format(file_count, nbits))
        return code, mask

    def reconstruct(self, imprefixL, imprefixR, threshold, camL, camR):
        """
        Simple reconstruction based on triangulating matched pairs of points
        between to view which have been encoded with a 20bit gray code.

        :param imprefixL: prefix for where the images are stored for the left cam
        :param imprefixR: prefix for where the immages are stored for the right cam
        :param threshold: decodability threshold
        :param camL: left camera parameters
        :param camR: right camera parameters
        :return: pts2L,pts2R, pts3 : 2D numpy.array (dtype=float)
        """
        HL, HLmask = self.decode(imprefixL, 0, threshold)
        VL, VLmask = self.decode(imprefixL, 20, threshold)
        HR, HRmask = self.decode(imprefixR, 0, threshold)
        VR, VRmask = self.decode(imprefixR, 20, threshold)

        # Constructs the combined 20 bit code C = H + 1024 * V and mask for each view
        CLmask = HL + 1024 * VL
        CRmask = HR + 1024 * VR
        maskL = HLmask * VLmask
        maskR = HRmask * VRmask

        h = CLmask.shape[0]
        w = CLmask.shape[1]

        # get positions of non-zero values
        Rpos = np.nonzero(maskR.flatten())
        Lpos = np.nonzero(maskL.flatten())

        # Find the indices of pixels in the left and right code image that have matching codes
        CR = CRmask.flatten()[Rpos]
        CL = CLmask.flatten()[Lpos]
        matchR = np.intersect1d(CR, CL, return_indices=True)[1].astype(int)
        matchL = np.intersect1d(CR, CL, return_indices=True)[2].astype(int)
        matchR = Rpos[0][matchR]
        matchL = Lpos[0][matchL]

        # Generates the corresponding pixel coordinates for the matched pixels.
        xx, yy = np.meshgrid(range(w), range(h))
        xx = np.reshape(xx, (-1, 1))
        yy = np.reshape(yy, (-1, 1))

        pts2R = np.concatenate((xx[matchR].T, yy[matchR].T), axis=0)
        pts2L = np.concatenate((xx[matchL].T, yy[matchL].T), axis=0)

        # triangulates the points
        pts3 = triangulate(pts2L, camL, pts2R, camR)
        return pts2L, pts2R, pts3

    def mesh_clean(self, trithresh=200, boxlimits=np.array([-140, 350, -120, 180, -190, 100])):
        """
         Removes any triangles for which the longest edge of the triangle has a length greater than trithresh.
        :param trithresh: int, Specify a longest allowed edge that can appear in the mesh.
        :param boxlimits: numpy arr(1,6), specifies the limits along the x,y and z axis of a box containing the object
        """
        # Mesh cleanup parameters

        # Specify limits along the x,y and z axis of a box containing the object
        # we will prune out triangulated points outside these limits

        # Specify a longest allowed edge that can appear in the mesh. Remove triangles
        # from the final mesh that have edges longer than this value

        #
        # bounding box pruning
        #
        pts3 = self.pts3
        pts2L = self.pts2L
        pts2R = self.pts2R
        goodpts = np.nonzero((pts3[0, :] > boxlimits[0]) & (pts3[0, :] < boxlimits[1]) & \
                             (pts3[1, :] > boxlimits[2]) & (pts3[1, :] < boxlimits[3]) & \
                             (pts3[2, :] > boxlimits[4]) & (pts3[2, :] < boxlimits[5]))
        pts3 = pts3[:, goodpts[0]]
        pts2L = pts2L[:, goodpts[0]]
        pts2R = pts2R[:, goodpts[0]]
        #
        # triangulate the 2D points to get the surface mesh
        #
        # compute initial triangulation
        triangles = Delaunay(pts2L.T)
        tri = triangles.simplices
        #
        # triangle pruning
        #
        d01 = np.sqrt(np.sum(np.power(pts3[:, tri[:, 0]] - pts3[:, tri[:, 1]], 2), axis=0))
        d02 = np.sqrt(np.sum(np.power(pts3[:, tri[:, 0]] - pts3[:, tri[:, 2]], 2), axis=0))
        d12 = np.sqrt(np.sum(np.power(pts3[:, tri[:, 1]] - pts3[:, tri[:, 2]], 2), axis=0))

        # removes the triangles based on the edge distances of each edge, if one qualifies, it is removed
        #
        goodtri = (d01 < trithresh) & (d02 < trithresh) & (d12 < trithresh)
        # remove any points which are not refenced in any triangle
        #
        tri = tri[goodtri, :]

        self.triangles = triangles
        self.simp = tri
        self.pts2L = pts2L
        self.pts2R = pts2R
        self.pts3 = pts3

    def show(self, num_pts3=0, num_mesh=0, num_pts2=False, mayavi=False, mesh=False):
        length = (np.max(self.pts3) - np.min(self.pts3)) / 15
        lookL = np.hstack((self.camL.t, self.camL.t + self.camL.R @ np.array([[0, 0, length]]).T))
        lookR = np.hstack((self.camR.t, self.camR.t + self.camR.R @ np.array([[0, 0, length]]).T))

        if num_pts2:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.pts3[0, :], self.pts3[2, :], '.')
            ax.plot(self.camR.t[0], self.camR.t[2], 'ro')
            ax.plot(self.camL.t[0], self.camL.t[2], 'bo')
            ax.plot(lookL[0, :], lookL[2, :], 'b')
            ax.plot(lookR[0, :], lookR[2, :], 'r')
            plt.title('XZ-view')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('z')
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.pts3[1, :], self.pts3[2, :], '.', animated=True)
            ax.plot(self.camR.t[1], self.camR.t[2], 'ro', animated=True)
            ax.plot(self.camL.t[1], self.camL.t[2], 'bo', animated=True)
            ax.plot(lookL[1, :], lookL[2, :], 'b', animated=True)
            ax.plot(lookR[1, :], lookR[2, :], 'r', animated=True)
            plt.title('YZ-view')
            plt.grid()
            plt.xlabel('y')
            plt.ylabel('z')
            plt.show()

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(self.pts3[0, :], self.pts3[1, :], '.')
            ax.plot(self.camR.t[0], self.camR.t[1], 'ro')
            ax.plot(self.camL.t[0], self.camL.t[1], 'bo')
            ax.plot(lookL[0, :], lookL[1, :], 'b')
            ax.plot(lookR[0, :], lookR[1, :], 'r')
            plt.title('XY-view')
            plt.grid()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()

        for i in range(num_pts3):
            fig = plt.figure()
            ax = fig.gca(projection="3d")
            ax.view_init(azim=(i * 45))
            ax.plot(self.pts3[0, :], self.pts3[1, :], self.pts3[2, :], '.')
            ax.plot(self.camR.t[0], self.camR.t[1], self.camR.t[2], 'ro')
            ax.plot(self.camL.t[0], self.camL.t[1], self.camL.t[2], 'bo')
            ax.plot(lookL[0, :], lookL[1, :], lookL[2, :], 'b')
            ax.plot(lookR[0, :], lookR[1, :], lookR[2, :], 'r')
            visutils.set_axes_equal_3d(ax)
            visutils.label_axes(ax)
            plt.show()

        for i in range(num_mesh):
            fig = plt.figure()
            x, y, z = self.pts3
            ax = fig.gca(projection="3d")
            ax.plot_trisurf(x, y, z, triangles=self.simp, cmap=plt.cm.Spectral)
            ax.view_init(azim=50 * i)
            plt.show()

        if mayavi:
            x, y, z = self.pts3
            plot3d(x,y,z, representation='points')

        if mesh:
            x, y, z = self.pts3
            triangular_mesh(x, y, z, self.triangles)



    def __add__(self, other):
        # TODO: add other decoders so that they mesh together
        pass

    def __sub__(self, other):
        # TODO: remove other decoders so that you undo somethings
        pass

    def __str__(self):
        decoder = ""
        for key in self.keys:
            decoder += key + ":\n"
            decoder += str(eval('self.' + key))
        return decoder

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
    '''
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coords relative to the
    Global coord system.
    '''
    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL, np.ones((1, npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR, np.ones((1, npts))))

    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t - camL.t)
    xL = np.ones(qL.shape, dtype=float)
    xR = np.ones(qR.shape, dtype=float)
    for i in range(npts):
        A = np.vstack((qL[:, i], -R @ qR[:, i])).T
        z, _, _, _ = np.linalg.lstsq(A, t, rcond=None)
        xL[:, i] = z[0] * qL[:, i]
        xR[:, i] = z[1] * qR[:, i]
    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = .5 * (pts3L + pts3R)
    return pts3


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
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
    intrinsic_file = directory / Path('intrinsics.pickle')

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
    all = False
    intrinsic_path = None
    calib_path = None
    title = "Decoder of {} Resolution"
    if len(sys.argv) > 1:
        if '-r' in sys.argv:
            find_rmv_files(DATA_FOLDER)
        if '-a' in sys.argv:
            all = True
            title = title.format("All")
        else:
            if '-f' in sys.argv:
                intrinsic_path = DATA_FOLDER / Path('teapot_small') / Path('grab_0_u')
                calib_path = DATA_FOLDER / Path('calib_png_small')
                title = title.format('Low')
            else:
                intrinsic_path = DATA_FOLDER / Path('teapot') / Path('grab_0_u')
                calib_path = DATA_FOLDER / Path('calib_jpg_u')
                title = title.format('High')

    else:
        intrinsic_path = DATA_FOLDER / Path('teapot') / Path('grab_0_u')
        calib_path = DATA_FOLDER / Path('calib_jpg_u')
        title = title.format('High')

    if not all:
        threshold = 0.02
        camera_C0 = Camera(calib_path, 'C0', None)
        camera_C1 = Camera(calib_path, 'C1', None)
        decoder = Decoder('C1', 'C0', threshold, camera_C1, camera_C0, intrinsic_path)
        print(title)
        decoder.show(2, 2, True)
    else:
        threshold = 0.02

        teapot = DATA_FOLDER / Path('teapot')
        calib_large_path = DATA_FOLDER / Path('calib_jpg_u')

        for directory in teapot.iterdir():
            camera_C0 = Camera(calib_large_path, 'C0', None)
            camera_C1 = Camera(calib_large_path, 'C1', None)
            decoder = Decoder('C1', 'C0', threshold, camera_C1, camera_C0, directory)

        threshold = 0.008
        teapot_small = DATA_FOLDER / Path('teapot_small')
        calib_small_path = DATA_FOLDER / Path('calib_png_small')

        for directory in teapot_small.iterdir():
            camera_C0 = Camera(calib_small_path, 'C0', None)
            camera_C1 = Camera(calib_small_path, 'C1', None)
            decoder = Decoder('C1', 'C0', threshold, camera_C1, camera_C0, directory)

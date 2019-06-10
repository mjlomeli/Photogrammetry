#
# To run:
#
# Install opencv modules in Anaconda environment:
#
#   conda install opencv
#   pip install --upgrade pip
#   pip install opencv-contrib-python
#
# Run calibrate.py from the commandline:
#
#   python calibrate.py

from pathlib import Path
import pickle
import numpy as np
import cv2

images_folder = Path.cwd() / Path("data") / Path("calib_jpg_u")

class Calibrate:
    def __init__(self, images=None, shape=(6, 8), length=2.8):
        if images == None:
            self.path = images_folder
            pickle_file = images_folder / Path('calibration.pickle')
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    calib = pickle.load(f)
                    self.fx = calib['fx']
                    self.fy = calib['fy']
                    self.cx = calib['cx']
                    self.cy = calib['cy']
                    self.dist = calib['dist']
            else:
                self.cam_calibfiles = list(images_folder.glob("*.jpg"))
                self.cam_calibfiles += list(self.path.glob("*.png"))
                self.__search_chess(shape, length)
        else:
            if isinstance(images, list):
                self.path = Path.cwd()
                pickle_file = self.path / Path('calibration.pickle')
                if pickle_file.exists():
                    with open(pickle_file, 'rb') as f:
                        calib = pickle.load(f)
                        self.fx = calib['fx']
                        self.fy = calib['fy']
                        self.cx = calib['cx']
                        self.cy = calib['cy']
                        self.dist = calib['dist']
                else:
                    self.cam_calibfiles = images
                    self.__search_chess(shape, length)
            else:
                self.path = Path(images)
                pickle_file = self.path / Path('calibration.pickle')
                if pickle_file.exists():
                    with open(pickle_file, 'rb') as f:
                        calib = pickle.load(f)
                        self.fx = calib['fx']
                        self.fy = calib['fy']
                        self.cx = calib['cx']
                        self.cy = calib['cy']
                        self.dist = calib['dist']
                else:
                    self.cam_calibfiles = list(self.path.glob("*.jpg"))
                    self.cam_calibfiles += list(self.path.glob("*.png"))
                    self.__search_chess(shape, length)

    def __search_chess(self, shape, length):
        resultfile = self.path / Path('calibration.pickle')

        # checkerboard coordinates in 3D
        objp = np.zeros((shape[0] * shape[1], 3), np.float32)  # NxM crosses (points) on the checkerboard
        objp[:, :2] = length * np.mgrid[0:shape[1], 0:shape[0]].T.reshape(-1, 2)  # 2.8cm x 2.8cm each square

        # arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space.
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        for idx, fname in enumerate(self.cam_calibfiles):
            img = cv2.imread(str(fname))
            img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (shape[1], shape[0]), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Display image with the corners overlayed
                cv2.drawChessboardCorners(img, (shape[1], shape[0]), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.cx = K[0][2]
        self.cy = K[1][2]
        self.dist = dist
        with open(resultfile, 'wb') as w:
            calib = dict(self)
            pickle.dump(calib, w)

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

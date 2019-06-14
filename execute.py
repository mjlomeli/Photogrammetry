from matplotlib import pyplot as plt
import numpy as np

def imports():
    statement = 'from pathlib import Path\n' +\
                'import numpy as np\n' +\
                'import cv2\n' +\
                'from matplotlib import pyplot as plt\n' + \
                'import sys\n' + \
                'import pickle\n' + \
                'import gdal\n' + \
                'from mayavi.mlab import *\n' + \
                'from scipy.spatial import Delaunay\n' + \
                'from camera import Camera\n' + \
                'from calibrate import Calibrate\n' + \
                'from decoder import Decoder\n' + \
                'DATA_FOLDER = Path.cwd() / Path("data")\n' + \
                'calib_small = DATA_FOLDER / Path("calib_png_small")\n' + \
                'calib = DATA_FOLDER / Path("calib_jpg_u")\n' + \
                'teapot = DATA_FOLDER / Path("teapot")\n' + \
                'teapot_small = DATA_FOLDER / Path("teapot_small")\n' + \
                'test_images = DATA_FOLDER / Path("testImages")\n' + \
                'from random import random\n' + \
                'import visutils\n' + \
                r'imprefix_C0 = r"C:\Users\mrtma\Desktop\Photogrammetry\data\teapot\grab_0_u\frame_C0_"' + '\n' + \
                r'imprefix_C1 = r"C:\Users\mrtma\Desktop\Photogrammetry\data\teapot\grab_0_u\frame_C1"' + '\n' + \
                r'intrinsic_path = DATA_FOLDER / Path("teapot") / Path("grab_0_u")' + '\n' + \
                "calib_path = DATA_FOLDER / Path('calib_jpg_u')\n" + \
                'threshold = 0.02\n' + \
                "camera_C0 = Camera(calib_path, 'C0', None)\n" + \
                "camera_C1 = Camera(calib_path, 'C1', None)\n" + \
                "decoder = Decoder('C1', 'C0', threshold, camera_C1, camera_C0, intrinsic_path)\n" + \
                'triangles = decoder.triangles\n' + \
                'x,y,z = decoder.pts3\n' + \
                'simp = decoder.simp\n' + \
                '\n'
    return statement

def test_mayavi():
    statement = 'import random\n' + \
                'import numpy as np\n' + \
                'from mayavi.mlab import *\n' + \
                'x,y,z = np.array(random.sample(range(100), 3)),np.array(random.sample(range(100), 3)),np.array(random.sample(range(100), 3))\n' + \
                'plot3d(x,y,z, representation="points")\n' + \
                '\n'
    return statement

def show(pts):
    plt.imshow(pts, cmap='gray')
    plt.show()


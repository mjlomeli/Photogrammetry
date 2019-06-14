# !/usr/bin/env python
"""Testing of Decoder.
"""

import unittest
import numpy as np
from calibrate import Calibrate
from decoder import Decoder
from pathlib import Path

DATA_FOLDER = Path.cwd() / Path("data")

__author__ = "Mauricio Lomeli"
__date__ = "6/13/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"


class TestDecoder(unittest.TestCase):

    def setUp(self):
        # TODO: write what needs to be instantiated for each test
        self.low_resolution = DATA_FOLDER / Path('calib_png_small')
        self.high_resolution = DATA_FOLDER / Path('calib_jpg_u')
        self.low_calib_C0 = Calibrate(self.low_resolution, 'C0')
        self.low_calib_C1 = Calibrate(self.low_resolution, 'C1')
        self.high_calib_C0 = Calibrate(self.high_resolution, 'C0')
        self.high_calib_C1 = Calibrate(self.high_resolution, 'C1')
        self.decoder = Decoder(testing=True)

    def test_decode(self):
        print("Testing Variables")
        printProgressBar(0, 1)
        pad = np.zeros((800, 128))
        hcode, vcode = np.meshgrid(range(1024), range(800))
        Htrue = np.concatenate((pad, hcode, pad), axis=1)
        Vtrue = np.concatenate((pad, vcode, pad), axis=1)
        masktrue = np.concatenate((pad, np.ones((800, 1024)), pad), axis=1)
        path = DATA_FOLDER / Path('testImages') / Path('gray')
        thresh = 0.0000001  # this data is perfect so we can use a very small threshold

        H, Hmask = self.decoder.decode(str(path) + '\\', 0, thresh)
        V, Vmask = self.decoder.decode(str(path) + '\\', 20, thresh)

        # compare to the known "true" code
        self.assertTrue((H == Htrue).all())
        self.assertTrue((V == Vtrue).all())
        self.assertTrue((Hmask == masktrue).all())
        self.assertTrue((Vmask == masktrue).all())
        printProgressBar(1, 1)


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """ 
    Displays a progress bar for each iteration.
    Title: Progress Bar
    Author: Benjamin Cordier
    Date: 6/10/2019
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


if __name__ == "__main__":
    print("Testing Decoder")
    unittest.main()

# !/usr/bin/env python
"""Testing of Calibrate.

Runs the calibration tests for Calibrate.
"""

import unittest
from calibrate import Calibrate
from pathlib import Path

from decoder import printProgressBar

DATA_FOLDER = Path.cwd() / Path("data")
CALIBRATION = Path.cwd() / Path("")

__author__ = "Mauricio Lomeli"
__date__ = "6/10/2019"
__maintainer__ = "Mauricio Lomeli"
__email__ = "mjlomeli@uci.edu"
__status__ = "Prototype"


class TestCalibrate(unittest.TestCase):

    def setUp(self):
        self.low_resolution = DATA_FOLDER / Path('calib_png_small')
        self.high_resolution = DATA_FOLDER / Path('calib_jpg_u')
        self.low_calib_C0 = Calibrate(self.low_resolution, 'C0')
        self.low_calib_C1 = Calibrate(self.low_resolution, 'C1')
        self.high_calib_C0 = Calibrate(self.high_resolution, 'C0')
        self.high_calib_C1 = Calibrate(self.high_resolution, 'C1')

    def testingVariables(self):
        print("Testing Variables")
        printProgressBar(0, 1)
        self.assertAlmostEqual(self.high_calib_C0.f)
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
    print("Testing Calibrate")
    unittest.main()

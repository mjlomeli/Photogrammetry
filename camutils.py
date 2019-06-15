import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

def makerotation(rx,ry,rz):
    """
    Provides a rotation matrix based on the rotation angles of each axis.
    :param rx: degree, rotation about the x-axis
    :param ry: degree, rotation about the x-axis
    :param rz: degree, rotation about the x-axis
    :return: 3D rotation matrix
    """
    x, y, z = np.deg2rad(rx), np.deg2rad(ry), np.deg2rad(ry)

    x_rot = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    y_rot = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    z_rot = np.array([[np.cos(z), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    return x_rot @ y_rot @ z_rot

class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    
    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
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
        assert(pts3.shape[0]==3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)
         
        # project
        p = self.f * (pcam / pcam[2,:])
        
        # offset principal point
        pts2 = p[0:2,:] + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0],params[1],params[2])
        self.t = np.array([[params[3]],[params[4]],[params[5]]])


def triangulate(pts2L,camL,pts2R,camR):
    """
    Triangulate the set of points seen at location pts2L / pts2R in the
    corresponding pair of cameras. Return the 3D coordinates relative
    to the global coordinate system
    :param pts2L: Coordinates of N points stored in a array of shape (2,N) seen from camL camera
    :param camL: Coordinates of N points stored in a array of shape (2,N) seen from camR camera
    :param pts2R: The first "left" camera view
    :param camR: The second "right" camera view
    :return: (3,N) array containing 3D coordinates of the points in global coordinates
    """

    npts = pts2L.shape[1]

    qL = (pts2L - camL.c) / camL.f
    qL = np.vstack((qL,np.ones((1,npts))))

    qR = (pts2R - camR.c) / camR.f
    qR = np.vstack((qR,np.ones((1,npts))))
    
    R = camL.R.T @ camR.R
    t = camL.R.T @ (camR.t-camL.t)

    xL = np.zeros((3,npts))
    xR = np.zeros((3,npts))

    for i in range(npts):
        A = np.vstack((qL[:,i],-R @ qR[:,i])).T
        z,_,_,_ = np.linalg.lstsq(A,t,rcond=None)
        xL[:,i] = z[0]*qL[:,i]
        xR[:,i] = z[1]*qR[:,i]
 
    pts3L = camL.R @ xL + camL.t
    pts3R = camR.R @ xR + camR.t
    pts3 = 0.5*(pts3L+pts3R)

    return pts3


def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations
    :param pts3: Coordinates of N points stored in a array of shape (3,N)
    :param pts2: Coordinates of N points stored in a array of shape (2,N)
    :param cam: camera to be updated
    :param params: Camera parameters we are optimizing over stored in a vector
    :return: Vector of residual 2D projection errors of size 2*N
    """
    cam.update_extrinsics(params)
    projected = cam.project(pts3)
    return (pts2 - projected).flatten()

def calibratePose(pts3,pts2,cam,params_init):
    """
    Calibrates the camera to match the view calibrated by updating R,t so that pts3 projects
    as close as possible to pts2
    :param pts3: Coordinates of N points stored in a array of shape (3,N)
    :param pts2: Coordinates of N points stored in a array of shape (2,N)
    :param cam_init: Initial estimate of camera
    :param params_init:
    :return: Refined estimate of camera with updated R,t parameters
    """

    func = lambda rt: residuals(pts3,pts2,cam,rt)
    least = scipy.optimize.leastsq(func,params_init)[0]
    cam.update_extrinsics(least)

    return cam



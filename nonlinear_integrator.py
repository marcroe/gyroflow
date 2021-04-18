
import numpy as np
import quaternion as quat
import cv2
import csv

from matplotlib import pyplot as plt

from gyro_integrator import GyroIntegrator

from scipy.spatial.transform import Rotation




class NonLinIntegrator(GyroIntegrator):

    def inverse_cam_mtx(self, K):
        # inverse for zero skew case
        if K.shape != (3,3):
            raise ValueError("Not 3x3 matrix")

        fx = K[0,0]
        fy = K[1,1]
        px = K[0,2]
        py = K[1,2]

        Kinv = np.array([[fy, 0,  -px*fy],
                         [0,  fx, -py*fx],
                         [0,  0,  fx*fy]])

        Kinv /= fx * fy

        return Kinv

    def setCalibrator(self, calibrator):
        self.dbg =0
        self.calibrator = calibrator
        self.calib_dimension = calibrator.calib_dimension
        self.K = np.copy(calibrator.K)
        self.D = np.copy(calibrator.D)

        (original_width, original_height) = self.calib_dimension
        numPoints = 9

        undistorted_points = []
        for i in range(numPoints-1):
            undistorted_points.append( (i*(original_width/(numPoints-1)), 0) )
        for i in range(numPoints-1):
            undistorted_points.append( (original_width, i*(original_height/(numPoints-1)) ) )
        for i in range(numPoints-1):
            p = numPoints-1 - i
            undistorted_points.append( (p*(original_width/(numPoints-1)), original_height) )
        for i in range(numPoints-1):
            p = numPoints-1 - i
            undistorted_points.append( (0, p*(original_height/(numPoints-1)) ) )

        self.bounderyPts = np.array(undistorted_points, np.float64)
        #self.bounderyPts = np.c_[ self.bounderyPts, np.ones(self.bounderyPts.shape[0]) ]

        #self.camPts =

        #plt.plot(undistorted_points[:,0], undistorted_points[:,1], 'ro')

        #self.camPts = np.matmul(self.bounderyPts, self.inverse_cam_mtx(self.K))
        self.camPts = self.bounderyPts - np.array([self.K[0][2], self.K[1][2]])
        self.camPts[:,0] = self.camPts[:,0] / self.K[0][0]
        self.camPts[:,1] = self.camPts[:,1] / self.K[1][1]
        self.camPts = np.c_[ self.camPts, np.ones(self.camPts.shape[0]) ]


        camPtsExt = np.expand_dims(self.camPts[:,0:2], axis=0) #add extra dimension so opencv accepts points
        distortedCam = cv2.fisheye.distortPoints(camPtsExt, self.K, self.D)
        distortedCam = distortedCam[0,:,:] #remove extra
        distortedCam = distortedCam - np.array([self.K[0][2], self.K[1][2]])# how far off center?
        distortedCam = np.abs(distortedCam)
        #print(distortedCam)

        maxY = np.max(distortedCam[:,1])/ self.K[1][2]
        maxX = np.max(distortedCam[:,0])/ self.K[0][2]

        self.scalingFactor = np.max((maxY,maxX))

        #print(self.scalingFactor)

        #plt.plot(self.bounderyPts[:,0], self.bounderyPts[:,1], 'bo')
        #plt.plot(distortedCam[:,0], distortedCam[:,1], 'yo')
        #plt.show()

        self.K[0][0] = self.K[0][0] / self.scalingFactor
        self.K[1][1] = self.K[1][1] / self.scalingFactor
        #self.camPts[:,0:2] = self.camPts[:,0:2] * self.scalingFactor

        #self.camPts = self.bounderyPts - np.array([original_width/2, original_height/2])
        #self.camPts[:,0] = self.camPts[:,0] / self.K[0][0]
        #self.camPts[:,1] = self.camPts[:,1] / self.K[1][1]

        #undistorted_points = np.expand_dims(undistorted_points, axis=0) #add extra dimension so opencv accepts points

        #undistorted_points = self.inverse_cam_mtx(self.K) * undistorted_points

        #distorted_points = cv2.fisheye.distortPoints(undistorted_points, self.K, self.D)
        #distorted_points = distorted_points[0,:,:] #remove extra dimension
        #self.distorted_points = np.c_[ distorted_points, np.ones(distorted_points.shape[0]) ]
        #print(distorted_points.shape)
        #self.distorted_points[]
        #print(self.distorted_points)

        #plt.plot(distorted_points[:,0], distorted_points[:,1], 'bo')
        #plt.show()

    def getSmoothedQuat(self, virtualVal, phyVal, smooth):
        lookahead = quat.slerp(virtualVal, phyVal, [smooth])[0]
        q = quat.rot_between(virtualVal, lookahead)
        q = q.flatten()
        #R = Rotation([0.7071068,0,0,0.7071068]).as_matrix()
        R = Rotation([-q[1],-q[2],q[3],-q[0]]).as_matrix() #inverse rotation
        #R = np.array([ [  1.0000000,  0.0000000,  0.0000000],
        #                    [0.0000000,  0.9396926,  0.3420202],
        #                    [0.0000000, -0.3420202,  0.9396926 ] ])
        rotated_pts =  np.matmul(self.camPts, R)
        rotated_pts[:,0] =  rotated_pts[:,0] #/ rotated_pts[:,2]
        rotated_pts[:,1] =  rotated_pts[:,1] #/ rotated_pts[:,2]
        rotated_ptsExp = np.expand_dims(rotated_pts[:,0:2], axis=0) #add extra dimension so opencv accepts points
        distorted_points = cv2.fisheye.distortPoints(rotated_ptsExp, self.K, self.D)
        distorted_points = distorted_points[0,:,:] #remove extra dimension
        distorted_pointsComp = distorted_points - np.array([self.K[0][2], self.K[1][2]])# how far off center?
        distorted_pointsComp = np.abs(distorted_pointsComp)
        dist = np.max(distorted_pointsComp, axis=0) - np.array([self.K[0][2], self.K[1][2]])
        maxDist = np.max(dist)
        maxDist = np.max((maxDist,0))
        self.dbg = self.dbg + 1
        if self.dbg < 0 and ((self.dbg % 30) == 0):
            print(self.dbg)
            print(q)
            print(maxDist)
            plt.plot(distorted_pointsComp[:,0], distorted_pointsComp[:,1], 'ro')
            plt.plot(self.bounderyPts[:,0], self.bounderyPts[:,1], 'bo')
            plt.plot(distorted_points[:,0], distorted_points[:,1], 'yo')
            plt.plot(self.camPts[:,0], self.camPts[:,1], 'go')
            plt.plot(rotated_pts[:,0], rotated_pts[:,1], 'ro')
            plt.show()

        remSmooth = np.min([ (1 - smooth), (1 - smooth) * maxDist/self.K[0][2]])
        totSmooth = smooth+remSmooth
        #print(totSmooth)
        return quat.slerp(virtualVal, phyVal, [totSmooth])[0]

    def get_smoothed_orientation(self, smooth = 0.94):
        # https://en.wikipedia.org/wiki/Exponential_smoothing
        # the smooth value corresponds to the time constant

        alpha = 1
        if smooth > 0:
            alpha = 1 - np.exp(-(1 / self.gyro_sample_rate) /smooth)

        smoothed_orientation = np.zeros(self.orientation_list.shape)

        value = self.orientation_list[0,:]

        #print(self.orientation_list)
        #print(self.orientation_list.shape)
        #exit()

        for i in range(self.num_data_points):
            value = self.getSmoothedQuat(value, self.orientation_list[i,:], alpha)
            smoothed_orientation[i] = value

        # reverse pass
        smoothed_orientation2 = np.zeros(self.orientation_list.shape)

        value2 = smoothed_orientation[-1,:]

        for i in range(self.num_data_points-1, -1, -1):
            value2 = self.getSmoothedQuat(value2, smoothed_orientation[i,:], alpha)
            smoothed_orientation2[i] = value2

        return (self.time_list, smoothed_orientation2)

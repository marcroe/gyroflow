import numpy as np
import cv2
import csv
import platform

from stabilizer import BBLStabilizer
from freqAnalysis import FreqAnalysis
from calibrate_video import FisheyeCalibrator, StandardCalibrator
from scipy.spatial.transform import Rotation
from gyro_integrator import GyroIntegrator, FrameRotationIntegrator
from nonlinear_integrator import NonLinIntegrator
from adaptive_zoom import AdaptiveZoom
from blackbox_extract import BlackboxExtractor
from GPMF_gyro import Extractor
from matplotlib import pyplot as plt
from vidgear.gears import WriteGear
from vidgear.gears import helper as vidgearHelper
from _version import __version__

from scipy import signal, interpolate

import time

import insta360_utility as insta360_util

class StabTestCase(BBLStabilizer):
    def manual_sync_correctionCLI(self, d1, d2, sliceframe1, sliceframe2, slicelength, smooth):
        v1 = (sliceframe1 + slicelength/2) / self.fps
        v2 = (sliceframe2 + slicelength/2) / self.fps

        print("v1: {}, v2: {}, d1: {}, d2: {}".format(v1, v2, d1, d2))

        g1 = v1 - d1
        g2 = v2 - d2
        slope =  (v2 - v1) / (g2 - g1)
        corrected_times = slope * (self.integrator.get_raw_data("t") - g1) + v1
        print("Gyro correction slope {}".format(slope))

        initial_orientation = Rotation.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()

        new_gyro_data = np.copy(self.gyro_data)

        # Correct time scale
        new_gyro_data[:,0] = slope * (self.integrator.get_raw_data("t") - g1) + v1 # (new_gyro_data[:,0]+gyro_start) *correction_slope

        new_integrator = NonLinIntegrator(new_gyro_data,zero_out_time=False, initial_orientation=initial_orientation)
        new_integrator.setCalibrator(self.undistort)
        new_integrator.integrate_all()
        self.last_smooth = smooth
        self.times, self.stab_transform = new_integrator.get_interpolated_stab_transform(smooth=smooth,start=0,interval = 1/self.fps)



if __name__ == "__main__":

    #test case
    stab = StabTestCase('/home/mroe/fpv_local/walchwil/tarsier/LOOP0095.mp4',
        '/home/mroe/gyroflow/camera_presets/Caddx/Caddx_Tarsier_4K_F_2_8_2160p_16by9.json',
        '/home/mroe/fpv_local/walchwil/tarsier/btfl_002.bbl.csv', fov_scale=1.5, cam_angle_degrees=10.0,
                                         use_csv=True, gyro_lpf_cutoff = -1, logtype='')
    stab.set_initial_offset(5.0)
    stab.set_rough_search(10.0)
    #stab.auto_sync_stab(0.24, 870, 2100, 120, debug_plots=True)
    #stab.manual_sync_correction(5.4744, 5.6012, smooth=0.24)
    stab.manual_sync_correctionCLI(5.4744, 5.6012, 870, 2100, 120, 0.24)
    stab.renderfile(63, 80, outpath = "/home/mroe/fpv_local/walchwil/tarsier/LOOP0095_stab-2.mp4", out_size = (3840,2160), split_screen = False,
                   bitrate_mbits = 20, display_preview = True, scale=1, vcodec = "libx264", vprofile="high", pix_fmt = "",
                   debug_text = True, custom_ffmpeg = "", zoom=0.6, smoothingFocus=1)
    exit()

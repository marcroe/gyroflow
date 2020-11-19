# Gyroflow - Video stabilization using gyroscope data targeting drone footage (WIP)

A program built around Python, OpenCV, and PySide2 for video stabilization using gyroscope data.

The project consists of three core parts: A utility for the generation of lens undistortion preset, a utility for stabilizing footage using gyro data, and a utility for stretching 4:3 video to 16:9 using non-linear horizontal stretching (similar to GoPro's superview).

This is currently a work in progress project, but the goal is to use the gyro data logged by drone flight controllers for stabilizing the onboard HD camera. Furthermore, the gyro data embedded in newer GoPro cameras should also be usable for stabilization purposes.

The launcher containing all the utilities is available by executing `gyroflow.py` if all the dependencies are met. Otherwise a (possibly outdated and/or buggy) binary can be found over in [releases](https://github.com/ElvinC/gyroflow/releases). The current release was for testing pyinstaller. This will be updated once the stabilization code works.

Also check out the [blackbox2gpmf](https://github.com/jaromeyer/blackbox2gpmf) project by jaromeyer for stitching blackbox data to Hero 7 files for use with Reelsteady Go.

### Status

[Latest test clip.](https://youtu.be/ZhVVRnuuMFc)

Working:
* Videoplayer based on OpenCV and Pyside2
* Gyro integration using quaternions
* Non-linear stretch utility
* Basic video import/export
* Camera calibration utility with preset import/export
* GoPro metadata import
* Symmetrical quaternion low-pass filter
* Blackbox data import
* Undistort and rotation perspective transform

Work in progress:
* Automatic/semi-automatic temporal gyro/video sync. Basic sync working (bit finnicky and needs tweaking)
* Camera orientation determination with respect to gyro

Not working (yet) and potential future additions:
* Gyro orientation presets
* Stabilization UI
* Incorporate acceleration data in orientation estimation for horizon lock (Complementary filter? Kalman is probably overkill but could be fun to learn).
* Rolling shutter determination/correction (may or may not be required)
* Improved low-pass filter and more stabilization modes (Time-lapse, separate pitch/yaw/roll smoothness control etc.)
* Streamlining/optimizing the image processing pipeline
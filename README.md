## Advanced Lane Finding
![Lanes Image](./example.jpg)


Overview:
---
Detect lanes using computer vision techniques. 

The following steps were performed for lane detection:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[Here](https://youtu.be/ntkTFBudEps) is the final video output on Youtube. The same video is 'out_video.mp4' in this repo. The original video is 'project_video.mp4'.

## Dependencies
* Python 3.6
* Numpy
* OpenCV-Python
* Matplotlib
* Pickle

you can use requirements.txt to install all requirements 

`pip install -r requirements.txt`

How it work
---

#### 1 -  calibrate camera

we save our calibration images <b>(chess board images)</b> and pass its path to camera_calibration module

Example

`python camera_calibration.py -p camera_cal -o cam_calibration -nx 9 -ny 6`

#### where..

`-p: path to calibration images folder`

`-o: name of output pickle file which contain mtx amd dist of our camera`

`-nx: number of internal corner in row`

`-ny: number of internal corner in column`

#### 2 - Apply Lane detection on video or cam 
we use main module which take different options

`-p: path to pickle file contain perspective transform matrix M and Minv, you can use init if you want generate new one, then pass path to save it in -np`

`-np: path to save generated M and Minv as pickle file if not given will save in current dir`

`-s: source of video, if cam, apply edge detection on real time camera, else source should be path to local video`

`-c: path to Pickle file contain camera calibration parameters [ mtx , dist]`

`-sp: Path to save generated Video, if not given output video won't be saved, use _c to save in current dir`

`-n: name of output video if saved path is given`

Example

`python main.py -p perspective_tf.pickle -s project_video.mp4 -c cam_calibration.pickle -sp _c -n out_video.mp4`

what is next
---
This work fine for good conditions roads, so next step to make it work fine in harder situations like dark frames, noise line.

Any suggestions to make it better are welcomed ^^
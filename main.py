import argparse
import os
import pickle
import sys
import numpy as np
import cv2

from edge_detection import apply_edge_detection, sliding_window, search_around_poly
from helper import perspective_tf, save_pickle, warped_img, write_text
from line import Line


def video(perspective_matrix_path, source="cam", save=False, save_path=None, file_name="out", cam_cal=None):
    """
    apply edge detection on video frames

    :param perspective_matrix_path: Path to pickle file contain transform matrix M and Minv
    :param source: source of video, if cam, apply edge detection on real time camera,
    else source should be path to local video
    :param save: True if want to save output video, available only when source is video not camera
    :param save_path: path to save output video, if save is True
    :param file_name: name of output video if saved is True, default = out
    :param cam_cal: path to Pickle file contain camera calibration parameters [ mtx , dist]
    :return: None
    """
    if not os.path.isfile(perspective_matrix_path):
        raise FileNotFoundError("Path to perspective matrix file not exist!")

    with open(perspective_matrix_path, "rb") as p:
        perspective_matrix = pickle.load(p)
        M = perspective_matrix["M"]
        Minv = perspective_matrix["Minv"]

    if source == "cam":
        cap = cv2.VideoCapture(0)
    else:
        if not os.path.isfile(source):
            raise FileNotFoundError(source, " not Exist!")
        cap = cv2.VideoCapture(source)

    # camera calibration parameters [ mtx , dist]
    mtx = None
    dist = None

    out = None
    if save:
        if not os.path.isdir(save_path):
            raise FileNotFoundError(save_path, " Not Exist!")
        file_name += ".mp4"
        out = cv2.VideoWriter(save_path + file_name, -1, 20, (int(cap.get(3)), int(cap.get(4))))

    if cam_cal:
        if not os.path.isfile(cam_cal):
            raise FileNotFoundError(cam_cal, " Not Exist!")

        with open(cam_cal, "rb") as p:
            calibration = pickle.load(p)
            mtx = calibration["mtx"]
            dist = calibration["dist"]

    left_line = Line(5)
    right_line = Line(5)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Finished..")
            sys.exit(0)

        # cv2 read frame as BGR, convert it to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # camera calibration
        if not (mtx is None or dist is None):
            frame = cv2.undistort(frame, mtx, dist, None, mtx)

        # get edges in image
        edges = apply_edge_detection(frame)

        # transform image to bird view
        warped = warped_img(edges, M)

        # init out image which will draw lane line on it then weight it with original frame
        out_img = np.zeros_like(warped)
        if len(warped.shape) == 3 and warped.shape[2] == 3:
            pass
        else:
            out_img = np.dstack((out_img, out_img, out_img))

        # if line not detected, apply sliding window
        if not left_line.detected or not right_line.detected:
            leftx, lefty, rightx, righty = sliding_window(warped, 9, 200)

        # if already detected apply search around detected line
        else:
            leftx, lefty = search_around_poly(left_line, warped)
            rightx, righty = search_around_poly(right_line, warped)

        # will used for plotting line, find x fitted
        ploty = np.linspace(warped.shape[0] // 4, warped.shape[0] - 1, warped.shape[0])

        # check if at least 100 pixels detected as line
        if len(leftx) > 100 and len(rightx) > 100:

            # make detected flag true
            left_line.detected = True
            right_line.detected = True

            left_line.current_x = leftx
            left_line.current_y = lefty

            right_line.current_x = rightx
            right_line.current_y = righty

            left_line.fit_polynomial(ploty)
            right_line.fit_polynomial(ploty)

        else:
            print("Line not detected in this frame ")
            # we just draw line form previous frame

            # make detected flag true
            left_line.detected = False
            right_line.detected = False

        # update Lane line radius
        left_line.radius()
        right_line.radius()

        # avg radius of to lines, and plot it
        radius = (left_line.radius_of_curvature + right_line.radius_of_curvature) // 2
        frame = write_text(frame, "Radius of Curvature = " + str(radius) + " M", pos=(20, 50))

        # calculate Alignment ( how much car away from center between Lane lines
        dir = "Left"  # car far from left or right

        left_line.car_offset(frame.shape)  # distance from left line
        right_line.car_offset(frame.shape)  # distance from right line

        distance = round(right_line.line_base_pos - left_line.line_base_pos, 2)

        if distance < 0:  # car far away from left line not right line
            distance = -distance
            dir = "Right"
        frame = write_text(frame, "Vehicle is {}m {} of center".format(distance, dir), pos=(20, 80))

        # ** plot lane lines on image **
        # left_line.draw_line(out_img, ploty)
        # right_line.draw_line(out_img, ploty)

        # color pixel which belong to lane lines
        left_line.color_pixel(out_img, (255, 0, 0))
        right_line.color_pixel(out_img, (255, 100, 0))

        # fit green triangle in area between lane lines
        pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(out_img, np.int_([pts]), (0, 255, 0))

        # return image to normal view from bird view
        out_img_undit = warped_img(out_img, Minv)

        # weight out_image_undit with original frame
        frame = cv2.addWeighted(out_img_undit, 0.5, frame, 1, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        cv2.imshow("frame", frame)

        # write video
        if save:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def perspective_tf_init(name=None, save_path=""):
    """
    Initiate perspective tf matrix and save it if path is given to use it
    :param name: name of file we will save
    :param save_path: path to save M and Minv as pickle file if not given will save in current dir
    :return: M and Minv, None if save path is given
    """

    src = np.float32(
        [[200, 720],
         [1100, 720],
         [595, 450],
         [685, 450]])

    dst = np.float32(
        [[300, 720],
         [980, 720],
         [300, 0],
         [980, 0]])

    # src = np.float32([[250, 700], [1200, 700], [550, 450], [750, 450]])
    # dst = np.float32([[250, 700], [1200, 700], [300, 50], [1000, 50]])

    M, Minv = perspective_tf(src, dst)
    if not name:
        return M, Minv

    dic = {"M": M,
           "Minv": Minv}

    save_pickle(dic, name, save_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--perspective", required=True,
                    help="Path to pickle file contain perspective transform matrix M and Minv, you can use init if "
                         "you want generate new one, then pass path to save it in -np")

    ap.add_argument("-np", "--new_perspective",
                    help="path to save generated M and Minv as pickle file if not given will save in current dir",
                    default=None)

    ap.add_argument("-s", "--source",
                    help=" source of video, if cam, apply edge detection on real time camera,\
                     else source should be path to local video", default="cam")

    ap.add_argument("-c", "--calibration",
                    help=" path to Pickle file contain camera calibration parameters [ mtx , dist]",
                    default=None)

    ap.add_argument("-sp", "--save_path",
                    help="Path to save generated Video, if not given output video won't be saved, use _c to save"
                         " in current dir", default=False)

    ap.add_argument("-n", "--name",
                    help="name of output video if saved path is given", default="out.mp4")

    args = vars(ap.parse_args())

    perspective_matrix_path = args['perspective']
    source = args["source"]
    cam_cal = args["calibration"]
    save_path = args["save_path"]
    save = False
    file_name = args["name"]
    if perspective_matrix_path == "init":
        new_perspective_path = args["new_perspective"]
        perspective_tf_init("perspective_tf.pickle", new_perspective_path)
        perspective_matrix_path = new_perspective_path + "//perspective_tf.pickle"

    if save_path:
        save = True
        if save_path == "_c":
            save_path = "./"

    video(perspective_matrix_path=perspective_matrix_path, source=source, cam_cal=cam_cal, save=save,
          save_path=save_path, file_name=file_name)

    # From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import pyrealsense2 as rs
import numpy as np
from pythonosc import udp_client
from pythonosc import osc_message_builder
import random
import time
import math as m

IP = '127.0.0.1'
PORT = 12345
client = udp_client.SimpleUDPClient(IP, PORT)


fr = 30

depthW = 320
depthH = 240
rgbW = 640
rgbH = 480

w = rgbW
h = rgbH

def sendPose_multi(pose, client):
    client.send_message('/tracker', pose3d.flatten())

def sendPose_multiLine(pose, poseAngles, client):
    for pid, person in enumerate(pose):
        msg = pose[pid, :, :].flatten()
        msg = np.concatenate(([pid], msg), axis=None)
        msg = np.concatenate((msg, poseAngles), axis=None)
        client.send_message('/tracker', msg)

        #client.send_message('/tracker', pose3d.flatten())

jointThreshold = 0.
jointMap = np.array([[0, 1],
                     [1, 2],
                     [1, 5],
                     [2, 3],
                     [3, 4],
                     [5, 6],
                     [6, 7],
                     [1, 8],
                     [8, 9],
                     [8, 12],
                     [9, 10],
                     [10, 11],
                     [11, 22],
                     [22, 23],
                     [11, 24],
                     [12, 13],
                     [13, 14],
                     [14, 19],
                     [19, 20],
                     [14, 21],
                     [0, 15],
                     [0, 16],
                     [15, 17],
                     [16, 18]])


def drawPose_multi(img, pose):
    for person in pose:
        joints = person
        for joint in jointMap:
            if joints[joint[0], 0] > jointThreshold and joints[joint[1], 1] > jointThreshold:
                x1 = int(joints[joint[0], 0])
                y1 = int(joints[joint[0], 1])
                x2 = int(joints[joint[1], 0])
                y2 = int(joints[joint[1], 1])
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(img, (x1, y1), 2, (0, 0, 255), -1)
                cv2.circle(img, (x2, y2), 2, (0, 0, 255), -1)


def drawPose(img, pose):
    for joint in jointMap:
        if pose[0, joint[0], 2] > jointThreshold and pose[0, joint[1], 2] > jointThreshold:
            x1 = int(pose[0, joint[0], 0])
            y1 = int(pose[0, joint[0], 1])
            x2 = int(pose[0, joint[1], 0])
            y2 = int(pose[0, joint[1], 1])
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)

    for i in range(pose.shape[1]):
        x = int(pose[0, i, 0])
        y = int(pose[0, i, 1])
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

def drawCoords(img, pose, p3d):
    for i in range(pose.shape[1]):
        x = int(pose[0, i, 0])
        y = int(pose[0, i, 1])
        #coord = f'{p3d[0, i, 0]:.2f} , {p3d[0, i, 1]:.2f} , {p3d[0, i, 2]:.2f}'
        coord = f'{p3d[0, i, 2]:.2f}'
        cv2.putText(img, coord, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1)


def get3dCoords(depthImg, depthFrame, depthIntrinsics, pose):
    pose3d = np.zeros(pose.shape)
    for pid, person in enumerate(pose):
        for jid, joint in enumerate(person):
            x = np.clip(joint[0], 0, w-1).astype(int);
            y = np.clip(joint[1], 0, h-1).astype(int);
            coord = [x, y, depthFrame.get_distance(x, y)]
            pose3d[pid, jid, :] = coord
            #pose3d[pid, jid, :] = rs.rs2_deproject_pixel_to_point(depthIntrinsics, pose[pid, jid, 0:2], depthFrame.get_distance(x, y))
    return pose3d

def normalizePose(pose, maxDepth):
    normalizedPose = pose;
    normalizedPose[:, :, 0] = normalizedPose[:, :, 0]/rgbW;
    normalizedPose[:, :, 1] = normalizedPose[:, :, 0]/rgbH;
    normalizedPose[:, :, 1] = normalizedPose[:, :, 0]/maxDepth;

    return normalizedPose;

def vecMod(v):
    return m.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def getLimbAngles(pose):
    poseAngles = np.zeros((pose.shape[0], 4))

    #print(poseAngles.shape)

    for pid, person in enumerate(pose):
        v1 = pose3d[pid, 4, :] - pose3d[pid, 3, :]
        v2 = pose3d[pid, 2, :] - pose3d[pid, 3, :]

        v3 = pose3d[pid, 8, :] - pose3d[pid, 1, :]
        v4 = pose3d[pid, 3, :] - pose3d[pid, 2, :]

        v5 = pose3d[pid, 7, :] - pose3d[pid, 6, :]
        v6 = pose3d[pid, 5, :] - pose3d[pid, 6, :]

        v7 = pose3d[pid, 8, :] - pose3d[pid, 1, :]
        v8 = pose3d[pid, 6, :] - pose3d[pid, 5, :]

        poseAngles[pid, 0] = np.arccos(np.clip(np.dot(v1, v2)/(vecMod(v1)*vecMod(v2)), -1., 1.))*180./m.pi
        poseAngles[pid, 1] = np.arccos(np.clip(np.dot(v3, v4)/(vecMod(v3)*vecMod(v4)), -1., 1.))*180./m.pi
        poseAngles[pid, 2] = np.arccos(np.clip(np.dot(v5, v6)/(vecMod(v5)*vecMod(v6)), -1., 1.))*180./m.pi
        poseAngles[pid, 3] = np.arccos(np.clip(np.dot(v7, v8)/(vecMod(v7)*vecMod(v8)), -1., 1.))*180./m.pi
    return poseAngles


#####################################
#OpenPose
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path);
os.environ['PATH']  = os.environ['PATH'] +  dir_path + ';'
import pyopenpose as op

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
args = parser.parse_known_args()
params = dict()
params["model_folder"] = "models/"

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1])-1: next_item = args[1][i+1]
    else: next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace('-','')
        if key not in params:  params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace('-','')
        if key not in params: params[key] = next_item

#start openpose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
############################################################


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, depthW, depthH, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, rgbW, rgbH, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, rgbW, rgbH, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 4 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        bg_removed_resized = cv2.resize(bg_removed, (w, h), interpolation = cv2.INTER_AREA)

        datum = op.Datum()
        imageToProcess = bg_removed_resized
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        if datum.poseKeypoints is not None:
            pose3d = get3dCoords(depth_image, aligned_depth_frame, depth_intrinsics, datum.poseKeypoints)
            poseAngles = getLimbAngles(pose3d)
            #print(pose3d)
            drawPose_multi(imageToProcess, pose3d);
            #print(datum.poseKeypoints.shape)

            #sendPose_multi(pose3d, client)
            npose = normalizePose(pose3d, 1);
            sendPose_multiLine(npose, poseAngles, client)
            #drawPose(bg_removed_resized, datum.poseKeypoints)


        #print(pose3d)
        # Display Image
        #depthColorMapped = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #drawCoords(bg_removed_resized, datum.poseKeypoints, pose3d)
        # Render images:
        #   depth align to color on left
        #   depth on right
        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        #images = np.hstack((bg_removed, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align Example', bg_removed_resized)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

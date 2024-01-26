import cv2
import numpy as np
import time

from pythonosc import udp_client
from pythonosc import osc_message_builder


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model_path = 'pose_landmarker_heavy.task'
model_path = r'/Users/jessicarosendorf/Documents/untitled folder 2/pose_landmarker_heavy.task'

# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

IP = '10.20.17.122'
PORT = 5557
client = udp_client.SimpleUDPClient(IP, PORT)

def send_one_angle(angle, client):
    client.send_message('/tracker', [angle])

# Create a pose landmarker instance with the live stream mode:
def print_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('pose landmarker result: {}'.format(result))


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image


def pose_from_image():
    
    image_file_name = r'/Users/jessicarosendorf/Documents/Hackathon/body2vec/girl-4051811_960_720.jpg'
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)
    image = mp.Image.create_from_file(image_file_name)
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    cv2.imshow('annotated', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

def calc_angle(point_a, point_b, point_c):
    vec1 = point_a - point_b
    vec2 = point_c - point_b
    dot_product = np.dot(vec1, vec2)
    mag1 = np.linalg.norm(vec1)
    mag2 = np.linalg.norm(vec2)
    cos_angle = dot_product/(mag1*mag2)
    angle = np.arccos(cos_angle)
    return angle

def get_joint_angles(results, fig_shape):
    joint_map = {'rwrist':16, 'relbow':14, 'rshoulder':12, 'lwrist':15, 'lelbow':13, 'lshoulder':11}
    # try:
    landmarks_list = results.pose_landmarks[0]
    # catch:
    #     return None
    pose = np.zeros((len(landmarks_list),3))
    for jid, landmark in enumerate(landmarks_list):
        pose[jid,:] = [landmark.x*fig_shape[0], landmark.y*fig_shape[1], landmark.z]
    if landmarks_list[joint_map['relbow']].presence > 0.5:
        r_elbow_ang = calc_angle(pose[joint_map['rwrist'],:], pose[joint_map['relbow'],:], pose[joint_map['rshoulder'],:])
        return r_elbow_ang
    else:
        return np.nan

def pose_from_webcam():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path))
    # running_mode=VisionRunningMode.LIVE_STREAM,
    # result_callback=print_result)
    cap = cv2.VideoCapture(0)

    with PoseLandmarker.create_from_options(options) as landmarker:
        # start_ms = datetime.now()
        while(True):
            ret, frame = cap.read()
            width, height, _ = frame.shape
            fig_shape = (width, height)
            time_d = round(int(time.time()*1000))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            result = landmarker.detect(mp_image)
            if cv2.waitKey(1) == ord('q'):
                break
            result_im = draw_landmarks_on_image(frame, result)
            if len(result.pose_landmarks) > 0:
                angle = get_joint_angles(result,fig_shape)*360/(2*np.pi)
            else:
                angle = np.nan
            send_one_angle(angle, client)
            cv2.putText(result_im, str(angle), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 1)
            cv2.imshow('result',result_im)
        cap.release()
        cv2.destroyAllWindows()
    

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


if __name__ == '__main__':
    pose_from_webcam()
    # pose_from_image()
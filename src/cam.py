import cv2

cap = cv2.VideoCapture(0)  # Initialize the webcam capture

# while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#         continue

#     # Process the image and draw pose landmarks
#     image.flags.writeable = False
#     results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     image.flags.writeable = True

#     mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

#     cv2.imshow('MediaPipe Pose', image)
#     if cv2.waitKey(5) & 0xFF == 27:
#         break

cap.release()
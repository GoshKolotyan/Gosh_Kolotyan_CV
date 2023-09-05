from ultralytics import YOLO
import numpy as np
import cv2
import math



# Function to calculate angle between three points
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# Function to calculate hip angle
def calculate_hip_angle(points):
    hip_left = (points[11][0], points[11][1])
    knee_left = (points[13][0], points[13][1])
    shoulder_left = (points[5][0], points[5][1])
    hip_angle = calculate_angle(shoulder_left, hip_left, knee_left)
    return hip_angle

# Function to calculate knee angle
def calculate_knee_angle(points):
    hip_left = (points[11][0], points[11][1])
    knee_left = (points[13][0], points[13][1])
    ankle_left = (points[15][0], points[15][1])
    knee_angle = calculate_angle(hip_left, knee_left, ankle_left)
    return knee_angle

def calculate_back_angle(points):
    left_hip = (points[11][0], points[11][1])
    left_shoulder = (points[5][0], points[5][1])
    horizontal_left = (left_hip[0] + 1, left_hip[1])
    angle_of_back = calculate_angle(left_shoulder, left_hip, horizontal_left)
    return angle_of_back

def find_org(res):
    if res:
        id = 0  # Define 'id' with a default value
        if id in res.boxes.cls:
            point = np.array(res.boxes.xyxy[id])
            return (int(point[0]), int(point[1]))
        else:
            return False
    return False  # Return a default value if 'res' is None or 'id' is not found





font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 0, 0)
thickness = 1

class FallDetection():
    def __init__(self):
      return

    def __call__(self, M_landmarks_array):
        hip_angles = []
        knee_angles = []
        back_angles = []
        M = M_landmarks_array.shape[0]
        for points in M_landmarks_array:
          hip_angles.append(calculate_hip_angle(points))
          knee_angles.append(calculate_knee_angle(points))
          back_angles.append(calculate_back_angle(points))
        dif_hip_between_frames = []
        dif_knee_between_frames = []
        dif_back_between_frames = []
        for i in range(M-1):
          dif_hip_between_frames.append(abs(hip_angles[i+1]-hip_angles[i]))
          dif_knee_between_frames.append(abs(knee_angles[i+1]-knee_angles[i]))
          dif_back_between_frames.append(abs(back_angles[i+1]-back_angles[i]))
        sum_hip = sum(dif_hip_between_frames)
        sum_knee = sum(dif_knee_between_frames)
        sum_back = sum(dif_back_between_frames)

        conf = round((sum_back/M/7), 2)
        if conf > 1.0:
            conf = 1.0
        if conf > 0.5:
          return "Fall Detected: " + str(conf)
        else:
          return "Not a Fall: " + str(conf)
        # return(str(sum_hip)+' '+str(sum_knee)+' '+str(sum_back))






cap = cv2.VideoCapture('video_2.mp4')




frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 10  # Frames per second

M_landmarks = []
M = 5

model = YOLO('yolov8m-pose.pt')
mod = YOLO('yolov8m.pt')

M_landmarks = []
M = 5
out = cv2.VideoWriter('video_output_2.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

for frame_index in range(0, int(frame_count)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, image = cap.read()
    results = model.predict(image, conf=0.25)
    result = results[0]
    points = np.array(result.keypoints.xy[0])
    M_landmarks.append(points)
    M_landmarks_array = np.array(M_landmarks)

    if M_landmarks_array.shape[0] == M:
        fall = FallDetection()
        text = fall(M_landmarks_array)
        if "Fall Detected" in text:
            text_color = (255, 0, 0)  # Red for "Fall Detected"
        else:
            text_color = (0, 255, 0)  # Green for other messages
        ress = mod.predict(image)
        res = ress[0]
        new_image = image
        # Example usage:
        if find_org(res):
            new_image = cv2.putText(image, text, find_org(res), font, fontScale, text_color, thickness, cv2.LINE_AA)
        else:
            new_image = image
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        out.write(new_image)
        if len(points):
            print(text, find_org(res), end = '\n')
        M_landmarks.pop(0)
    print(frame_index)


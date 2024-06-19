import face_recognition
import os
from PIL import Image as ImagePIL
import numpy as np
import pandas as pd
import cv2
from cvzone.PoseModule import PoseDetector
from database import Conexiones as con
def mascaraFrame(numFrame,video_path):
    video_tag = os.path.split(video_path)
    video_tag = video_tag[1]
    video_tag = video_tag.split(".")[0]
    video_final = "videos-mascaras/"+video_tag+"/"+"mask.avi"
    # Open the video file
    cap = cv2.VideoCapture(video_final)

    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_final}")
        exit()
    # Set the frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, numFrame)

    # Read the frame
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if ret:
        # Display the frame
        cap.release()
        return frame
    else:
        print(f"Error: Could not read frame {numFrame}")

    # Release the video capture object and close all OpenCV windows
    cap.release()
    
def procesarVideo(video_path,Numframe,etiqueta):
    # Initialize the PoseDetector
    detector = PoseDetector()
    respond = None
    maxFrames = 63
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    output_path = "output/"+str(etiqueta)+"_"+str(Numframe)+".png"
    
    # Check if video file opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Initialize variables to store the bounding box dimensions
    rect_x, rect_y, rect_w, rect_h = 0, 0, 0, 0
    

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        exit()
    cap.set(cv2.CAP_PROP_POS_FRAMES, Numframe)

    
    ret, frame = cap.read()
    if not ret:
        exit()
    
    # Find the pose in the frame
    img = detector.findPose(frame, draw=False)
    
    # Get the list of landmarks
    lmList, _ = detector.findPosition(img, bboxWithHands=False)
    
    if lmList:
        # Calculate bounding box dimensions based on landmarks
        x_min = min(lmList, key=lambda x: x[0])[0]
        x_max = max(lmList, key=lambda x: x[0])[0]
        y_min = min(lmList, key=lambda x: x[1])[1]
        y_max = max(lmList, key=lambda x: x[1])[1]
        
        # Update rectangle dimensions
        rect_x = x_min
        rect_y = y_min
        rect_w = x_max - x_min
        rect_h = y_max - y_min
    
    # Ensure rectangle is within frame boundaries
    rect_x = max(0, rect_x)
    rect_y = max(0, rect_y)
    rect_w = min(width - rect_x, rect_w)
    rect_h = min(height - rect_y, rect_h)
    
    if rect_w > 0 and rect_h > 0:
        # Draw the rectangle on the frame
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 0, 0), 2)
        
        # Crop the frame to the rectangle region
        
        maskFrame = mascaraFrame(Numframe,video_path)
        cropped_frame = maskFrame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

        if cropped_frame.size != 0:
            # Resize cropped frame back to original frame size
            resized_cropped_frame = cv2.resize(cropped_frame, (150, 150))
            resized_cropped_frame = cv2.cvtColor(resized_cropped_frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(output_path,resized_cropped_frame)
            respond = resized_cropped_frame
            
    # Display the frame with the pose and bounding box (for debugging purposes)
    #cv2.imshow('Frame', frame)
   

    # Release the video capture and writer objects and close all OpenCV windows
    cap.release()
    return respond


def getKeypoints(etiqueta,frame):
    cur, connectio = con.conexion()
    cur.execute("select nose,left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle,mid_shoulder,mid_hip from fun_generar_reporte_keypoints('"+etiqueta+"') where CAST(frame as integer) = "+str(frame))
    keypoints = cur.fetchall()
    return keypoints
    
    

def getAngles(etiqueta,frame):
    cur, connectio = con.conexion()
    cur.execute("select nose_mid_shoulder,mid_shoulder_mid_hip,left_shoulder_left_elbow,left_elbow_left_wrist,right_shoulder_right_elbow,right_elbow_right_wrist,left_hip_left_knee,left_knee_left_ankle,right_hip_right_knee,right_knee_right_ankle,mid_shoulder_angle,left_shoulder_angle,left_elbow_angle,right_shoulder_angle,right_elbow_angle,left_hip_angle,left_knee_angle,right_hip_angle,right_knee_angle from fun_generar_reporte_angles('"+etiqueta+"') where CAST(frame as integer) = "+str(frame))
    angles = cur.fetchall()
    return angles


videos = "video-input"
output_dir = "output"
features = []
labels = []
datos_imagen_recortada = []
directorio_b = {'Mascara': [], 'Keypoints': [], 'Angles': [], 'Etiqueta': []}
directorio = {'frame': [],'mascara':[], 'nose':[],'left_shoulder':[],'right_shoulder':[],'left_elbow':[],'right_elbow':[],
            'left_wrist':[],'right_wrist':[],'left_hip':[],'right_hip':[],'left_knee':[],'right_knee':[],'left_ankle':[],
            'right_ankle':[],'mid_shoulder':[],'mid_hip':[],
            'nose_mid_shoulder':[],'mid_shoulder_mid_hip':[],'left_shoulder_left_elbow':[],'left_elbow_left_wrist':[],
            'right_shoulder_right_elbow':[],'right_elbow_right_wrist':[],'left_hip_left_knee':[],'left_knee_left_ankle':[],
            'right_hip_right_knee':[],'right_knee_right_ankle':[],'mid_shoulder_angle':[],'left_shoulder_angle':[],
            'left_elbow_angle':[],'right_shoulder_angle':[],'right_elbow_angle':[],'left_hip_angle':[],'left_knee_angle':[],
            'right_hip_angle':[],'right_knee_angle':[],
            'Etiqueta': []}
for carpetas in os.listdir(videos):
    if carpetas == "MARIO":
        ruta = os.path.join(videos, carpetas)
        for video_file in os.listdir(ruta):
            try:
                etiqueta = carpetas
                print(carpetas +" "+video_file)
                video_path = os.path.join(ruta, video_file)
                video_file = video_file.split(".")[0]
                for i in range(0,63):
                    frame = i+1
                    directorio['frame'].append(frame)
                    directorio['mascara'].append(procesarVideo(video_path,i,video_file))
                    keypoints = getKeypoints(video_file,frame)
                    angles = getAngles(video_file,frame)
                    directorio['nose'].append(keypoints[0][0])
                    directorio['left_shoulder'].append(keypoints[0][1])
                    directorio['right_shoulder'].append(keypoints[0][2])
                    directorio['left_elbow'].append(keypoints[0][3])
                    directorio['right_elbow'].append(keypoints[0][4])
                    directorio['left_wrist'].append(keypoints[0][5])
                    directorio['right_wrist'].append(keypoints[0][6])
                    directorio['left_hip'].append(keypoints[0][7])
                    directorio['right_hip'].append(keypoints[0][8])
                    directorio['left_knee'].append(keypoints[0][9])
                    directorio['right_knee'].append(keypoints[0][10])
                    directorio['left_ankle'].append(keypoints[0][11])
                    directorio['right_ankle'].append(keypoints[0][12])
                    directorio['mid_shoulder'].append(keypoints[0][13])
                    directorio['mid_hip'].append(keypoints[0][14])
                    directorio['nose_mid_shoulder'].append(angles[0][0])
                    directorio['mid_shoulder_mid_hip'].append(angles[0][1])
                    directorio['left_shoulder_left_elbow'].append(angles[0][2])
                    directorio['left_elbow_left_wrist'].append(angles[0][3])
                    directorio['right_shoulder_right_elbow'].append(angles[0][4])
                    directorio['right_elbow_right_wrist'].append(angles[0][5])
                    directorio['left_hip_left_knee'].append(angles[0][6])
                    directorio['left_knee_left_ankle'].append(angles[0][7])
                    directorio['right_hip_right_knee'].append(angles[0][8])
                    directorio['right_knee_right_ankle'].append(angles[0][9])
                    directorio['mid_shoulder_angle'].append(angles[0][10])
                    directorio['left_shoulder_angle'].append(angles[0][11])
                    directorio['left_elbow_angle'].append(angles[0][12])
                    directorio['right_shoulder_angle'].append(angles[0][13])
                    directorio['right_elbow_angle'].append(angles[0][14])
                    directorio['left_hip_angle'].append(angles[0][15])
                    directorio['left_knee_angle'].append(angles[0][16])
                    directorio['right_hip_angle'].append(angles[0][17])
                    directorio['right_knee_angle'].append(angles[0][18])
                    directorio['Etiqueta'].append(etiqueta)
            except Exception as e:
                print(e)
                continue


print(len(directorio['frame']))
print(len(directorio['mascara']))
print(len(directorio['nose']))
print(len(directorio['left_shoulder']))
print(len(directorio['right_shoulder']))
print(len(directorio['left_elbow']))
print(len(directorio['right_elbow']))
print(len(directorio['left_wrist']))
print(len(directorio['right_wrist']))
print(len(directorio['left_hip']))
print(len(directorio['right_hip']))
print(len(directorio['left_knee']))
print(len(directorio['right_knee']))
print(len(directorio['left_ankle']))
print(len(directorio['right_ankle']))
print(len(directorio['mid_shoulder']))
print(len(directorio['mid_hip']))
print(len(directorio['nose_mid_shoulder']))
print(len(directorio['mid_shoulder_mid_hip']))
print(len(directorio['left_shoulder_left_elbow']))
print(len(directorio['left_elbow_left_wrist']))
print(len(directorio['right_shoulder_right_elbow']))
print(len(directorio['right_elbow_right_wrist']))
print(len(directorio['left_hip_left_knee']))
print(len(directorio['left_knee_left_ankle']))
print(len(directorio['right_hip_right_knee']))
print(len(directorio['right_knee_right_ankle']))
print(len(directorio['Etiqueta']))

df_mascara = pd.DataFrame(directorio)
df_mascara.to_csv('df_mascara.csv',index=False)
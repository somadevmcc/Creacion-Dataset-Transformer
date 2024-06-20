import face_recognition
import os
from PIL import Image as ImagePIL
import numpy as np
import pandas as pd
import cv2
from cvzone.PoseModule import PoseDetector
from database import Conexiones as con
import gc
from PIL import Image

def getAngles(etiqueta):
    cur, connectio = con.conexion()
    cur.execute("select nose_mid_shoulder,mid_shoulder_mid_hip,left_shoulder_left_elbow,left_elbow_left_wrist,right_shoulder_right_elbow,right_elbow_right_wrist,left_hip_left_knee,left_knee_left_ankle,right_hip_right_knee,right_knee_right_ankle,mid_shoulder_angle,left_shoulder_angle,left_elbow_angle,right_shoulder_angle,right_elbow_angle,left_hip_angle,left_knee_angle,right_hip_angle,right_knee_angle from fun_generar_reporte_angles('"+etiqueta+"') ")
    angles = cur.fetchall()
    return angles


def getKeypoints(etiqueta):
    cur, connectio = con.conexion()
    cur.execute("select nose,left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle,mid_shoulder,mid_hip from fun_generar_reporte_keypoints('"+etiqueta+"')")
    keypoints = cur.fetchall()
    return keypoints
    


frames = "frames"
output_dir = "output"
directorio = {'etiqueta': [],'frame': [],'mascara':[], 'nose':[],'left_shoulder':[],'right_shoulder':[],'left_elbow':[],'right_elbow':[],
            'left_wrist':[],'right_wrist':[],'left_hip':[],'right_hip':[],'left_knee':[],'right_knee':[],'left_ankle':[],
            'right_ankle':[],'mid_shoulder':[],'mid_hip':[],
            'nose_mid_shoulder':[],'mid_shoulder_mid_hip':[],'left_shoulder_left_elbow':[],'left_elbow_left_wrist':[],
            'right_shoulder_right_elbow':[],'right_elbow_right_wrist':[],'left_hip_left_knee':[],'left_knee_left_ankle':[],
            'right_hip_right_knee':[],'right_knee_right_ankle':[],'mid_shoulder_angle':[],'left_shoulder_angle':[],
            'left_elbow_angle':[],'right_shoulder_angle':[],'right_elbow_angle':[],'left_hip_angle':[],'left_knee_angle':[],
            'right_hip_angle':[],'right_knee_angle':[]
            }



for carpetas in os.listdir(frames):
    try:
        
        ruta = os.path.join(frames, carpetas)
        
        etiqueta = carpetas.split("_")[0]
        print(carpetas)
        keypoints = getKeypoints(carpetas)
        angles = getAngles(carpetas)
        for i in range(0,63):
            
            frame = i+1
            if frame == 63:
                print("hello")
            print(carpetas + " " + str(frame))
            frame_file = Image.open(os.path.join(ruta, os.listdir(ruta)[i]))
            
            directorio['frame'].append(frame)
            directorio['mascara'].append(np.array(frame_file))
            
            directorio['nose'].append(keypoints[i][0])
            directorio['left_shoulder'].append(keypoints[i][1])
            directorio['right_shoulder'].append(keypoints[i][2])
            directorio['left_elbow'].append(keypoints[i][3])
            directorio['right_elbow'].append(keypoints[i][4])
            directorio['left_wrist'].append(keypoints[i][5])
            directorio['right_wrist'].append(keypoints[i][6])
            directorio['left_hip'].append(keypoints[i][7])
            directorio['right_hip'].append(keypoints[i][8])
            directorio['left_knee'].append(keypoints[i][9])
            directorio['right_knee'].append(keypoints[i][10])
            directorio['left_ankle'].append(keypoints[i][11])
            directorio['right_ankle'].append(keypoints[i][12])
            directorio['mid_shoulder'].append(keypoints[i][13])
            directorio['mid_hip'].append(keypoints[i][14])
            directorio['nose_mid_shoulder'].append(angles[i][0])
            directorio['mid_shoulder_mid_hip'].append(angles[i][1])
            directorio['left_shoulder_left_elbow'].append(angles[i][2])
            directorio['left_elbow_left_wrist'].append(angles[i][3])
            directorio['right_shoulder_right_elbow'].append(angles[i][4])
            directorio['right_elbow_right_wrist'].append(angles[i][5])
            directorio['left_hip_left_knee'].append(angles[i][6])
            directorio['left_knee_left_ankle'].append(angles[i][7])
            directorio['right_hip_right_knee'].append(angles[i][8])
            directorio['right_knee_right_ankle'].append(angles[i][9])
            directorio['mid_shoulder_angle'].append(angles[i][10])
            directorio['left_shoulder_angle'].append(angles[i][11])
            directorio['left_elbow_angle'].append(angles[i][12])
            directorio['right_shoulder_angle'].append(angles[i][13])
            directorio['right_elbow_angle'].append(angles[i][14])
            directorio['left_hip_angle'].append(angles[i][15])
            directorio['left_knee_angle'].append(angles[i][16])
            directorio['right_hip_angle'].append(angles[i][17])
            directorio['right_knee_angle'].append(angles[i][18])
            directorio['etiqueta'].append(etiqueta)


    except Exception as e:
        print(e)
        continue




if not os.path.exists("output/"+etiqueta):
    os.makedirs("output/"+etiqueta)

df = pd.DataFrame(directorio)
df.to_csv("output/"+etiqueta+".csv",index=False)
directorio.clear()
gc.collect()

directorio = {'etiqueta': [],'frame': [],'mascara':[], 'nose':[],'left_shoulder':[],'right_shoulder':[],'left_elbow':[],'right_elbow':[],
    'left_wrist':[],'right_wrist':[],'left_hip':[],'right_hip':[],'left_knee':[],'right_knee':[],'left_ankle':[],
    'right_ankle':[],'mid_shoulder':[],'mid_hip':[],
    'nose_mid_shoulder':[],'mid_shoulder_mid_hip':[],'left_shoulder_left_elbow':[],'left_elbow_left_wrist':[],
    'right_shoulder_right_elbow':[],'right_elbow_right_wrist':[],'left_hip_left_knee':[],'left_knee_left_ankle':[],
    'right_hip_right_knee':[],'right_knee_right_ankle':[],'mid_shoulder_angle':[],'left_shoulder_angle':[],
    'left_elbow_angle':[],'right_shoulder_angle':[],'right_elbow_angle':[],'left_hip_angle':[],'left_knee_angle':[],
    'right_hip_angle':[],'right_knee_angle':[]}


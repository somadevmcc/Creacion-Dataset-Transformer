import os
from PIL import Image as ImagePIL
import numpy as np
import pandas as pd
import cv2
from database import Conexiones as con

def get_keypoints(frame,etiqueta):
    response = {}
    cur, connection = con.conexion()
    cur.execute("SELECT * FROM fun_generar_reporte_keypoints(%s) WHERE CAST(frame as integer)  = %s", (etiqueta,frame))
    rows = cur.fetchall()
    con.close_connection(connection)
    
    return rows[0][0]

def get_angles(frame,etiqueta):
    cur, connection = con.conexion()
    cur.execute("SELECT * FROM fun_generar_reporte_angles(%s) WHERE CAST(frame as integer)  = %s", (etiqueta,frame))
    rows = cur.fetchall()
    return rows[0][0]

videos = "videos"
output_dir = "output"
features = []
labels = []
datos_imagen_recortada = []
directorio = {'frame':[],'mascara': [], 'keypoints': [], 'angles': [], 'Etiqueta': []}
for carpetas in os.listdir(videos):
    ruta = os.path.join(videos, carpetas)
    for video_file in os.listdir(ruta):
        try:
            print(carpetas +" "+video_file)
            video_path = os.path.join(ruta, video_file)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            
            video = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to numpy array
                frame_array = np.array(frame)
                # Append frame to video list
                frame_count += 1
                directorio['frame'].append(frame_count)
                directorio['mascara'].append(frame_array)
                directorio['keypoints'].append(get_keypoints(frame_count,carpetas))
                directorio['angles'].append(get_angles(frame_count,carpetas))
                #break

            cap.release()
        except Exception as e:
            print(e)
            continue
    

print(directorio)
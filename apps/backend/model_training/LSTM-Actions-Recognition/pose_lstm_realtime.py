import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import tensorflow as tf
import threading
import h5py
import json
from tensorflow.keras.models import model_from_json

# Global MediaPipe initialization
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def load_model(model_path):
    try:
        with h5py.File(model_path, 'r') as f:
            model_config = f.attrs.get('model_config')
            model_config = json.loads(model_config)  
            for layer in model_config['config']['layers']:
                if 'time_major' in layer['config']:
                    del layer['config']['time_major']
            
            custom_objects = {
                'Orthogonal': tf.keras.initializers.Orthogonal
            }
            
            model_json = json.dumps(model_config)
            model = model_from_json(model_json, custom_objects=custom_objects)
            
            weights_group = f['model_weights']
            for layer in model.layers:
                layer_name = layer.name
                if layer_name in weights_group:
                    weight_names = weights_group[layer_name].attrs['weight_names']
                    layer_weights = [weights_group[layer_name][weight_name][()] for weight_name in weight_names]
                    layer.set_weights(layer_weights)
            return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def make_landmark_timestep(results):
    c_lm = []
    for lm in results.pose_landmarks.landmark:
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(results, frame):
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    for lm in results.pose_landmarks.landmark:
        h, w, _ = frame.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), cv2.FILLED)
    return frame

def draw_class_on_image(label, img, neutral_label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0) if label == neutral_label else (0, 0, 255)
    thickness = 2
    lineType = 2
    cv2.putText(img, str(label),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

def detect(model, lm_list, label_ref):
    try:
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        result = model.predict(lm_list, verbose=0)
        label_ref['label'] = "violent" if result[0][0] > 0.5 else "neutral"
    except Exception as e:
        print(f"Error in detection: {str(e)}")
        label_ref['label'] = "error"

def main():
    # Initialize variables
    label_ref = {'label': "neutral"}  # Using dict to make it mutable in threads
    neutral_label = "neutral"
    lm_list = []
    i = 0
    warm_up_frames = 60
    
    # Initialize video capture
    print("Initializing video capture...")
    cap = cv2.VideoCapture(0)  # Changed to 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return

    # Initialize pose detection
    print("Initializing pose detection...")
    pose = mp_pose.Pose()

    # Load model
    print("Loading model...")
    model = load_model("lstm-model.h5")
    if model is None:
        print("Error: Could not load model")
        return

    print("Starting main loop...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)
            i += 1

            if i > warm_up_frames:
                if results.pose_landmarks:
                    lm = make_landmark_timestep(results)
                    lm_list.append(lm)
                    
                    if len(lm_list) == 20:
                        t1 = threading.Thread(target=detect, args=(model, lm_list.copy(), label_ref))
                        t1.start()
                        lm_list = []

                    # Draw bounding box
                    x_coordinate = []
                    y_coordinate = []
                    for lm in results.pose_landmarks.landmark:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        x_coordinate.append(cx)
                        y_coordinate.append(cy)
                    
                    if x_coordinate and y_coordinate:
                        cv2.rectangle(frame,
                                    (min(x_coordinate), max(y_coordinate)),
                                    (max(x_coordinate), min(y_coordinate) - 25),
                                    (0, 255, 0),
                                    1)
                    
                    frame = draw_landmark_on_image(results, frame)
                
                frame = draw_class_on_image(label_ref['label'], frame, neutral_label)
                cv2.imshow("Pose Detection", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except Exception as e:
        print(f"Runtime error: {str(e)}")
    
    finally:
        print("Cleaning up...")
        if len(lm_list) > 0:
            df = pd.DataFrame(lm_list)
            df.to_csv(f"{label_ref['label']}.txt", index=False)
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
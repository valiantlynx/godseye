import numpy as np
from keras.models import load_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

class ViolenceDetector:
    def __init__(self, model_path, mean=None, std=None):
        self.no_of_timesteps = 20
        self.keypoint_labels = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        logging.info(f"Loading model from {model_path}")
        self.model = load_model(model_path)
        self.mean = mean
        self.std = std
        
    def process_keypoints(self, json_data):
        """
        Process keypoints for the given frames, assuming `json_data` contains exactly `self.no_of_timesteps` frames.
        """
        sequences = []
        for i, frame in enumerate(json_data):

            if frame.get("detections"):
                # Take the first detected person in the frame
                # TODO fix this to take  everybody
                person = frame["detections"][0]
                person_keypoints = []
                keypoints_dict = {kp["label"]: kp["coordinates"] for kp in person["keypoints"]}
                for label in self.keypoint_labels:
                    if label in keypoints_dict:
                        coords = keypoints_dict[label]
                        person_keypoints.extend([coords["x"], coords["y"]])
                    else:
                        # Add zeros if keypoint label is missing
                        person_keypoints.extend([0.0, 0.0])
            else:
                # If no detections, append zeros for all keypoints
                person_keypoints = [0.0, 0.0] * len(self.keypoint_labels)

            sequences.append(person_keypoints)

        # Convert to numpy array and add batch dimension
        sequences = np.array([sequences], dtype=np.float32)
        return sequences
    
    def normalize_sequences(self, sequences):
        if self.mean is None or self.std is None:
            logging.warning("No normalization parameters provided, using raw values")
            return sequences
            
        return (sequences - self.mean) / self.std
    
    def predict_video(self, sequences):
        """Make a single prediction for a video based on all its sequences"""
        if sequences is None:
            return None, None
        
        sequences = self.normalize_sequences(sequences)
        pred_probs = self.model.predict(sequences, verbose=0)
        
        # Average probabilities across all sequences for final prediction
        avg_prob = np.mean(pred_probs)
        final_prediction = 1 if avg_prob >= 0.8 else 0
        
        return avg_prob, final_prediction

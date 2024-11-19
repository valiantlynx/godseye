import os
import json
import numpy as np
from keras.models import load_model
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

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
        """Process keypoints using sliding window approach matching training"""
        if len(json_data) < self.no_of_timesteps:
            return None
            
        sequences = []
        # Use sliding window approach
        for i in range(self.no_of_timesteps, len(json_data)):
            sequence = []
            frames = json_data[i - self.no_of_timesteps:i]
            
            for frame in frames:
                if frame["detections"]:
                    person = frame["detections"][0]
                    person_keypoints = []
                    
                    keypoints_dict = {kp['label']: kp['coordinates'] for kp in person['keypoints']}
                    
                    for label in self.keypoint_labels:
                        if label in keypoints_dict:
                            coords = keypoints_dict[label]
                            person_keypoints.extend([coords['x'], coords['y']])
                        else:
                            person_keypoints.extend([0.0, 0.0])
                else:
                    person_keypoints = [0.0, 0.0] * len(self.keypoint_labels)
                    
                sequence.append(person_keypoints)
            
            sequences.append(sequence)
            
        return np.array(sequences, dtype=np.float32)
    
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
        final_prediction = 1 if avg_prob >= 0.5 else 0
        
        return avg_prob, final_prediction

def evaluate_validation_dataset(model_path, val_dir, mean=None, std=None):
    detector = ViolenceDetector(model_path, mean, std)
    
    video_predictions = []
    video_labels = []
    processed_files = 0
    failed_files = 0
    
    # Process Fight videos
    fight_dir = os.path.join(val_dir, "Fight")
    nonfight_dir = os.path.join(val_dir, "NonFight")
    
    logging.info("Processing validation dataset...")
    
    # Helper function to process directories
    def process_directory(directory, label):
        nonlocal processed_files, failed_files, video_predictions, video_labels
        
        for video_folder in tqdm(os.listdir(directory), desc=f"Processing {os.path.basename(directory)}"):
            video_folder_path = os.path.join(directory, video_folder)
            if os.path.isdir(video_folder_path):
                json_path = os.path.join(video_folder_path, f"{video_folder}.json")
                
                if os.path.isfile(json_path):
                    try:
                        with open(json_path) as f:
                            json_data = json.load(f)
                        
                        sequences = detector.process_keypoints(json_data)
                        if sequences is not None:
                            avg_prob, final_prediction = detector.predict_video(sequences)
                            
                            # Store video-level prediction
                            video_predictions.append(final_prediction)
                            video_labels.append(label)
                            
                            processed_files += 1
                            
                            # Log prediction details
                            logging.info(f"Video: {video_folder}, True Label: {'Fight' if label == 1 else 'NonFight'}, "
                                       f"Predicted: {'Fight' if final_prediction == 1 else 'NonFight'}, "
                                       f"Confidence: {avg_prob:.2%}")
                        else:
                            failed_files += 1
                            logging.warning(f"Skipped {json_path}: insufficient frames")
                            
                    except Exception as e:
                        failed_files += 1
                        logging.error(f"Error processing {json_path}: {e}")
    
    # Process both Fight and NonFight directories
    process_directory(fight_dir, 1)
    process_directory(nonfight_dir, 0)
    
    # Calculate metrics
    if video_predictions:
        accuracy = accuracy_score(video_labels, video_predictions)
        precision = precision_score(video_labels, video_predictions)
        recall = recall_score(video_labels, video_predictions)
        f1 = f1_score(video_labels, video_predictions)
        conf_matrix = confusion_matrix(video_labels, video_predictions)
        
        # Print results
        print("/nValidation Results:")
        print("=" * 50)
        print(f"Total videos processed: {processed_files}")
        print(f"Failed videos: {failed_files}")
        print("/nMetrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("/nConfusion Matrix:")
        print("                 Predicted")
        print("                 Non-Fight  Fight")
        print(f"Actual Non-Fight    {conf_matrix[0][0]}        {conf_matrix[0][1]}")
        print(f"      Fight         {conf_matrix[1][0]}        {conf_matrix[1][1]}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'processed_files': processed_files,
            'failed_files': failed_files
        }
    else:
        logging.error("No predictions were made. Check the dataset and paths.")
        return None

def main():
    # Update these paths to match your system
    model_path = "violence_detector_best.keras"
    val_dir = "C:/Users/gorme/projects/godseye/apps/backend/dataset_processing/archive/keypoints-rwf-2000/val"
    
    # Load normalization parameters (if you saved them during training)
    try:
        mean = np.load('mean.npy')
        std = np.load('std.npy')
    except FileNotFoundError:
        logging.warning("Normalization parameters not found. Proceeding without normalization.")
        mean = None
        std = None
    
    # Run evaluation
    results = evaluate_validation_dataset(model_path, val_dir, mean, std)

if __name__ == "__main__":
    main()
# HVAC Predictive Maintenance using Acoustic Anomaly Detection
# Compatible with Python 3.13
# Author: Nermine.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import os
import warnings
import soundfile as sf  
warnings.filterwarnings('ignore')

class AcousticHVACMonitor:
    """
    A class for predicting HVAC system failures using acoustic anomaly detection.
    This system analyzes audio recordings to identify abnormal sounds indicating potential failures.
    """
    
    def __init__(self, data_path=None, model_path=None):
        """Initialize the HVAC monitor with data and model paths."""
        self.data_path = data_path
        self.model_path = model_path
        self.model = None
        self.features = None
        self.scaler = None
        self.labels = None  # Added labels attribute to store them
        
    def extract_audio_features(self, audio_file):
        """Extract relevant features from audio files."""
        # Load audio file
        y, sr = librosa.load(audio_file, sr=None)
        
        # Extract features
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0])
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)[0])
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)[0])
        
        # Rhythmic features
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y)[0])
        
        # MFCC (Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        
        # RMS energy
        rms = np.mean(librosa.feature.rms(y=y)[0])
        
        # Combine all features
        features = np.hstack([
            spectral_centroid, 
            spectral_bandwidth, 
            spectral_rolloff, 
            zero_crossing_rate, 
            mfcc_means,
            rms
        ])
        
        return features
    
    def load_dataset(self):
        """Load audio files and extract features."""
        normal_files = [f for f in os.listdir(f"{self.data_path}/normal") if f.endswith('.wav')]
        abnormal_files = [f for f in os.listdir(f"{self.data_path}/abnormal") if f.endswith('.wav')]
        
        features = []
        labels = []
        
        # Process normal files
        for file in normal_files:
            feature_vector = self.extract_audio_features(f"{self.data_path}/normal/{file}")
            features.append(feature_vector)
            labels.append(0)  # 0 for normal
            
        # Process abnormal files
        for file in abnormal_files:
            feature_vector = self.extract_audio_features(f"{self.data_path}/abnormal/{file}")
            features.append(feature_vector)
            labels.append(1)  # 1 for abnormal
            
        return np.array(features), np.array(labels)
    
    def preprocess_data(self, features):
        """Scale features."""
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        return scaled_features
    
    def train_model(self, contamination=0.1):
        """Train an anomaly detection model."""
        features, labels = self.load_dataset()
        self.features = features
        self.labels = labels  # Store labels when loading the dataset
        
        # Preprocess data
        scaled_features = self.preprocess_data(features)
        
        # Create and train pipeline with PCA and Isolation Forest
        self.model = Pipeline([
            ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
            ('isolation_forest', IsolationForest(contamination=contamination, random_state=42))
        ])
        
        self.model.fit(scaled_features)
        
        # Save model
        if self.model_path:
            joblib.dump(self.model, f"{self.model_path}/hvac_acoustic_model.pkl")
            joblib.dump(self.scaler, f"{self.model_path}/hvac_scaler.pkl")
        
        return self.model
    
    def load_model(self):
        """Load a pre-trained model."""
        self.model = joblib.load(f"{self.model_path}/hvac_acoustic_model.pkl")
        self.scaler = joblib.load(f"{self.model_path}/hvac_scaler.pkl")
        return self.model
    
    def evaluate_model(self, test_features=None, test_labels=None):
        """Evaluate model performance."""
        if test_features is None or test_labels is None:
            features, labels = self.load_dataset()
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.3, random_state=42
            )
            test_features = X_test
            test_labels = y_test
        
        # Preprocess test data
        scaled_test = self.preprocess_data(test_features)
        
        # Make predictions
        # Convert from Isolation Forest format (-1 for anomalies, 1 for inliers) to our format (1 for anomalies, 0 for normal)
        predictions = self.model.predict(scaled_test)
        predictions = np.where(predictions == -1, 1, 0)
        
        # Evaluate
        print("Classification Report:")
        print(classification_report(test_labels, predictions))
        
        # Confusion Matrix
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return predictions
    
    def predict_single_sample(self, audio_file):
        """Predict if a single audio sample is anomalous."""
        if self.model is None:
            self.load_model()
            
        # Extract features
        features = self.extract_audio_features(audio_file)
        features = features.reshape(1, -1)
        
        # Preprocess
        scaled_features = self.preprocess_data(features)
        
        # Predict
        prediction = self.model.predict(scaled_features)
        
        # Convert from Isolation Forest format to our format
        result = 'Abnormal' if prediction[0] == -1 else 'Normal'
        
        return result
    
    def visualize_features(self):
        """Visualize feature distribution using PCA."""
        if self.features is None or self.labels is None:
            self.features, self.labels = self.load_dataset()
            
        # Preprocess data
        scaled_features = self.preprocess_data(self.features)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)
        
        # Create DataFrame for visualization
        df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        df['label'] = self.labels  # Use self.labels instead of undefined labels variable
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PC1', y='PC2', hue='label', data=df, palette={0: 'blue', 1: 'red'})
        plt.title('PCA of Audio Features')
        plt.legend(['Normal', 'Abnormal'])
        plt.savefig('feature_visualization.png')
        plt.close()
        
        return df

# Example usage
def generate_synthetic_dataset(output_path):
    """Generate synthetic HVAC audio data for demonstration."""
    os.makedirs(f"{output_path}/normal", exist_ok=True)
    os.makedirs(f"{output_path}/abnormal", exist_ok=True)
    
    # Generate synthetic normal data
    print("Generating synthetic normal HVAC data...")
    for i in range(50):
        # Create white noise with specific characteristics for normal operation
        duration = 3  # seconds
        sr = 22050  # sample rate
        
        # Normal HVAC sounds (smoother frequency profile)
        y = np.random.normal(0, 0.1, size=int(duration * sr))
        
        # Add some periodic components to simulate fan/motor
        t = np.linspace(0, duration, int(duration * sr))
        # Motor hum at 60Hz 
        y += 0.1 * np.sin(2 * np.pi * 60 * t)
        # Low fan noise (smoother)
        y += 0.05 * np.sin(2 * np.pi * 120 * t)
        
        # Save as WAV file using soundfile instead of librosa.output
        sf.write(f"{output_path}/normal/normal_{i}.wav", y, sr)
    
    # Generate synthetic abnormal data
    print("Generating synthetic abnormal HVAC data...")
    for i in range(20):
        # Create white noise with different characteristics for abnormal operation
        duration = 3  # seconds
        sr = 22050  # sample rate
        
        # Base noise
        y = np.random.normal(0, 0.1, size=int(duration * sr))
        
        # Add base motor sound
        t = np.linspace(0, duration, int(duration * sr))
        y += 0.1 * np.sin(2 * np.pi * 60 * t)
        
        # Now add anomalies
        anomaly_type = i % 4
        
        if anomaly_type == 0:
            # Loose bearing - high frequency components
            y += 0.3 * np.sin(2 * np.pi * 1800 * t)
        elif anomaly_type == 1:
            # Refrigerant leak - hissing sound (white noise burst)
            burst_start = int(duration * sr * 0.4)
            burst_end = int(duration * sr * 0.6)
            y[burst_start:burst_end] += np.random.normal(0, 0.4, size=(burst_end-burst_start))
        elif anomaly_type == 2:
            # Belt slipping - irregular rhythm
            for j in range(5):
                start = int(duration * sr * (j/5 + np.random.uniform(0, 0.1)))
                end = min(start + int(sr * 0.2), int(duration * sr))
                y[start:end] += 0.4 * np.sin(2 * np.pi * 300 * t[0:(end-start)])
        else:
            # Compressor knock - low frequency thumps
            for j in range(8):
                start = int(duration * sr * (j/8))
                y[start:start+int(sr*0.1)] += 0.5 * np.sin(2 * np.pi * 40 * t[0:int(sr*0.1)])
        
        # Save as WAV file using soundfile instead of librosa.output
        sf.write(f"{output_path}/abnormal/abnormal_{i}.wav", y, sr)
    
    print(f"Dataset created at {output_path}")

def main():
    """Main function to demonstrate the HVAC monitor."""
    # Set paths
    data_path = "data/hvac_sounds"
    model_path = "models"
    
    # Create directories
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    # Generate synthetic data for demonstration
    generate_synthetic_dataset(data_path)
    
    # Create and train model
    hvac_monitor = AcousticHVACMonitor(data_path=data_path, model_path=model_path)
    hvac_monitor.train_model(contamination=0.2)  # 20% of data is expected to be anomalous 
    
    # Evaluate model
    hvac_monitor.evaluate_model()
    
    # Visualize features
    hvac_monitor.visualize_features()
    
    # Example prediction
    print("\nExample predictions:")
    normal_file = f"{data_path}/normal/normal_0.wav"
    abnormal_file = f"{data_path}/abnormal/abnormal_0.wav"
    
    print(f"Normal file prediction: {hvac_monitor.predict_single_sample(normal_file)}")
    print(f"Abnormal file prediction: {hvac_monitor.predict_single_sample(abnormal_file)}")
    
    print("\n Check the generated files for results.")

if __name__ == "__main__":
    main()
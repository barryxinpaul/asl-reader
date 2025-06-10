import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional
import csv
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

class ASLRecognizer:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Training data storage
        self.training_data_file = 'asl_training_data.csv'
        self.training_data = []
        self.knn_classifier = None
        self.scaler = StandardScaler()
        
        # Load existing training data if available
        self._load_training_data()
        
        # Define reference vectors for all ASL letters
        self.reference_vectors = {
            'A': self._create_reference_vector([
                # Thumb out, others closed
                [0.5, 0.2, 0.0],  # Thumb tip
                [0.3, 0.8, 0.0],  # Index tip
                [0.3, 0.9, 0.0],  # Middle tip
                [0.3, 0.9, 0.0],  # Ring tip
                [0.3, 0.9, 0.0],  # Pinky tip
            ]),
            'B': self._create_reference_vector([
                # All fingers up, thumb out
                [0.5, 0.2, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.1, 0.0],  # Middle tip
                [0.3, 0.1, 0.0],  # Ring tip
                [0.3, 0.1, 0.0],  # Pinky tip
            ]),
            'C': self._create_reference_vector([
                # Curved hand shape
                [0.4, 0.4, 0.0],  # Thumb tip
                [0.3, 0.4, 0.0],  # Index tip
                [0.3, 0.4, 0.0],  # Middle tip
                [0.3, 0.4, 0.0],  # Ring tip
                [0.3, 0.4, 0.0],  # Pinky tip
            ]),
            'D': self._create_reference_vector([
                # Index up, thumb touching middle
                [0.3, 0.5, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.5, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'E': self._create_reference_vector([
                # All fingers closed
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.9, 0.0],  # Index tip
                [0.3, 0.9, 0.0],  # Middle tip
                [0.3, 0.9, 0.0],  # Ring tip
                [0.3, 0.9, 0.0],  # Pinky tip
            ]),
            'F': self._create_reference_vector([
                # Index and middle up, thumb touching
                [0.3, 0.5, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.1, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'G': self._create_reference_vector([
                # Index pointing, thumb out
                [0.5, 0.2, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'H': self._create_reference_vector([
                # Index and middle up, others closed
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.1, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'I': self._create_reference_vector([
                # Pinky up, others closed
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.8, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.1, 0.0],  # Pinky tip
            ]),
            'K': self._create_reference_vector([
                # Index and middle up in V shape
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.2, 0.1, 0.0],  # Index tip
                [0.4, 0.1, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'L': self._create_reference_vector([
                # Thumb and index in L shape
                [0.5, 0.2, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'M': self._create_reference_vector([
                # Three fingers down
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.5, 0.0],  # Index tip
                [0.3, 0.5, 0.0],  # Middle tip
                [0.3, 0.5, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'N': self._create_reference_vector([
                # Two fingers down
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.5, 0.0],  # Index tip
                [0.3, 0.5, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'O': self._create_reference_vector([
                # All fingers curved
                [0.4, 0.4, 0.0],  # Thumb tip
                [0.3, 0.4, 0.0],  # Index tip
                [0.3, 0.4, 0.0],  # Middle tip
                [0.3, 0.4, 0.0],  # Ring tip
                [0.3, 0.4, 0.0],  # Pinky tip
            ]),
            'P': self._create_reference_vector([
                # Thumb and index pointing down
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.8, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'Q': self._create_reference_vector([
                # Thumb and index pointing down
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.8, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'R': self._create_reference_vector([
                # Index and middle crossed
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.2, 0.1, 0.0],  # Index tip
                [0.4, 0.1, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'S': self._create_reference_vector([
                # Fist
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.8, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'T': self._create_reference_vector([
                # Index up, thumb touching
                [0.3, 0.5, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'U': self._create_reference_vector([
                # Index and middle up together
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.1, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'V': self._create_reference_vector([
                # Index and middle in V shape
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.2, 0.1, 0.0],  # Index tip
                [0.4, 0.1, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'W': self._create_reference_vector([
                # Three fingers up
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.2, 0.1, 0.0],  # Index tip
                [0.3, 0.1, 0.0],  # Middle tip
                [0.4, 0.1, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'X': self._create_reference_vector([
                # Index bent
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.5, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'Y': self._create_reference_vector([
                # Thumb and pinky out
                [0.5, 0.2, 0.0],  # Thumb tip
                [0.3, 0.8, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.5, 0.2, 0.0],  # Pinky tip
            ]),
            'Z': self._create_reference_vector([
                # Index drawing Z
                [0.3, 0.8, 0.0],  # Thumb tip
                [0.3, 0.1, 0.0],  # Index tip
                [0.3, 0.8, 0.0],  # Middle tip
                [0.3, 0.8, 0.0],  # Ring tip
                [0.3, 0.8, 0.0],  # Pinky tip
            ]),
            'LOVE': self._create_reference_vector([
                # Thumb, index, pinky out; middle and ring down
                [0.5, 0.2, 0.0],  # Thumb tip (out)
                [0.3, 0.1, 0.0],  # Index tip (out)
                [0.3, 0.8, 0.0],  # Middle tip (down)
                [0.3, 0.8, 0.0],  # Ring tip (down)
                [0.5, 0.2, 0.0],  # Pinky tip (out)
            ])
        }

    def _load_training_data(self):
        """Load training data from CSV file if it exists"""
        if os.path.exists(self.training_data_file):
            with open(self.training_data_file, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:  # Skip empty rows
                        label = row[0]
                        features = [float(x) for x in row[1:]]
                        self.training_data.append((label, features))
            self._train_knn()

    def _save_training_data(self):
        """Save training data to CSV file"""
        with open(self.training_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for label, features in self.training_data:
                writer.writerow([label] + features)

    def _train_knn(self):
        """Train KNN classifier with current training data"""
        if len(self.training_data) > 0:
            X = np.array([features for _, features in self.training_data])
            y = np.array([label for label, _ in self.training_data])
            
            # Scale the features
            X = self.scaler.fit_transform(X)
            
            # Calculate appropriate number of neighbors
            n_neighbors = min(3, len(self.training_data))
            
            # Train KNN classifier
            self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
            self.knn_classifier.fit(X, y)

    def add_training_example(self, landmarks: List[List[float]], label: str):
        """Add a new training example"""
        vector = self._get_landmark_vector(landmarks)
        self.training_data.append((label, vector.tolist()))
        self._save_training_data()
        self._train_knn()
        print(f"Added training example for letter {label} (Total examples: {len(self.training_data)})")

    def _create_reference_vector(self, landmarks: List[List[float]]) -> np.ndarray:
        """Create a normalized reference vector from landmark positions"""
        vector = np.array(landmarks).flatten()
        return vector / np.linalg.norm(vector)

    def _get_landmark_vector(self, landmarks: List[List[float]]) -> np.ndarray:
        """Extract and normalize landmark vector from hand landmarks"""
        key_points = [
            landmarks[4],   # Thumb tip
            landmarks[8],   # Index tip
            landmarks[12],  # Middle tip
            landmarks[16],  # Ring tip
            landmarks[20],  # Pinky tip
        ]
        vector = np.array(key_points).flatten()
        return vector / np.linalg.norm(vector)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def detect_letter(self, landmarks: List[List[float]], threshold: float = 0.85) -> Optional[str]:
        """Detect ASL letter using either KNN or cosine similarity"""
        current_vector = self._get_landmark_vector(landmarks)
        
        # If we have trained data, use KNN
        if self.knn_classifier is not None and len(self.training_data) > 0:
            try:
                # Scale the input vector
                scaled_vector = self.scaler.transform([current_vector])
                # Get prediction
                prediction = self.knn_classifier.predict(scaled_vector)[0]
                # Get probability
                probabilities = self.knn_classifier.predict_proba(scaled_vector)[0]
                max_prob = max(probabilities)
                
                if max_prob > threshold:
                    # If the prediction is 'LOVE', display as 'LOVE'
                    return 'LOVE' if prediction == 'LOVE' else prediction
            except Exception as e:
                print(f"KNN prediction failed: {e}")
                # Fall back to cosine similarity
        
        # Fallback to cosine similarity if KNN is not available or confidence is low
        best_match = None
        best_similarity = threshold

        for letter, ref_vector in self.reference_vectors.items():
            similarity = self._cosine_similarity(current_vector, ref_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = letter

        # If the best match is 'LOVE', display as 'LOVE'
        if best_match == 'LOVE':
            return 'LOVE'
        return best_match

    def process_frame(self, frame: np.ndarray, training_mode: bool = False) -> Tuple[np.ndarray, Optional[str], Optional[List[List[float]]]]:
        """Process a single frame and return the annotated frame, detected letter, and landmarks"""
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        detected_letter = None
        current_landmarks = None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append([landmark.x, landmark.y, landmark.z])

                current_landmarks = landmarks
                detected_letter = self.detect_letter(landmarks)
                if detected_letter:
                    cv2.putText(
                        frame,
                        f"Letter: {detected_letter}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

        # Display training mode status and example count
        if training_mode:
            status_text = f"Training Mode - Press a-z to record (Examples: {len(self.training_data)})"
            cv2.putText(
                frame,
                status_text,
                (50, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        return frame, detected_letter, current_landmarks

def main():
    recognizer = ASLRecognizer()
    cap = cv2.VideoCapture(0)
    training_mode = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break

        # Process frame
        processed_frame, letter, landmarks = recognizer.process_frame(frame, training_mode)

        # Display the frame
        cv2.imshow('ASL Reader', processed_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            training_mode = not training_mode
            print(f"Training mode {'enabled' if training_mode else 'disabled'}")
        elif training_mode and (key in range(ord('a'), ord('z') + 1) or key == ord('1')):
            if landmarks is not None:
                # Use 'LOVE' label for '1' key
                if key == ord('1'):
                    recognizer.add_training_example(landmarks, 'LOVE')
                else:
                    label = chr(key).upper()
                    recognizer.add_training_example(landmarks, label)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
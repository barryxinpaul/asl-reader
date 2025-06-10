# ASL Reader

A real-time American Sign Language (ASL) recognition system using computer vision and machine learning. This project uses OpenCV and MediaPipe for hand tracking, and scikit-learn for gesture classification.

## Features

- Real-time hand tracking and gesture recognition
- Support for ASL alphabet recognition
- Training mode for collecting new gesture data
- Machine learning-based classification using K-Nearest Neighbors

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- scikit-learn
- NumPy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/asl-reader.git
cd asl-reader
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python asl_reader.py
```

### Training Mode
To collect training data for new gestures:
1. Run the script in training mode
2. Hold your hand in the desired ASL gesture
3. Press 't' to capture the gesture
4. Enter the corresponding letter when prompted

### Recognition Mode
- Hold your hand in front of the camera
- The recognized letter will be displayed on screen
- Press 'q' to quit

## Project Structure

- `asl_reader.py`: Main application file
- `asl_training_data.csv`: Training data storage
- `requirements.txt`: Python package dependencies

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
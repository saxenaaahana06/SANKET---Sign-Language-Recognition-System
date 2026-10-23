# SANKET – Sign Language Recognition System

## 1. Project Overview

**SANKET** is an AI-powered Sign Language Recognition System designed to recognize Indian Sign Language (ISL) alphabet and number gestures using a camera.

The project aims to make communication easier between people who use sign language and those who may not understand it. It uses a deep learning model to classify hand gestures and display the corresponding prediction.

## 2. Features

- **Camera-Based Recognition:** Captures hand gestures using a webcam.
- **AI-Powered Prediction:** Uses a trained ResNet-18 deep learning model to classify signs.
- **35-Class Classification:** Recognizes English alphabets (A–Z) and digits (1–9).
- **Real-Time Interaction:** Sends camera frames to the backend for prediction.
- **Prediction API:** Connects the frontend with the trained machine learning model.
- **User-Friendly Interface:** Provides an interface for interacting with the recognition system.
- **Model Checkpointing:** Saves trained model weights for future use without retraining.

## 3. Technologies Used

### Programming Languages
- Python
- JavaScript
- HTML and CSS

### AI and Machine Learning
- PyTorch
- Torchvision
- ResNet-18
- NumPy
- Pillow (PIL)

### Backend
- FastAPI
- Uvicorn

### Frontend
- HTML
- CSS
- JavaScript

### Development Tools
- Jupyter Notebook
- Visual Studio Code
- Ubuntu on WSL
- CUDA-enabled PyTorch
- NVIDIA GeForce RTX 4050 Laptop GPU

## 4. Project Structure

```text
SANKET_Backend/
├── data/
│   └── combined_dataset/
│       ├── 1/
│       ├── 2/
│       ├── ...
│       ├── A/
│       ├── B/
│       └── ...
├── best_resnet18_combined.pth
├── last_resnet18_combined.pth
├── frontend/
│   └── SANKET_frontend/
│       ├── templates/
│       └── static/
└── README.md
```

**Note:** The structure above shows the main project components. Exact filenames and folders may vary depending on the current implementation. The trained model checkpoints and dataset may be kept locally rather than uploaded to GitHub.

## 5. Installation

### Prerequisites

- Python 3.10 or later
- pip
- Git
- A webcam
- NVIDIA GPU with compatible CUDA support (recommended for GPU inference)

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SANKET_Backend
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Ubuntu/Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

Install the required packages:

```bash
pip install torch torchvision fastapi uvicorn numpy pillow python-multipart
```

Install any additional packages required by the project.

**Note:** For GPU acceleration, install a PyTorch build compatible with your NVIDIA GPU and CUDA environment. The exact dependency versions should be documented in a `requirements.txt` file.

## 6. Usage

### Step 1: Start the Backend

Navigate to the backend directory and run the FastAPI application using the appropriate module path.

For example, if the application entry point is `app/main.py`:

```bash
uvicorn app.main:app --reload
```

### Step 2: Open the Frontend

Open the frontend using the project's configured web server or frontend development command.

### Step 3: Start Recognition

1. Open the SANKET web interface.
2. Allow access to the webcam.
3. Show a supported ISL alphabet or number gesture.
4. The frontend sends image frames to the backend prediction API.
5. The trained model processes the image and returns the predicted class.
6. View the prediction on the interface.

**Important:** Update the backend command, frontend command, and API endpoint according to the actual project configuration.

## 7. Results

The current machine learning implementation uses a **ResNet-18** model trained on a combined dataset of Indian Sign Language alphabet and number images.

| Metric | Result |
|---|---|
| Model architecture | ResNet-18 |
| Number of output classes | 35 |
| Recognized classes | A–Z and digits 1–9 |
| Training dataset size | 89,686 images |
| Validation dataset size | 19,219 images |
| Test dataset size | 19,219 images |
| Best validation accuracy | 99.94% |
| Hardware used | NVIDIA GeForce RTX 4050 Laptop GPU |

The best-performing checkpoint is saved as `best_resnet18_combined.pth`.

The reported 99.94% is the best validation accuracy. Test accuracy and real-world webcam recognition performance should be measured and reported separately.

## 8. Future Improvements

- Add recognition for the digit zero and additional sign classes.
- Improve recognition under different lighting conditions and backgrounds.
- Add confidence scores and prediction stabilization.
- Support continuous sign recognition and sentence formation.
- Integrate text-to-speech and speech-to-text functionality.
- Expand the dataset with more signers and diverse hand gestures.
- Improve the user interface and accessibility.
- Optimize inference speed for real-time recognition.
- Deploy the application on a cloud platform.
- Add user practice sessions, feedback, and learning progress tracking.

## 9. Author

**Project:** SANKET – Sign Language Recognition System

**Author:** Aahana

**Project Type:** Academic / College Project

**Domain:** Artificial Intelligence, Machine Learning, Computer Vision, and Accessibility

**GitHub:** Add your GitHub profile or repository link here.

## 10. License

This project is developed for educational and research purposes. Add an appropriate open-source license if you intend to distribute the project publicly.

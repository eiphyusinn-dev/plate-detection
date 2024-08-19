# License Plate Detection and Recognition

This project uses YOLOX for custom training to detect car license plates and recognizes the text in license plates using PaddleOCR and Tesseract OCR. Additionally, the project integrates CI/CD with pytest to ensure that the detected text matches the ground truth in an image.

## Project Overview

### YOLOX Custom Training

- **Objective**: Detect car license plates in images.
- **Dataset**: A dataset of car images with labeled license plates.
- **Model**: YOLOX, customized and trained for accurate license plate detection.

### OCR Integration

- **PaddleOCR**: Used to recognize text within the detected license plates.
- **Tesseract OCR**: Alternative OCR method used to compare the results with PaddleOCR.

## CI/CD Integration with Docker

### Continuous Integration and Deployment

This project uses Docker and GitHub Actions to automate the CI/CD pipeline. The Dockerfile sets up the environment, and every time a new commit is pushed to the repository, GitHub Actions will:

1. **Build the Docker Image**: The Dockerfile installs all necessary dependencies and sets up the environment for running the model and OCR.
2. **Run Tests with pytest**: The CI pipeline automatically runs tests to verify that the OCR results match the expected ground truth.

### How It Works

- **Dockerfile**: Defines the environment for the project.
- **GitHub Actions Workflow**: Triggers the CI/CD pipeline on each commit to the repository.

### Steps
#### Clone the Repository:

''' 
git clone https://github.com/your-username/license-plate-detection.git
cd license-plate-detection ''' 

#### Install Dependencies:

''' pip install -r requirements.txt '''

#### Set Up Docker:
docker build -t license-plate-detection .

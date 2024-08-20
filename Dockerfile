# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    g++ \
    cmake \
    tesseract-ocr 

# Copy the requirements file first
COPY requirements.txt .

# Install Python packages from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code into the container
COPY . .

# Install the package in editable mode
RUN pip install -e .

WORKDIR /app/testing_pipeline
# Set the command to run your tests or application
CMD ["pytest", "test_tesseract.py"]

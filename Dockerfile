# Start from a CUDA base image
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Set app as working dir in the container
WORKDIR /app

# Copy application files from app dir into the container
COPY ./ /app

# Install essentials
RUN apt-get update -y

# Install python
RUN apt-get install python3.10 -y
RUN apt -y install python3-pip

# Update the package lists and install required libraries
RUN apt-get update 
RUN pip install -r requirements.txt



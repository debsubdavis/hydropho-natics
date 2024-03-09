# Use python:slim as the base image
FROM python:slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for OpenCV and other graphical operations
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY docker_requirements.txt /app/requirements.txt
# one_minus/requirements.txt

# Install the required packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container at /app
COPY . /app

# The command to run your application
CMD ["python", "train_model.py"]

# docker build -t test-yolo:latest .

# docker run --name test-yolo-container --mount type=bind,source=$(pwd),target=/app test-yolo:latest

# docker run --name test-yolo-container \
#   --mount type=bind,source="$(pwd)",target=/app \
#   --mount type=bind,source="/Users/debbiesubocz/capstone/minus-iter-2/training_dataset",target=/app/data \
#   test-yolo:latest python train_model.py --data_dir /app/data --epochs 5 --img_size 640 --output /app/output --exp_name my_experiment




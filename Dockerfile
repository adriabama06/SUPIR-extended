# Base image with Python
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel

# Install system requeriments cv2 -- libGL.so.1 and some other requirements
RUN apt-get update -y && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements_static.txt .

# Install required packages
RUN pip install --no-cache-dir -r requirements_static.txt

# Copy the repository files
COPY . .

# Define port and volume
EXPOSE 6688
VOLUME [ "/app/models" ]

ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Default command
ENTRYPOINT [ "bash", "-c" ]
CMD [ "/app/entrypoint.sh" ]

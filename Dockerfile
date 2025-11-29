# Use a stable, official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
WORKDIR $APP_HOME

# Install system dependencies required for 'av' and other potential libraries (like OpenCV)
# This is crucial for a Streamlit app that uses video/webcam
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libglib2.0-0 \
    libavformat-dev \
    libswscale-dev \
    libavcodec-dev \
    libavutil-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
# Assuming the user will clone the repo and this Dockerfile will be in the root
COPY . $APP_HOME

# Install Python dependencies
# The requirements file is located in the 'app' directory
RUN pip install --no-cache-dir -r app/requirements.txt

# Create a start script to handle the Streamlit server command
RUN echo '#!/bin/bash' > start.sh && \
    echo 'streamlit run app/main.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false' >> start.sh && \
    chmod +x start.sh

# Expose the port Streamlit will run on (Render will inject the correct port via $PORT)
EXPOSE 8501

# Run the start script
CMD ["./start.sh"]

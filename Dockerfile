# Use your colleague's EC2 base image with PyTorch and FFMPEG
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.3.0-gpu-py311-cu121-ubuntu20.04-ec2

# Set up working directory for your project
WORKDIR /app

# Install FFMPEG
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy model configuration and weights
COPY files /app/files

# Copy your source code files into the container
COPY src /app/src

# Copy the entry script
COPY entry.sh .
RUN chmod +x entry.sh

# Install additional dependencies from requirements.txt (including numpy)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure that the model weights and configuration are in the correct location
RUN mkdir -p /app/files/sorter-1

# Command to run your application
ENTRYPOINT ["./entry.sh"]

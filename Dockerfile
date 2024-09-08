# Use Amazon Linux or any lightweight Python base image
FROM amazonlinux:2

# Install Python 3, pip, and AWS CLI
RUN yum update -y && \
    yum install -y python3-pip && \
    pip3 install --upgrade pip && \
    pip3 install awscli

# Install any additional Python libraries required by your inference and model scripts
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy model configuration and weights
COPY files/sorter-1 /app/files/sorter-1

# Copy your source code files into the container
COPY src /app/src

# Make the entry script executable
COPY entry.sh /app/entry.sh
RUN chmod +x /app/entry.sh

# Set working directory to /app
WORKDIR /app

# Ensure that the model weights and configuration are in the correct location
RUN mkdir -p /app/files/sorter-1

# Expose any necessary ports (e.g., if running a web server or API)
EXPOSE 8080

# Run the entry script which downloads the input file from S3 and runs inference
ENTRYPOINT ["/app/entry.sh"]

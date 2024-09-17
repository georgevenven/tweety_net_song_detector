#!/bin/bash

# Retrieve environment variables passed from the ECS task
bucket_name="${bucket_name}"
prefix="${prefix}"
filename="${filename}"

# Define source and destination paths
src="s3://${bucket_name}/${filename}"
dest="./${filename}"

# Check if the directory exists, if not, pull the file from S3
if [ ! -d "$prefix" ]; then
    aws s3 cp $src $dest
fi

# Run the Python inference script with the environment variables
python src/inference.py --mode aws --input $filename --output $prefix --separate_json

# # # Upload the results back to S3
aws s3 cp "./${prefix}/activity_detection_tweety" "s3://${bucket_name}/${prefix}/activity_detection_tweety/" --recursive

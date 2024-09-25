import os
import shutil

def move_mp4_files(source_dir, target_dir):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.mp4'):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)
                
                # Move the file
                shutil.move(source_file, target_file)
                print(f"Moved: {source_file} -> {target_file}")

if __name__ == "__main__":
    source_directory = "/media/george-vengrovski/disk2/canary/aws_recordings"
    target_directory = "/media/george-vengrovski/disk2/canary/aws_recordings"
    
    move_mp4_files(source_directory, target_directory)

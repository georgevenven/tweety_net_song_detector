import os
from pydub import AudioSegment
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_audio_tracks_pydub(mp4_file_path, output_dir, segment_length=10):
    """
    Extracts audio segments from an MP4 file and saves them as WAV files using pydub.

    Parameters:
    - mp4_file_path (str): Path to the input MP4 file.
    - output_dir (str): Directory where the audio segments will be saved.
    - segment_length (int): Length of each audio segment in seconds.
    """
    try:
        # Load the audio from the video file
        logging.info(f"Loading audio from video file: {mp4_file_path}")
        audio = AudioSegment.from_file(mp4_file_path, "mp4")
        
        duration = len(audio) / 1000.0  # Duration in seconds
        base_name = os.path.splitext(os.path.basename(mp4_file_path))[0]
        
        logging.info(f"Total audio duration: {duration:.2f} seconds")
        
        # Create the main output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory: {output_dir}")
        
        # Create a subdirectory for the audio segments
        segment_dir = os.path.join(output_dir, base_name)
        os.makedirs(segment_dir, exist_ok=True)
        logging.info(f"Segment directory: {segment_dir}")
        
        # Calculate the number of segments
        num_segments = math.ceil(duration / segment_length)
        logging.info(f"Number of segments to extract: {num_segments}")
        
        for i in range(num_segments):
            start_time = i * segment_length * 1000  # Convert to milliseconds
            end_time = min((i + 1) * segment_length * 1000, len(audio))
            
            # Extract the audio segment
            logging.info(f"Extracting segment {i+1}: {start_time/1000:.2f} to {end_time/1000:.2f} seconds")
            audio_segment = audio[start_time:end_time]
            
            # Verify the duration of the segment
            segment_duration = (end_time - start_time) / 1000.0
            logging.info(f"Segment {i+1} duration: {segment_duration:.2f} seconds")
            
            # Define the output file path
            output_file = os.path.join(segment_dir, f"{base_name}_part_{i+1}.wav")
            
            # Export the audio segment to a WAV file
            logging.info(f"Saving segment {i+1} to: {output_file}")
            audio_segment.export(output_file, format="wav")
            logging.info(f"Saved {output_file} (Duration: {segment_duration:.2f} seconds)")
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        logging.info("Finished processing.")

def process_folder_of_mp4s(input_dir, output_dir, segment_length=10):
    """
    Processes all MP4 files in the specified input directory.

    Parameters:
    - input_dir (str): Directory containing MP4 files.
    - output_dir (str): Directory where the audio segments will be saved.
    - segment_length (int): Length of each audio segment in seconds.
    """
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp4"):
            mp4_file_path = os.path.join(input_dir, file_name)
            extract_audio_tracks_pydub(mp4_file_path, output_dir, segment_length)

if __name__ == "__main__":
    # Define the input directory containing MP4 files and output directory
    input_dir = "/media/george-vengrovski/disk2/canary/aws_recordings"
    output_dir = "/media/george-vengrovski/disk2/canary/aws_wav"
    
    # Verify that the input directory exists
    if not os.path.isdir(input_dir):
        logging.error(f"The input directory does not exist: {input_dir}")
    else:
        # Verify that the output directory is writable
        if not os.access(output_dir, os.W_OK):
            logging.error(f"The output directory is not writable: {output_dir}")
        else:
            # Process all MP4 files in the input directory
            process_folder_of_mp4s(input_dir, output_dir)

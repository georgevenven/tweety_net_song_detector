import os
import torch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from pathlib import Path
import csv
from tqdm import tqdm
from spectogram_generator import WavtoSpec
from utils import load_model
import post_processing
import argparse
import boto3

class Inference:
    def __init__(self, input_path=None, output_path=None, plot_spec_results=False, model_path=None, threshold=.5, min_length=500, pad_song=50):
        self.input_path = input_path
        self.output_path = output_path
        self.plot_spec_results = plot_spec_results
        self.threshold = threshold
        self.min_length = min_length
        self.pad_song = pad_song

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight_path = os.path.join(model_path, "weights.pth")
        config_path = os.path.join(model_path, "config.json")
        model = load_model(weight_path=weight_path, config_path=config_path)
        self.model = model.to(self.device)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if plot_spec_results:
            os.makedirs(os.path.join(self.output_path, 'specs'), exist_ok=True)
        
        self.csv_path = os.path.join(self.output_path, 'results.csv')
        self.create_csv_header()

    def create_csv_header(self):
        with open(self.csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'onset_timebin', 'offset_timebin', 'onset_ms', 'offset_ms']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    def sort_all_songs(self):
        total_songs = sum([len(files) for r, d, files in os.walk(self.input_path)])
        with tqdm(total=total_songs, desc="Processing songs") as pbar:
            for root, dirs, files in os.walk(self.input_path):
                for file in files:
                    if file.endswith('.wav'):
                        song_path = os.path.join(root, file)
                        self.sort_single_song(song_path)
                        pbar.update(1)

        print(f"Results saved to {self.csv_path}")

    def sort_single_song(self, song_path):
        wav_to_spec = WavtoSpec()
        song_name = Path(song_path).stem 

        spec = wav_to_spec.process_file(song_path)
        if spec is None:
            print(f"Skipping {song_name}, unable to generate spectrogram.")
            return

        sample_rate, wavfile_signal = wavfile.read(song_path)
        spec_mean = spec.mean()
        spec_std = spec.std()
        spec = (spec - spec_mean) / spec_std

        predictions = post_processing.process_spectrogram(model=self.model, spec=spec, device=self.device, max_length=2048)
        smoothed_song = post_processing.moving_average(predictions, window_size=100)
        processed_song = post_processing.post_process_segments(smoothed_song, min_length=self.min_length, pad_song=self.pad_song, threshold=self.threshold)

        song_status = np.where(processed_song > self.threshold, 1, 0)
        wav_length_ms = (len(wavfile_signal) / sample_rate) * 1000
        timebin_duration_ms = wav_length_ms / len(song_status)

        onsets_offsets = []
        start_index = None
        for index, status in enumerate(song_status):
            if status == 1 and start_index is None:
                start_index = index
            elif status == 0 and start_index is not None:
                end_index = index - 1
                start_ms = start_index * timebin_duration_ms
                end_ms = end_index * timebin_duration_ms
                onsets_offsets.append((start_index, end_index, start_ms, end_ms))
                start_index = None

        if start_index is not None:
            end_index = len(song_status) - 1
            start_ms = start_index * timebin_duration_ms
            end_ms = end_index * timebin_duration_ms
            onsets_offsets.append((start_index, end_index, start_ms, end_ms))

        # Update CSV with results for this song
        self.update_csv(onsets_offsets, song_name)

        # Generate and save spectrogram if plot_spec_results is True
        if self.plot_spec_results:
            post_processing.plot_spectrogram_with_processed_song(
                file_name=song_name,
                spectrogram=spec,
                smoothed_song=smoothed_song,
                processed_song=processed_song,
                directory=os.path.join(self.output_path, 'specs')
            )

    def update_csv(self, onsets_offsets, song_name):
        with open(self.csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['filename', 'onset_timebin', 'offset_timebin', 'onset_ms', 'offset_ms'])
            for onset_timebin, offset_timebin, onset_ms, offset_ms in onsets_offsets:
                writer.writerow({
                    'filename': song_name,
                    'onset_timebin': onset_timebin,
                    'offset_timebin': offset_timebin,
                    'onset_ms': onset_ms,
                    'offset_ms': offset_ms
                })

def main():
    parser = argparse.ArgumentParser(description="Bird song inference")
    parser.add_argument("--mode", choices=["aws", "local_file", "local_dir"], default="local_dir", help="Mode of operation")
    parser.add_argument("--input", help="Input file or directory path for local modes")
    parser.add_argument("--output", help="Output directory path for local modes")
    parser.add_argument("--model", help="Path to the model directory")
    parser.add_argument("--plot_spec", action="store_true", help="Generate spectrograms")
    args = parser.parse_args()

    if args.mode == "aws":
        # S3 trigger mode
        bucket_name = os.environ.get('bucket_name')
        prefix = os.environ.get('prefix')
        filename = os.environ.get('filename')
        model_path = "/path/to/model"  # Update this to the correct path in your Docker image
        output_path = "/tmp/output"  # Temporary output directory

        if bucket_name and prefix and filename:
            s3 = boto3.client('s3')
            local_file_path = f"/tmp/{filename}"
            s3.download_file(bucket_name, filename, local_file_path)

            sorter = Inference(model_path=model_path, output_path=output_path, plot_spec_results=args.plot_spec)
            output_csv_path = sorter.sort_single_song(local_file_path)

            # Upload results back to S3
            s3.upload_file(
                output_csv_path,
                bucket_name,
                f"{prefix}/activity_detection/{os.path.basename(output_csv_path)}"
            )
            if args.plot_spec:
                spec_file = os.path.join(output_path, 'specs', f"{Path(filename).stem}_spectrogram.png")
                if os.path.exists(spec_file):
                    s3.upload_file(
                        spec_file,
                        bucket_name,
                        f"{prefix}/activity_detection/specs/{os.path.basename(spec_file)}"
                    )
        else:
            print("Error: Missing required environment variables for AWS mode")
    elif args.mode == "local_file":
        if not args.input or not args.output or not args.model:
            print("Error: --input, --output, and --model are required for local_file mode")
            return
        sorter = Inference(model_path=args.model, output_path=args.output, plot_spec_results=args.plot_spec)
        sorter.sort_single_song(args.input)
    elif args.mode == "local_dir":
        if not args.input or not args.output or not args.model:
            print("Error: --input, --output, and --model are required for local_dir mode")
            return
        sorter = Inference(input_path=args.input, output_path=args.output, model_path=args.model, plot_spec_results=args.plot_spec)
        sorter.sort_all_songs()
    else:
        print("Invalid mode selected")

if __name__ == "__main__":
    main()

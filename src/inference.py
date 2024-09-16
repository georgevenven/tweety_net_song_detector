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
import librosa
from scipy.signal import windows, spectrogram, ellip, filtfilt
import time
import json

def get_default_model_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "files", "sorter-specs-with-nothreshold-0.1")

def get_default_output_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "output")

class Inference:
    def __init__(self, input_path=None, output_path=None, plot_spec_results=False, model_path=None, threshold=.5, min_length=500, pad_song=50, device=None, aws_mode=False, create_json=True, separate_json=False, step_size=119, nfft=1024):
        self.input_path = input_path
        self.output_path = output_path if output_path else get_default_output_path()
        self.plot_spec_results = plot_spec_results
        self.threshold = threshold
        self.min_length = min_length
        self.pad_song = pad_song
        self.create_json = create_json
        self.separate_json = separate_json
        self.step_size = step_size
        self.nfft = nfft

        if aws_mode:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = get_default_model_path()

        weight_path = os.path.join(model_path, "weights.pth")
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(weight_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"Model files not found at {model_path}")

        model = load_model(weight_path=weight_path, config_path=config_path)
        self.model = model.to(self.device)

        os.makedirs(self.output_path, exist_ok=True)
        if plot_spec_results:
            os.makedirs(os.path.join(self.output_path, 'specs'), exist_ok=True)
        
        if self.create_json and not self.separate_json:
            self.json_path = os.path.join(self.output_path, 'onset_offset_results.json')
            self.create_json_file()

    def create_json_file(self):
        with open(self.json_path, 'w') as jsonfile:
            json.dump([], jsonfile)

    def sort_all_songs(self):
        total_songs = sum([len(files) for r, d, files in os.walk(self.input_path)])
        with tqdm(total=total_songs, desc="Processing songs") as pbar:
            for root, dirs, files in os.walk(self.input_path):
                for file in files:
                    if file.endswith('.wav'):
                        song_path = os.path.join(root, file)
                        start_time = time.time()  # Start timing
                        self.sort_single_song(song_path)
                        elapsed_time = time.time() - start_time  # Calculate elapsed time
                        print(f"Processed {file} in {elapsed_time:.2f} seconds")
                        pbar.update(1)

        if not self.separate_json:
            print(f"Results saved to {self.json_path}")

    def sort_single_song(self, song_path, return_json=False):
        start_time = time.time()  # Start timing

        wav_to_spec = WavtoSpec(None, None, None, step_size=self.step_size, nfft=self.nfft)
        spec, vocalization, labels = wav_to_spec.process_file(wav_to_spec, file_path=song_path)
        song_name = os.path.basename(song_path)

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

        json_data = {
            "filename": song_name,
            "spec_parameters": {
                "sample_rate": sample_rate,
                "spec_mean": spec_mean,
                "spec_std": spec_std,
                "step_size": self.step_size,
                "nfft": self.nfft
            },
            "segments": [
                {
                    "onset_timebin": onset,
                    "offset_timebin": offset,
                    "onset_ms": onset_ms,
                    "offset_ms": offset_ms
                }
                for onset, offset, onset_ms, offset_ms in onsets_offsets
            ]
        }

        if return_json:
            return json.dumps(json_data)
        else:
            if self.create_json:
                if self.separate_json:
                    json_path = os.path.join(self.output_path, f'{song_name}_results.json')
                    with open(json_path, 'w') as jsonfile:
                        json.dump(json_data, jsonfile, indent=4)
                else:
                    self.update_json(onsets_offsets, song_name, json_data["spec_parameters"])

        if self.plot_spec_results:
            post_processing.plot_spectrogram_with_processed_song(
                file_name=song_name,
                spectrogram=spec,
                smoothed_song=smoothed_song,
                processed_song=processed_song,
                directory=os.path.join(self.output_path, 'specs')
            )

        elapsed_time = time.time() - start_time  # Calculate elapsed time
        print(f"Processed {song_name} in {elapsed_time:.2f} seconds")

    def sort_aws_files(self, audio, filenames, sr):
        print(f"Starting sort_aws_files. Number of audio files: {len(audio)}")
        total_start_time = time.time()
        
        for i, (data, filename) in enumerate(zip(audio, filenames)):
            file_start_time = time.time()
            print(f"Processing file {i+1}/{len(audio)}: {filename}")
            print(f"Audio data shape: {data.shape}")
            
            length_in_ms = (data.shape[0] / sr) * 1000
            print(f"Audio length: {length_in_ms:.2f} ms")
            
            filter_start_time = time.time()
            print("Applying high-pass filter")
            b, a = ellip(5, 0.2, 40, 500/(sr/2), 'high')
            data = filtfilt(b, a, data)
            print(f"Filtering time: {time.time() - filter_start_time:.2f} seconds")
            
            spec_start_time = time.time()
            print("Computing spectrogram")
            NFFT = self.nfft
            step_size = self.step_size
            overlap_samples = NFFT - step_size
            window = windows.gaussian(NFFT, std=NFFT/8)
            frequencies, times, spectrogram_data = spectrogram(data, fs=sr, window=window, nperseg=NFFT, noverlap=overlap_samples)
            print(f"Spectrogram shape: {spectrogram_data.shape}")
            print(f"Spectrogram computation time: {time.time() - spec_start_time:.2f} seconds")
            
            post_start_time = time.time()
            print("Post-processing spectrogram")
            clipping_level = -2
            Sxx_log = 10 * np.log10(spectrogram_data)
            Sxx_log_clipped = np.clip(Sxx_log, a_min=clipping_level, a_max=None)
            spectrogram_data = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))
            print(f"Post-processing time: {time.time() - post_start_time:.2f} seconds")
            
            inference_start_time = time.time()
            print("Running model inference")
            predictions = post_processing.process_spectrogram(model=self.model, spec=spectrogram_data, device=self.device, max_length=2048)
            print(f"Model inference time: {time.time() - inference_start_time:.2f} seconds")
            
            post_proc_start_time = time.time()
            print("Applying post-processing")
            smoothed_song = post_processing.moving_average(predictions, window_size=100)
            processed_song = post_processing.post_process_segments(smoothed_song, min_length=self.min_length, pad_song=self.pad_song, threshold=self.threshold)
            print(f"Post-processing time: {time.time() - post_proc_start_time:.2f} seconds")
            
            detection_start_time = time.time()
            print("Detecting onsets and offsets")
            song_status = np.where(processed_song > self.threshold, 1, 0)
            wav_length_ms = (len(data) / sr) * 1000
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

            print(f"Number of detected segments: {len(onsets_offsets)}")
            print(f"Onset/offset detection time: {time.time() - detection_start_time:.2f} seconds")
            
            json_data = {
                "filename": filename,
                "spec_parameters": {
                    "sample_rate": sr,
                    "spec_mean": spectrogram_data.mean(),
                    "spec_std": spectrogram_data.std(),
                    "step_size": self.step_size,
                    "nfft": self.nfft
                },
                "segments": [
                    {
                        "onset_timebin": onset_timebin,
                        "offset_timebin": offset_timebin,
                        "onset_ms": onset_ms,
                        "offset_ms": offset_ms
                    }
                    for onset_timebin, offset_timebin, onset_ms, offset_ms in onsets_offsets
                ]
            }

            json_path = os.path.join(self.output_path, f'{filename}_results.json')
            with open(json_path, 'w') as jsonfile:
                json.dump(json_data, jsonfile, indent=4)
            
            print(f"Finished processing {filename}")
            print(f"Total processing time for this file: {time.time() - file_start_time:.2f} seconds")
            print("--------------------")

        total_time = time.time() - total_start_time
        print(f"Completed sort_aws_files")
        print(f"Total processing time for all files: {total_time:.2f} seconds")
        print(f"Average time per file: {total_time / len(audio):.2f} seconds")

    def update_json(self, onsets_offsets, song_name, spec_parameters):
        with open(self.json_path, 'r+') as jsonfile:
            data = json.load(jsonfile)
            data.append({
                'filename': song_name,
                'spec_parameters': spec_parameters,
                'segments': [
                    {
                        'onset_timebin': onset_timebin,
                        'offset_timebin': offset_timebin,
                        'onset_ms': onset_ms,
                        'offset_ms': offset_ms
                    }
                    for onset_timebin, offset_timebin, onset_ms, offset_ms in onsets_offsets
                ]
            })
            jsonfile.seek(0)
            json.dump(data, jsonfile, indent=4)

def process_filename(file_path):
    print(f"Processing filename: {file_path}")
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    parts = filename_without_ext.split("-")
    processed_parts = parts[1:]
    print(f"Processed filename parts: {processed_parts}")
    return processed_parts

def open_and_process_aws_file(filename, model_path, output_path):
    print(f"Starting to process AWS file: {filename}")
    names = process_filename(filename)
    print(f"Loading audio file: {filename}")
    audio, sr = librosa.load(filename, sr=None, mono=False)
    print(f"Audio loaded. Shape: {audio.shape}, Sample rate: {sr}")

    audio_data = audio[0:len(names)]
    print(f"Extracted audio data. Shape: {audio_data.shape}")

    print(f"Initializing Inference with model_path: {model_path}, output_path: {output_path}")
    sorter = Inference(model_path=model_path, output_path=output_path, device="cpu", aws_mode=True, separate_json=True)
    print("Calling sort_aws_files method")
    sorter.sort_aws_files(audio=audio_data, filenames=names, sr=sr)

def main():
    parser = argparse.ArgumentParser(description="Bird song inference")
    parser.add_argument("--mode", choices=["aws", "local_file", "local_dir"], default="local_dir", help="Mode of operation")
    parser.add_argument("--input", help="Input file or directory path")
    parser.add_argument("--output", default=get_default_output_path(), help="Output directory path")
    parser.add_argument("--model", default=get_default_model_path(), help="Path to the model directory")
    parser.add_argument("--plot_spec", action="store_true", help="Generate spectrograms")
    parser.add_argument("--return_json", action="store_true", help="Return JSON instead of creating CSV (only for local_file mode)")
    parser.add_argument("--separate_json", action="store_true", help="Create separate JSON files for each song")
    parser.add_argument("--step_size", type=int, default=119, help="Step size for spectrogram generation")
    parser.add_argument("--nfft", type=int, default=1024, help="NFFT for spectrogram generation")
    args = parser.parse_args()

    aws_mode = args.mode == "aws"
    local_file_mode = args.mode == "local_file"
    local_dir_mode = args.mode == "local_dir"

    if aws_mode:
        print("Running in AWS mode")
        bucket_name = os.environ.get('bucket_name')
        prefix = os.environ.get('prefix')
        filename = os.environ.get('filename')
        print(f"AWS environment variables: bucket_name={bucket_name}, prefix={prefix}, filename={filename}")
        
        model_path = args.model
        output_path = os.path.join(args.output, 'activity_detection_tweety')
        os.makedirs(output_path, exist_ok=True)
        print(f"Created output directory: {output_path}")

        print(f"Model path: {model_path}")
        print(f"Output path: {output_path}")

        print("Calling open_and_process_aws_file")
        open_and_process_aws_file(filename, args.model, output_path)

    elif local_file_mode:
        sorter = Inference(model_path=args.model, output_path=args.output, plot_spec_results=args.plot_spec, aws_mode=False, create_json=not args.return_json, separate_json=args.separate_json, step_size=args.step_size, nfft=args.nfft, device="cuda")
        if args.input:
            result = sorter.sort_single_song(args.input, return_json=args.return_json)
            if args.return_json:
                print(result)
        else:
            print("Warning: No input file specified. Please provide an input file using --input.")
    elif local_dir_mode:
        sorter = Inference(input_path=args.input, output_path=args.output, model_path=args.model, plot_spec_results=args.plot_spec, aws_mode=False, separate_json=args.separate_json, step_size=args.step_size, nfft=args.nfft, device="cuda" if torch.cuda.is_available() else "cpu")
        if args.input:
            sorter.sort_all_songs()
        else:
            print("Warning: No input directory specified. Please provide an input directory using --input.")
    else:
        print("Invalid mode selected")

if __name__ == "__main__":
    main()
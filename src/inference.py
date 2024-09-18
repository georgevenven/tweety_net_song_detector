import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import ellip, filtfilt, spectrogram, windows
from tqdm import tqdm

from spectogram_generator import WavtoSpec
from utils import load_model
import post_processing
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


def get_default_model_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "files", "sorter-specs-with-nothreshold-0.1")


def get_default_output_path() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "..", "output")

class Inference:
    def __init__(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        plot_spec_results: bool = False,
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        min_length: int = 500,
        pad_song: int = 50,
        device: Optional[str] = None,
        aws_mode: bool = False,
        create_json: bool = True,
        separate_json: bool = False,
        step_size: int = 119,
        nfft: int = 1024,
    ):
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

        self.device = torch.device("cpu") if aws_mode else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = get_default_model_path()

        weight_path = os.path.join(model_path, "weights.pth")
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(weight_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"Model files not found at {model_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        self.context_size = config.get("context_size", 2048)  # Default to 2048 if not found

        # Pass the device as map_location
        self.model = load_model(weight_path=weight_path, config_path=config_path, map_location=self.device).to(self.device)
        self.model.eval()  # Set model to evaluation mode

        os.makedirs(self.output_path, exist_ok=True)
        if plot_spec_results:
            os.makedirs(os.path.join(self.output_path, 'specs'), exist_ok=True)

        if self.create_json and not self.separate_json:
            self.json_path = os.path.join(self.output_path, 'onset_offset_results.json')
            if not os.path.exists(self.json_path):
                self.create_json_file()


    def create_json_file(self) -> None:
        with open(self.json_path, 'w') as jsonfile:
            json.dump([], jsonfile, indent=4)

    def sort_all_songs(self) -> None:
        processed_files = set()
        if not self.separate_json and os.path.exists(self.json_path):
            with open(self.json_path, 'r') as jsonfile:
                data = json.load(jsonfile)
                processed_files = {entry['filename'] for entry in data}

        total_songs = sum(len(files) for _, _, files in os.walk(self.input_path))
        with tqdm(total=total_songs, desc="Processing songs") as pbar:
            for root, _, files in os.walk(self.input_path):
                for file in files:
                    if file.lower().endswith('.wav') and file not in processed_files:
                        song_path = os.path.join(root, file)
                        try:
                            self.sort_single_song(song_path)
                            print(f"Processed {file}")
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
                        pbar.update(1)
                    else:
                        print(f"already processed {file}")
                        pbar.update(1)

        if not self.separate_json:
            print(f"Results saved to {self.json_path}")

    def sort_single_song(self, song_path: str, return_json: bool = False) -> Optional[str]:
        wav_to_spec = WavtoSpec(
            src_dir=None,
            dst_dir=None,
            csv_file_dir=None,
            step_size=self.step_size,
            nfft=self.nfft
        )
        spec, vocalization, labels = wav_to_spec.process_file(file_path=song_path)
        song_name = os.path.basename(song_path)

        if spec is None:
            print(f"Skipping {song_name}, unable to generate spectrogram.")
            return None

        sample_rate, wavfile_signal = wavfile.read(song_path)
        spec_mean = spec.mean()
        spec_std = spec.std()
        spec_normalized = (spec - spec_mean) / spec_std

        with torch.no_grad():
            predictions = post_processing.process_spectrogram(
                model=self.model,
                spec=spec_normalized,
                device=self.device,
                max_length=self.context_size
            )

        smoothed_song = post_processing.moving_average(predictions, window_size=100)
        processed_song = post_processing.post_process_segments(
            smoothed_song,
            min_length=self.min_length,
            pad_song=self.pad_song,
            threshold=self.threshold
        )

        song_status = (processed_song > self.threshold).astype(int)
        wav_length_ms = (len(wavfile_signal) / sample_rate) * 1000
        timebin_duration_ms = (self.step_size / sample_rate) * 1000  # Accurate calculation

        onsets_offsets = self.detect_onsets_offsets(song_status, timebin_duration_ms)

        json_data = {
            "filename": song_name,
            "song_present": bool(onsets_offsets),
            "segments": [
                {
                    "onset_timebin": onset,
                    "offset_timebin": offset,
                    "onset_ms": onset_ms,
                    "offset_ms": offset_ms
                }
                for onset, offset, onset_ms, offset_ms in onsets_offsets
            ],
            "spec_parameters": {
                "step_size": self.step_size,
                "nfft": self.nfft
            }
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
                spectrogram=spec_normalized,
                smoothed_song=smoothed_song,
                processed_song=processed_song,
                directory=os.path.join(self.output_path, 'specs')
            )

        print(f"Processed {song_name}")
        return None

    def sort_aws_files(self, audio: np.ndarray, filenames: List[str], sr: int) -> None:
        print(f"Starting sort_aws_files. Number of audio files: {len(audio)}")
        total_start_time = time.time()

        for i, (data, filename) in enumerate(zip(audio, filenames)):
            file_start_time = time.time()
            print(f"Processing file {i+1}/{len(audio)}: {filename}")
            print(f"Audio data shape: {data.shape}")

            length_in_ms = (len(data) / sr) * 1000
            print(f"Audio length: {length_in_ms:.2f} ms")

            # Apply high-pass filter
            filter_start_time = time.time()
            print("Applying high-pass filter")
            b, a = ellip(5, 0.2, 40, 500 / (sr / 2), 'high')
            filtered_data = filtfilt(b, a, data)
            print(f"Filtering time: {time.time() - filter_start_time:.2f} seconds")

            # Compute spectrogram
            spec_start_time = time.time()
            print("Computing spectrogram")
            frequencies, times, Sxx = spectrogram(
                filtered_data,
                fs=sr,
                window=windows.gaussian(self.nfft, std=self.nfft / 8),
                nperseg=self.nfft,
                noverlap=self.nfft - self.step_size,
                scaling='density',
                mode='magnitude'
            )
            print(f"Spectrogram shape: {Sxx.shape}")
            print(f"Spectrogram computation time: {time.time() - spec_start_time:.2f} seconds")

            # Post-process spectrogram
            post_start_time = time.time()
            print("Post-processing spectrogram")
            Sxx_log = 10 * np.log10(Sxx + 1e-6)
            Sxx_log_clipped = np.clip(Sxx_log, a_min=-2, a_max=None)
            Sxx_normalized = (Sxx_log_clipped - np.min(Sxx_log_clipped)) / (np.max(Sxx_log_clipped) - np.min(Sxx_log_clipped))
            print(f"Post-processing time: {time.time() - post_start_time:.2f} seconds")

            with torch.no_grad():
                print("Running model inference")
                predictions = post_processing.process_spectrogram(
                    model=self.model,
                    spec=Sxx_normalized,
                    device=self.device,
                    max_length=self.context_size
                )
                print(f"Model inference time: {time.time() - spec_start_time:.2f} seconds")

            # Apply post-processing
            print("Applying post-processing")
            smoothed_song = post_processing.moving_average(predictions, window_size=100)
            processed_song = post_processing.post_process_segments(
                smoothed_song,
                min_length=self.min_length,
                pad_song=self.pad_song,
                threshold=self.threshold
            )

            # Detect onsets and offsets
            song_status = (processed_song > self.threshold).astype(int)
            timebin_duration_ms = (self.step_size / sr) * 1000  # Accurate calculation

            onsets_offsets = self.detect_onsets_offsets(song_status, timebin_duration_ms)

            print(f"Number of detected segments: {len(onsets_offsets)}")

            json_data = {
                "filename": filename,
                "song_present": bool(onsets_offsets),
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

    def update_json(self, onsets_offsets: List[Tuple[int, int, float, float]], song_name: str, spec_parameters: Dict[str, Any]) -> None:
        with open(self.json_path, 'r+') as jsonfile:
            data = json.load(jsonfile)
            data.append({
                'filename': song_name,
                'song_present': bool(onsets_offsets),
                'segments': [
                    {
                        'onset_timebin': onset_timebin,
                        'offset_timebin': offset_timebin,
                        'onset_ms': onset_ms,
                        'offset_ms': offset_ms
                    }
                    for onset_timebin, offset_timebin, onset_ms, offset_ms in onsets_offsets
                ],
                'spec_parameters': spec_parameters
            })
            jsonfile.seek(0)
            json.dump(data, jsonfile, indent=4)

    @staticmethod
    def detect_onsets_offsets(song_status: np.ndarray, timebin_duration_ms: float) -> List[Tuple[int, int, float, float]]:
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

        return onsets_offsets


def process_filename(file_path: str) -> List[str]:
    print(f"Processing filename: {file_path}")
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    parts = filename_without_ext.split("-")
    processed_parts = parts[1:]
    print(f"Processed filename parts: {processed_parts}")
    return processed_parts


def open_and_process_aws_file(filename: str, model_path: str, output_path: str) -> None:
    print(f"Starting to process AWS file: {filename}")
    names = process_filename(filename)
    print(f"Loading audio file: {filename}")
    audio, sr = librosa.load(filename, sr=None, mono=False)
    print(f"Audio loaded. Shape: {audio.shape}, Sample rate: {sr}")

    # Ensure that the length of audio data matches the number of names
    # This part is unclear; assuming each name corresponds to a channel
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)
    if audio.shape[0] < len(names):
        print("Warning: Number of audio channels is less than number of names. Truncating names.")
        names = names[:audio.shape[0]]
    audio_data = audio[:len(names)]
    print(f"Extracted audio data. Shape: {audio_data.shape}")

    print(f"Initializing Inference with model_path: {model_path}, output_path: {output_path}")
    sorter = Inference(
        model_path=model_path,
        output_path=output_path,
        device="cpu",
        aws_mode=True,
        separate_json=True,
        step_size=119,
        nfft=1024
    )
    print("Calling sort_aws_files method")
    sorter.sort_aws_files(audio=audio_data, filenames=names, sr=sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bird song inference")
    parser.add_argument("--mode", choices=["aws", "local_file", "local_dir"], default="local_dir",
                        help="Mode of operation")
    parser.add_argument("--input", help="Input file or directory path")
    parser.add_argument("--output", default=get_default_output_path(), help="Output directory path")
    parser.add_argument("--model", default=get_default_model_path(), help="Path to the model directory")
    parser.add_argument("--plot_spec", action="store_true", help="Generate spectrograms")
    parser.add_argument("--return_json", action="store_true",
                        help="Return JSON instead of creating JSON (only for local_file mode)")
    parser.add_argument("--separate_json", action="store_true",
                        help="Create separate JSON files for each song")
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

        if not filename:
            print("Error: 'filename' environment variable is not set.")
            sys.exit(1)

        model_path = args.model
        output_path = os.path.join(args.output, 'activity_detection_tweety')
        os.makedirs(output_path, exist_ok=True)
        print(f"Created output directory: {output_path}")

        print(f"Model path: {model_path}")
        print(f"Output path: {output_path}")

        print("Calling open_and_process_aws_file")
        open_and_process_aws_file(filename, model_path, output_path)

    elif local_file_mode:
        if not args.input:
            print("Warning: No input file specified. Please provide an input file using --input.")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sorter = Inference(
            model_path=args.model,
            output_path=args.output,
            plot_spec_results=args.plot_spec,
            aws_mode=False,
            create_json=not args.return_json,
            separate_json=args.separate_json,
            step_size=args.step_size,
            nfft=args.nfft,
            device=device
        )
        result = sorter.sort_single_song(args.input, return_json=args.return_json)
        if args.return_json and result:
            print(result)

    elif local_dir_mode:
        if not args.input:
            print("Warning: No input directory specified. Please provide an input directory using --input.")
            sys.exit(1)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sorter = Inference(
            input_path=args.input,
            output_path=args.output,
            model_path=args.model,
            plot_spec_results=args.plot_spec,
            aws_mode=False,
            separate_json=args.separate_json,
            step_size=args.step_size,
            nfft=args.nfft,
            device=device
        )
        sorter.sort_all_songs()

    else:
        print("Invalid mode selected")
        sys.exit(1)


if __name__ == "__main__":
    main()

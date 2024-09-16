import csv
import json
import os
import gc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import ellip, filtfilt, spectrogram, windows
from tqdm import tqdm
import matplotlib.pyplot as plt


class WavtoSpec:
    def __init__(
        self,
        src_dir: Optional[str],
        dst_dir: Optional[str],
        csv_file_dir: Optional[str] = None,
        step_size: int = 119,
        nfft: int = 1024
    ):
        self.src_dir = src_dir
        self.dst_dir = dst_dir
        self.csv_file_dir = csv_file_dir
        self.use_csv = csv_file_dir is not None
        self.step_size = step_size
        self.nfft = nfft

    def process_directory(self) -> None:
        audio_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.src_dir)
            for file in files if file.lower().endswith('.wav')
        ]

        skipped_files_count = 0

        for file_path in tqdm(audio_files, desc="Processing files"):
            result = self.convert_to_spectrogram(file_path, save_npz=True)
            if result is None:
                skipped_files_count += 1

        print(f"Total files processed: {len(audio_files)}")
        print(f"Total files skipped due to no vocalization data: {skipped_files_count}")

    def process_file(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[float, float]]], Optional[Dict[str, List[Tuple[float, float]]]]]:
        return self.convert_to_spectrogram(file_path, save_npz=False)

    def convert_to_spectrogram(
        self,
        file_path: str,
        save_npz: bool = False,
        min_length_ms: int = 500,
        min_timebins: int = 200
    ) -> Tuple[Optional[np.ndarray], Optional[List[Tuple[float, float]]], Optional[Dict[str, List[Tuple[float, float]]]]]:
        try:
            with sf.SoundFile(file_path, 'r') as wav_file:
                samplerate = wav_file.samplerate
                data = wav_file.read(dtype='int16')
                if wav_file.channels > 1:
                    data = data[:, 0]

            # Skip small files (less than min_length_ms)
            length_in_ms = (len(data) / samplerate) * 1000
            if length_in_ms < min_length_ms:
                print(f"File {file_path} is below the length threshold and will be skipped.")
                return None, None, None

            file_name = os.path.basename(file_path)

            # Check for vocalization data
            if self.use_csv:
                vocalization_data, phrase_labels = self.check_vocalization(
                    file_name=file_name,
                    data=data,
                    samplerate=samplerate
                )
                if not vocalization_data:
                    print(f"File {file_path} skipped due to no vocalization data.")
                    return None, None, None
            else:
                vocalization_data = [(0, len(data) / samplerate)]
                phrase_labels = {}

            # Apply high-pass filter
            b, a = ellip(5, 0.2, 40, 500 / (samplerate / 2), 'high')
            filtered_data = filtfilt(b, a, data)

            # Compute spectrogram
            frequencies, times, Sxx = spectrogram(
                filtered_data,
                fs=samplerate,
                window=windows.gaussian(self.nfft, std=self.nfft / 8),
                nperseg=self.nfft,
                noverlap=self.nfft - self.step_size,
                scaling='density',
                mode='magnitude'
            )

            Sxx_log = 10 * np.log10(Sxx + 1e-6)

            # Convert phrase labels to timebins
            labels = np.zeros(times.size, dtype=int)
            for label, intervals in phrase_labels.items():
                for start_sec, end_sec in intervals:
                    start_bin = np.searchsorted(times, start_sec)
                    end_bin = np.searchsorted(times, end_sec)
                    labels[start_bin:end_bin] = int(label)

            if times.size >= min_timebins:
                for i, (start_sec, end_sec) in enumerate(vocalization_data):
                    start_bin = np.searchsorted(times, start_sec)
                    end_bin = np.searchsorted(times, end_sec)

                    segment_Sxx_log = Sxx_log[:, start_bin:end_bin]
                    segment_labels = labels[start_bin:end_bin]
                    segment_vocalization = np.ones(end_bin - start_bin, dtype=int)

                    if save_npz and self.dst_dir:
                        spec_filename = os.path.splitext(os.path.basename(file_path))[0]
                        segment_spec_file_path = os.path.join(self.dst_dir, f"{spec_filename}_segment_{i}.npz")
                        np.savez(segment_spec_file_path,
                                 s=segment_Sxx_log,
                                 vocalization=segment_vocalization,
                                 labels=segment_labels)
                        print(f"Segment {i} spectrogram, vocalization data, and labels saved to {segment_spec_file_path}")

                return Sxx_log, vocalization_data, labels

            else:
                print(f"Spectrogram for {file_path} has less than {min_timebins} timebins and will not be saved.")
                return None, None, None

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None, None, None

        finally:
            plt.close('all')
            gc.collect()

    def check_vocalization(
        self,
        file_name: str,
        data: np.ndarray,
        samplerate: int
    ) -> Tuple[List[Tuple[float, float]], Dict[str, List[Tuple[float, float]]]]:
        if not self.use_csv:
            return [(0, len(data) / samplerate)], {}

        # Assuming the CSV file contains mappings for vocalizations
        # Adjust the path if necessary
        csv_file_path = self.csv_file_dir
        if not os.path.exists(csv_file_path):
            print(f"CSV file {csv_file_path} does not exist.")
            return [], {}

        vocalization_data = []
        phrase_labels = {}

        with open(csv_file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row.get('file_name') == file_name:
                    onset_offset_list = eval(row.get('onset/offset', '[]'))
                    vocalization_data = [(onset, offset) for onset, offset in onset_offset_list]

                    # Process phrase labels
                    phrase_data = row.get('phrase_label onset/offsets', '{}')
                    try:
                        phrase_data = json.loads(phrase_data.replace("'", '"'))
                        for label, intervals in phrase_data.items():
                            phrase_labels[label] = intervals
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON for {file_name}. Raw data: {row.get('phrase_label onset/offsets')}")

                    break

        if not vocalization_data:
            print(f"No vocalization data found for {file_name} in CSV.")
        return vocalization_data, phrase_labels

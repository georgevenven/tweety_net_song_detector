# convert wav files to spectograms 
from model import TweetyNet
from utils import load_model
from spectogram_generator import WavtoSpec
import post_processing 
from tqdm import tqdm
import tempfile
import os 
import shutil
import torch 
import numpy as np
import pandas as pd 
from scipy.io import wavfile

class Inference():
    def __init__(self, input_path=None, output_path=None, plot_spec_results=False, model_path=None, threshold=.5, min_length=500, pad_song=50):
        """
        input path == nested bird song structure
        """
        self.input_path = input_path
        self.output_path = output_path
        self.plot_spec_results = plot_spec_results

        weight_path = os.path.join(model_path, "weights.pth")
        config_path = os.path.join(model_path, "config.json")

        self.threshold=threshold
        self.min_length=min_length
        self.pad_song=pad_song

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(weight_path=weight_path, config_path=config_path)
        self.model = model.to(self.device)


        # make sure output dir exists 
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if plot_spec_results:
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.join(self.output_path, 'specs'))
        
        self.database = pd.DataFrame(columns=["song_name", "directory", "song_timebins", "song_ms"])

    def calculate_length_ms(sample_rate, data):
        """
        Calculate the length of an audio file in milliseconds.

        Parameters:
        - sample_rate: int, the sample rate of the audio file (samples per second).
        - data: numpy.ndarray, the audio data.

        Returns:
        - length_ms: float, the length of the audio file in milliseconds.
        """
        # Calculate the number of samples
        num_samples = data.shape[0]

        # Calculate the duration in seconds
        duration_seconds = num_samples / sample_rate

        # Convert the duration to milliseconds
        length_ms = duration_seconds * 1000

        return length_ms
    


    def sort_all_songs(self):
        wav_to_spec = WavtoSpec()
        rows_to_add = []  # Initialize an empty list to collect rows

        for bird in tqdm(os.listdir(self.input_path), desc="Processing birds"):
            bird_path = os.path.join(self.input_path, bird)
            if not os.path.isdir(bird_path):
                continue

            for day in os.listdir(bird_path):
                day_path = os.path.join(bird_path, day)
                if not os.path.isdir(day_path):
                    continue

                for song in os.listdir(day_path):
                    song_src_path = os.path.join(day_path, song)
                    spec = wav_to_spec.process_file(song_src_path)

                    if spec is not None:
                        sample_rate, wavfile_signal = wavfile.read(song_src_path)
                        spec_mean = spec.mean()
                        spec_std = spec.std()
                        spec = (spec - spec_mean) / spec_std

                        predictions = post_processing.process_spectrogram(model=self.model, spec=spec, device=self.device, max_length=2048)
                        smoothed_song = post_processing.moving_average(predictions, window_size=100)
                        processed_song = post_processing.post_process_segments(smoothed_song, min_length=self.min_length, pad_song=self.pad_song, threshold=self.threshold)

                        song_name = os.path.basename(song).split(".")[0]  # Get the base name of the file and remove the extension

                        if self.plot_spec_results:
                            post_processing.plot_spectrogram_with_processed_song(file_name=song_name, spectrogram=spec, smoothed_song=smoothed_song, processed_song=processed_song, directory=os.path.join(self.output_path, 'specs'))
                        
                        song_ms = [index * (1000 / sample_rate) for index in processed_song]
                        new_row = {"song_name": song_name, "directory": song_src_path, "song_timebins": processed_song, "song_ms": song_ms}
                        rows_to_add.append(new_row)  # Append the new row to the list

        # After the loop, concatenate all new rows to the DataFrame at once
        if rows_to_add:  # Check if there are any rows to add
            self.database = pd.concat([self.database, pd.DataFrame(rows_to_add)], ignore_index=True)

        output_csv_path = os.path.join(self.output_path, 'database.csv')
        self.database.to_csv(output_csv_path, index=False)
        print(f"Database saved to {output_csv_path}")

    def sort_single_song(self, song_path):
        wav_to_spec = WavtoSpec()
        song_name = os.path.basename(song_path).split(".")[0]  # Get the base name of the file and remove the extension

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

        song_ms = [index * (1000 / sample_rate) for index in processed_song]
        new_row = {"song_name": song_name, "directory": song_path, "song_timebins": processed_song, "song_ms": song_ms}

        # Check if the song is already in the database and update it, otherwise append a new row
        if song_name in self.database['song_name'].values:
            self.database.loc[self.database['song_name'] == song_name, ['directory', 'song_timebins', 'song_ms']] = [song_path, processed_song, song_ms]
        else:
            self.database = pd.concat([self.database, pd.DataFrame([new_row])], ignore_index=True)

    def visualize_single_spec(self, song_path):
        self.sort_single_song(song_path)  # Process the song to get the necessary data

        song_name = os.path.basename(song_path).split(".")[0]
        spec_row = self.database.loc[self.database['song_name'] == song_name].iloc[0]
        spectrogram = spec_row['spectrogram']  # Assuming 'spectrogram' is stored in the database
        smoothed_song = spec_row['smoothed_song']  # Assuming 'smoothed_song' is stored in the database
        processed_song = spec_row['processed_song']  # Assuming 'processed_song' is stored in the database

        # Visualize the spectrogram without saving the image
        post_processing.plot_spectrogram_with_processed_song(directory=None, file_name=song_name, spectrogram=spectrogram, smoothed_song=smoothed_song, processed_song=processed_song)


sorter = Inference(input_path = "/home/george-vengrovski/Documents/data/song_sorter_test", output_path = "/home/george-vengrovski/Documents/projects/tweety_net_song_detector/temp", plot_spec_results=True, model_path="/home/george-vengrovski/Documents/projects/tweety_net_song_detector/files/sorter1")
sorter.sort_all_songs()

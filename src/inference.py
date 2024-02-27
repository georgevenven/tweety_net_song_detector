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

    def create_temp_folder(self, base_path="/files"):
        # Ensure the base path exists
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # Create a temporary directory within the base path
        temp_folder_path = tempfile.mkdtemp(dir=base_path)
        return temp_folder_path

    def sort_songs(self):
        wav_to_spec = WavtoSpec()

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
                    spec_mean = spec.mean()
                    spec_std = spec.std()
                    spec = (spec - spec_mean) / spec_std
                    if spec is not None:
                        predictions = post_processing.process_spectrogram(threshold=self.threshold, min_length=self.min_length, model=self.model, pad_song=self.pad_song, spec=spec, device=self.device)
                        smoothed_song = post_processing.moving_average(song, window_size=100)
                        smoothed_times = np.arange(len(smoothed_song)) + 50  # Offset for alignment

                        print(predictions.shape)
                        # put model through 
            #         break   
            #     break
            # break 


sorter = Inference(input_path = "/home/george-vengrovski/Documents/data/song_sorter_test", output_path = "", plot_spec_results=True, model_path="/home/george-vengrovski/Documents/projects/tweety_net_song_detector/files/sorter1")
sorter.sort_songs()

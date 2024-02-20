# convert wav files to spectograms 
from model import TweetyNet
from utils import load_model
from spectogram_generator import WavtoSpec
import tempfile


import os 

class Inference():
    def __init__(self, input_path=None, output_path=None, plot_spec_results=False, model_path=None):
        self.input_path = input_path
        self.output_path = output_path
        self.plot_spec_results = plot_spec_results

        weight_path = os.path.join(model_path, "weights.pth")
        config_path = os.path.join(model_path, "config.json")

        self.model = load_model(weight_path=weight_path, config_path=config_path)

    def create_temp_folder(self, base_path="/files"):
        # Ensure the base path exists
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        # Create a temporary directory within the base path
        temp_folder_path = tempfile.mkdtemp(dir=base_path)
        return temp_folder_path

    def process(self):
        # Create a temporary folder to store .npz files
        temp_folder = self.create_temp_folder()

        # Assuming WavtoSpec is a class that takes an input wav file path and an output directory
        # Initialize WavtoSpec with the input path and the temporary folder path
        wav_to_spec = WavtoSpec(self.input_path, temp_folder)
        wav_to_spec.process_directory()
        





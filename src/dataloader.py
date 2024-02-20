import torch.nn.functional as F
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

class SongDetectorDataClass(Dataset):
    def __init__(self, file_dir):
        self.file_paths = []

        for file in os.listdir(file_dir):
            self.file_paths.append(os.path.join(file_dir, file))

    def __getitem__(self, index):        
        file_path = self.file_paths[index]

        data = np.load(file_path, allow_pickle=True)
        spectogram = data['s']
        ground_truth_labels = data['song']

        # This needs to be done because we are no longer operating on preprocessed data 
        # Z-score normalization
        mean_val, std_val = spectogram.mean(), spectogram.std()
        spectogram = (spectogram - mean_val) / (std_val + 1e-7)
        spectogram[np.isnan(spectogram)] = 0
        # spectogram = spectogram[20:216, :]

        ground_truth_labels = torch.tensor(ground_truth_labels, dtype=torch.int64).squeeze(0)
        spectogram = torch.from_numpy(spectogram).float().permute(1, 0)

        return spectogram.T, ground_truth_labels

    def __len__(self):
        return len(self.file_paths)

class CollateFunctionSongDetection:
    def __init__(self, segment_length=768):
        self.segment_length = segment_length

    def __call__(self, batch):
        # Unzip the batch
        spectograms, ground_truth_labels = zip(*batch)

        # Create lists to hold the processed tensors
        spectograms_processed = []
        ground_truth_labels_processed = []

        for spectogram, ground_truth_label in zip(spectograms, ground_truth_labels):            
            if spectogram.shape[1] < self.segment_length:
                pad_amount = self.segment_length - spectogram.shape[1]
                spectogram = F.pad(spectogram, (0, pad_amount), 'constant', 0)
                ground_truth_label = F.pad(ground_truth_label.unsqueeze(0), (0, pad_amount), 'constant', 0).squeeze(0)

            # Truncate if larger than context window
            if spectogram.shape[1] > self.segment_length:
                # get random view of size segment
                # find range of valid starting pts (essentially these are the possible starting pts for the length to equal segment window)
                starting_points_range = spectogram.shape[1] - self.segment_length        
                start = torch.randint(0, starting_points_range, (1,)).item()  
                end = start + self.segment_length     

                spectogram = spectogram[:, start:end]
                ground_truth_label = ground_truth_label[start:end]

            spectograms_processed.append(spectogram)
            ground_truth_labels_processed.append(ground_truth_label)


        # Stack tensors along a new dimension
        spectograms = torch.stack(spectograms_processed, dim=0)
        spectograms = spectograms.unsqueeze(1)
        ground_truth_labels = torch.stack(ground_truth_labels_processed, dim=0)

        return spectograms, ground_truth_labels

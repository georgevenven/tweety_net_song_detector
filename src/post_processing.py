import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tqdm import tqdm  # Import tqdm for progress tracking

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def moving_average(signal, window_size):
    """Compute the moving average of the given signal with the specified window size."""
    cumsum_vec = np.cumsum(np.insert(signal, 0, 0)) 
    return (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size

def post_process_segments(smoothed_song, threshold, min_length, pad_song):
    """Post-process the smoothed song to adjust segments shorter than min_length and apply padding."""
    processed_song = np.zeros_like(smoothed_song)
    above_threshold = smoothed_song >= threshold
    start = None

    for i, value in enumerate(above_threshold):
        if value and start is None:
            start = i  # Mark the start of a new segment
        elif not value and start is not None:
            # Segment end found; check if it meets the min_length requirement
            if i - start >= min_length:
                # Apply padding to segments that meet the min_length requirement
                start_pad = max(start - pad_song, 0)  # Ensure start_pad is not less than 0
                end_pad = min(i + pad_song, len(above_threshold))  # Ensure end_pad does not exceed array length
                processed_song[start_pad:end_pad] = smoothed_song[start_pad:end_pad]
            start = None  # Reset start for the next segment

    # Handle the case where a segment extends to the end of the array
    if start is not None and len(above_threshold) - start >= min_length:
        start_pad = max(start - pad_song, 0)
        end_pad = min(len(above_threshold) + pad_song, len(above_threshold))
        processed_song[start_pad:end_pad] = smoothed_song[start_pad:end_pad]

    return processed_song

def process_spectrogram(model, spec, device, max_length=2048):
    """
    Process the spectrogram in chunks, pass through the classifier, and return the binary predictions based on BCE.
    """
    # Calculate the number of chunks needed
    num_chunks = int(np.ceil(spec.shape[1] / max_length))
    combined_predictions = []

    for i in range(num_chunks):
        # Extract the chunk
        start_idx = i * max_length
        end_idx = min((i + 1) * max_length, spec.shape[1])
        chunk = spec[:, start_idx:end_idx]
        # Forward pass through the model
        # Ensure chunk is on the correct device
        chunk_tensor = torch.Tensor(chunk).unsqueeze(0).unsqueeze(0).to(device)
        
        logits = model(chunk_tensor)
        logits = logits.squeeze().detach().cpu()
        logits = sigmoid(logits)

        combined_predictions.append(logits)

    # Concatenate all chunks' predictions
    final_predictions = np.concatenate(combined_predictions, axis=-1)

    return final_predictions

def process_files(src):
    """
    Process each file in the directory, reshape predictions, and overwrite the original files with the processed data.
    """
    files = os.listdir(src)
    for file in tqdm(files, desc="Processing files"):  # Wrap the loop with tqdm for progress tracking
        file_path = os.path.join(src, file)

        try:
            # Load the spectrogram from the file
            f = np.load(file_path, allow_pickle=True)
            spec = f['s']

            # Z-score normalization
            spec_mean = spec.mean()
            spec_std = spec.std()
            spec = (spec - spec_mean) / spec_std

            # Process the spectrogram and get predictions
            predictions = process_spectrogram(spec)

            # Overwrite the original file with the spectrogram and predictions
            np.savez(file_path, s=spec, song=predictions)  # Use the original `file_path` to overwrite

        except Exception as e:
            print(f"Failed to process file {file}: {str(e)}")
import torch 
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, model, train_loader, test_loader, device, lr=1e-2, plotting=False, batches_per_eval=100, desired_total_batches=1e4, patience=8, use_tqdm=True, moving_avg_window = 200):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-2, weight_decay=0.0)
        self.plotting = plotting
        self.batches_per_eval = batches_per_eval
        self.desired_total_batches = desired_total_batches
        self.patience = patience
        self.use_tqdm = use_tqdm
        self.moving_avg_window = moving_avg_window  # Window size for moving average

    def frame_error_rate(self, y_pred, y_true):
        # Threshold y_pred at 0.5 to obtain binary predictions
        y_pred_binary = (y_pred > 0.5).float()
        # Calculate mismatches between binary predictions and true labels
        mismatches = (y_pred_binary != y_true).float()
        # Compute the error rate
        error = mismatches.sum() / y_true.numel()
        
        return error * 100

    def validate_model(self):
        self.model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        total_frame_error = 0
        num_val_batches = 0

        with torch.no_grad():
            for i, (spectrogram, label) in enumerate(self.test_loader):
                if i > self.batches_per_eval:
                    break
                spectrogram, label = spectrogram.to(self.device), label.to(self.device).float()

                output = self.model.forward(spectrogram)
                output = output.squeeze(1)
                output = torch.sigmoid(output)
                loss = self.model.loss_function(y_pred=output, y_true=label)

                total_val_loss += loss.item()
                total_frame_error += self.frame_error_rate(output, label).item()
                num_val_batches += 1

        self.model.train()  # Set the model back to training mode
        avg_val_loss = total_val_loss / num_val_batches
        avg_frame_error = total_frame_error / num_val_batches
        return avg_val_loss, avg_frame_error

    def moving_average(self, values, window):
        """Simple moving average over a list of values"""
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma.tolist()

    def train(self):
        total_batches = 0
        best_val_loss = float('inf')
        num_val_no_improve = 0
        stop_training = False

        raw_loss_list, raw_val_loss_list, raw_frame_error_rate_list, smooth_train_loss = [], [], [], []

        while total_batches < self.desired_total_batches:
            for i, (spectrogram, label) in enumerate(self.train_loader):
                if total_batches >= self.desired_total_batches:
                    break

                spectrogram, label = spectrogram.to(self.device), label.to(self.device).float()

                output = self.model.forward(spectrogram)
                output = output.squeeze(1)
                output = torch.sigmoid(output)

                loss = self.model.loss_function(y_pred=output, y_true=label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                raw_loss_list.append(loss.item())
                total_batches += 1

                if total_batches % self.batches_per_eval == 0:
                    avg_val_loss, avg_frame_error = self.validate_model()
                    raw_val_loss_list.append(avg_val_loss)
                    raw_frame_error_rate_list.append(avg_frame_error)

                    if len(raw_loss_list) < self.moving_avg_window:
                        # Print non-smoothed loss values
                        print(f'Batch {total_batches}: FER = {avg_frame_error:.2f}%, Train Loss = {raw_loss_list[-1]:.4f}, Val Loss = {avg_val_loss:.4f}')
                    else:
                        # Print smoothed loss values
                        smooth_train_loss = self.moving_average(raw_loss_list, self.moving_avg_window)[-1]
                        smooth_val_loss = self.moving_average(raw_val_loss_list, self.moving_avg_window)[-1]
                        print(f'Batch {total_batches}: FER = {avg_frame_error:.2f}%, Train Loss = {smooth_train_loss:.4f}, Val Loss = {smooth_val_loss:.4f}')
                        
                        if smooth_val_loss < best_val_loss:
                            best_val_loss = smooth_val_loss
                            num_val_no_improve = 0
                        else:
                            num_val_no_improve += 1
                            if num_val_no_improve >= self.patience:
                                print("Early stopping triggered")
                                stop_training = True
                                break

                if stop_training:
                    break
            if stop_training:
                break

        if self.plotting:
            self.plot_results(smooth_train_loss, raw_val_loss_list, raw_frame_error_rate_list)

    def plot_results(self, loss_list, val_loss_list, frame_error_rate_list):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(loss_list, label='Training Loss')
        plt.plot(val_loss_list, label='Validation Loss')
        plt.title('Loss over Batches')
        plt.xlabel('Batches')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(frame_error_rate_list, label='Frame Error Rate', color='red')
        plt.title('Frame Error Rate over Batches')
        plt.xlabel('Batches')
        plt.ylabel('Error Rate (%)')
        plt.legend()

        plt.tight_layout()
        plt.show()
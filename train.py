import sys
import torch 
sys.path.append("src")
from utils import save_model_config_and_weights

config = {
"model_name": "sorter-specs-with-nothreshold-0.1",
"hidden_size": 32,
"context_size": 4096,
"batch_size": 24,
"num_batches_train": 100,
"lr": 3e-4
}

from torch.utils.data import DataLoader
from dataloader import CollateFunctionSongDetection, SongDetectorDataClass


collate_fn = CollateFunctionSongDetection(segment_length=4096)

test_class = SongDetectorDataClass("/media/george-vengrovski/disk2/training_song_detector/labeled_song_dataset/test", augment=False)
test_loader = DataLoader(test_class, batch_size=config['batch_size'], shuffle=True, num_workers=16, collate_fn=collate_fn)

train_class = SongDetectorDataClass("/media/george-vengrovski/disk2/training_song_detector/labeled_song_dataset/train", augment=False)
train_loader = DataLoader(train_class, batch_size=config['batch_size'], shuffle=True, num_workers=16, collate_fn=collate_fn)


from model import TweetyNet
from trainer import Trainer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = TweetyNet(hidden_size=config['hidden_size'], rnn_dropout=0.2, num_classes=1, input_shape=(1, 512, 4096))
model = model.to(device)

trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, device=device, lr=config['lr'], plotting=True, batches_per_eval=10, desired_total_batches=200, patience=8)
trainer.train()

### save model config to /files/modelname/config.py and final weights to files/modelname/weights.pth 
save_model_config_and_weights(trainer, config, config['model_name'])

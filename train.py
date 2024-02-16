import sys
import torch 
sys.path.append("src")

config = {
"model_name": "Test",
"hidden_size": 384,
"context_size": 768,
"batch_size": 24,
"num_batches_train": 5e3
}

from torch.utils.data import DataLoader
from dataloader import CollateFunctionSongDetection, SongDetectorDataClass
from model import TweetyNet
from trainer import Trainer 

collate_fn = CollateFunctionSongDetection(segment_length=config['context_size'])

test_class = SongDetectorDataClass("/home/george-vengrovski/Documents/data/tweety_net_test")
test_loader = DataLoader(test_class, batch_size=config['batch_size'], shuffle=True, num_workers=16, collate_fn=collate_fn)

train_class = SongDetectorDataClass("/home/george-vengrovski/Documents/data/tweety_net_train")
train_loader = DataLoader(test_class, batch_size=config['batch_size'], shuffle=True, num_workers=16, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TweetyNet(hidden_size=config['hidden_size'], rnn_dropout=0.2, num_classes=1)
model = model.to(device)

trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, device=device, lr=3e-4, plotting=True, batches_per_eval=25, desired_total_batches=5e3, patience=8)
trainer.train()
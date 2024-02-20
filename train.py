import sys
import torch 
sys.path.append("src")
from utils import save_model_config_and_weights

config = {
"model_name": "sorter1",
"hidden_size": 384,
"context_size": 1024,
"batch_size": 24,
"num_batches_train": 5e3,
"lr": 3e-4
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

model = TweetyNet(input_shape=(1, 512, config['context_size']), hidden_size=config['hidden_size'], rnn_dropout=0.2, num_classes=1)
model = model.to(device)

trainer = Trainer(model=model, train_loader=train_loader, test_loader=test_loader, device=device, lr=config["lr"], plotting=True, batches_per_eval=25, desired_total_batches=5e3, patience=8)
trainer.train()

### save model config to /files/modelname/config.py and final weights to files/modelname/weights.pth 
save_model_config_and_weights(trainer, config, config['model_name'])

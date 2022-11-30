from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.datasets.ego_autoencoder_dataset import EGOFlowDataset
from sense.utils.arguments import parse_args

import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys
import math
import pickle
from datetime import datetime
import numpy as np

# https://discuss.pytorch.org/t/rmsle-loss-function/67281/2
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def data_cacher(path):
    cache_file_path = 'cache/ego_flow.pkl'
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_data = pickle.load(f)
            train_data = cached_data['train_data']
            test_data = cached_data['test_data']
    else:
        data_list = os.listdir(path)
        train_data, test_data = np.split(data_list, [int(len(data_list)*0.8)])
        with open(cache_file_path, 'wb') as f:
            pickle.dump(
                {
                    'train_data': train_data,
                    'test_data': test_data
                },
                f, pickle.HIGHEST_PROTOCOL)
    return train_data, test_data

  
def make_data_loader(path, args):
    transform = None
    train_data, test_data = data_cacher(path)
    train_set = EGOFlowDataset(root=str(path), path_list=train_data, transform=transform)
    test_set = EGOFlowDataset(root=str(path), path_list=test_data, transform=transform)
    return torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1)))
        ), torch.utils.data.DataLoader(
            test_set,
            batch_size=2,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=True
        )

def train(model, optimizer, data, criteria, args):
    print(f"Shape => {data.shape}")
    model.train()
    data = data.cuda()
    optimizer.zero_grad()
    data_pred = model(data)
    loss = criteria(data, data_pred)
    loss.backward()
    optimizer.step()
    loss = loss.item()
    return loss

def validation(model, data, criteria):
    model.eval()
    data = data.cuda()
    
    with torch.no_grad():
        data_pred = model(data)
        loss = criteria(data, data_pred)
    loss = loss.item()
    return loss
        
def save_checkpoint(model, optimizer, epoch, global_step, args):
    pass

def main(args):
    # TODO Data load
    train_loader, validation_loader = make_data_loader(args.savemodel, args)

    # Make model
    model = EGOAutoEncoder()
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()]))
    )
    # Optimizer
    optimizer = optim.AdamW(model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.0004
    )
    # Criteria
    criteria = RMSLELoss()

    # Save & Load model

    # Print format
    print_format = '{}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{}\t{:.6f}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Train
    start_epoch = 1
    global_step = 0
    lr = args.lr
    train_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = datetime.now()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_start = datetime.now()
            train_loss = train(model, optimizer, batch_data, criteria, args)
            global_step += 1
            if (batch_idx + 1) % args.print_freq == 0:
                print(print_format.format(
                    'Train', epoch, batch_idx, len(train_loader),
                    train_loss, str(datetime.now() - batch_start), lr))
                sys.stdout.flush()

        val_start = datetime.now()
        val_loss = 0
        for batch_idx, batch_data in enumerate(validation_loader):
            val_loss += validation(model, batch_data, criteria)
        print(print_format.format(
            'Val', epoch, 0, len(validation_loader),
            val_loss /  len(validation_loader), str(datetime.now() - val_start), lr))
        # Save model
        print(f'Epoch {epoch} elapsed time => {str(datetime.now() - epoch_start)}')
    print(f'Train elapsed time => {str(datetime.now() - train_start)}')


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    main(args)
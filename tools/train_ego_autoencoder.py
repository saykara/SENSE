from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.datasets.ego_autoencoder_dataset import EGOFlowDataset
from sense.utils.arguments import parse_args

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

import os
import sys
import cv2
import math
import pickle
import random
import imageio
from datetime import datetime
import numpy as np

import sense.models.model_utils as model_utils
import sense.utils.kitti_viz as kitti_viz
import tools.demo as demo
import tools.train_joint_synthetic_sceneflow as tjss

# https://discuss.pytorch.org/t/rmsle-loss-function/67281/2
class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def mse_loss(self, y_pred, y_true):
        squared_error = (y_pred - y_true) ** 2
        sum_squared_error = torch.sum(squared_error)
        a = list(y_true.size())
        div = 1
        for i in a:
            div *= i
        
        loss = torch.divide(sum_squared_error, div)
        return loss

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1))).cuda()

def make_flow_data_helper(base_dir):
    train_list = []
    val_list = []

    # Flyingthings3d
    # fly_train_dir = os.path.join(base_dir, "flyingthings3d", "train")
    fly_train_dir = base_dir + "/" + "flyingthings3d" + "/" + "train"
    for img in os.listdir(fly_train_dir):
        train_list.append(fly_train_dir + "/" + img)

    # fly_val_dir = os.path.join(base_dir, "flyingthings3d", "val")
    fly_val_dir = base_dir + "/" + "flyingthings3d" + "/" + "val"
    for img in os.listdir(fly_val_dir):
        val_list.append(fly_val_dir + "/" + img)
  
     # Monkaa
    # monkaa_dir = os.path.join(base_dir, "monkaa")
    monkaa_dir = base_dir + "/" + "monkaa"
    for dir in os.listdir(monkaa_dir):
        for img in os.listdir(monkaa_dir + "/" + dir):
            train_list.append(monkaa_dir + "/" + dir + "/" + img)

     # Driving
    # driving_dir = os.path.join(base_dir, "driving")
    driving_dir = base_dir + "/" + "driving"
    for focal in os.listdir(driving_dir):
        for direction in os.listdir(driving_dir + "/" + focal):
            for speed in os.listdir(driving_dir + "/" + focal + "/" + direction):
                for img in os.listdir(driving_dir + "/" + focal + "/" + direction + "/" + speed):
                    train_list.append(driving_dir + "/" + focal + "/" + direction + "/" + speed  + "/" + img)
     # Sintel training
    # driving_dir = os.path.join(base_dir, "driving")
    sintel_dir = base_dir + "/" + "sintel"
    for dir in os.listdir(sintel_dir + "/" + "training"):
        for img in os.listdir(sintel_dir + "/" + "training" + "/" + dir):
            train_list.append(sintel_dir + "/" + "training" + "/" + dir + "/" + img)
   
    # Sintel stereo
    for dir in os.listdir(sintel_dir + "/" + "stereo"):
        for img in os.listdir(sintel_dir + "/" + "stereo" + "/" + dir):
            train_list.append(sintel_dir + "/" + "stereo" + "/" + dir + "/" + img)

    return train_list, val_list

def data_cacher(path):
    cache_file_path = 'cache/ego_flow.pkl'
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_data = pickle.load(f)
            train_data = cached_data['train_data']
            test_data = cached_data['test_data']
    else:
        # data_list = os.listdir(path)
        train_data, test_data = make_flow_data_helper(path)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(
                {
                    'train_data': train_data,
                    'test_data': test_data
                },
                f, pickle.HIGHEST_PROTOCOL)
    return train_data, test_data
    
def make_data_loader(path, args):
    # TODO Reconsider transforms
    transform = torchvision.transforms.Compose([
            transforms.RandomResizedCrop((544, 992)),
            transforms.RandomHorizontalFlip(0.3)])
    train_data, test_data = data_cacher(path)
    print("Train data sample size: ", len(train_data))
    print("Test data sample size: ", len(test_data))
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
    model.train()
    data = data.cuda()
    optimizer.zero_grad()
    data_pred = model(data)
    loss = criteria(data_pred, data)
    loss.backward()
    optimizer.step()
    loss = loss.item()
    return loss

def validation(model, data, criteria):
    model.eval()
    data = data.cuda()
    
    with torch.no_grad():
        data_pred = model(data)
        loss = criteria(data_pred, data)
    loss = loss.item()
    return loss
        
def save_checkpoint(model, optimizer, epoch, global_step, args, final=False):
    #SAVE
    now = datetime.now().strftime("%d-%m-%H-%M")
    save_dir = f"ego_autoencoder_{now}"
    if final:
        save_dir = f"ego_encoder_{now}"
    save_dir = os.path.join(args.savemodel, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, 'model_{:04d}.pth'.format(epoch))

    if epoch % args.save_freq == 0:
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
            }, model_path)
        print('<=== checkpoint has been saved to {}.'.format(model_path))

def main(args):
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)   
    np.random.seed(args.seed)  
    random.seed(args.seed)
    
    # Data load
    dataset = "E:/Thesis/content/flow_dataset"
    train_loader, validation_loader = make_data_loader(dataset, args)
    
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
    if args.loadmodel is not None:
        ckpt = torch.load(args.loadmodel)
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)
        print('==> A pre-trained checkpoint has been loaded.')
 
    if args.resume is not None:
        ckpt = torch.load(args.resume)
        start_epoch = ckpt['epoch'] + 1
        optimizer.load_state_dict(ckpt['optimizer'])
        model.load_state_dict(ckpt['state_dict'])
        print('==> Manually resumed training from {}.'.format(args.resume))

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
        save_checkpoint(model, optimizer, epoch, global_step, args)
    save_checkpoint(model.encoder, optimizer, epoch, global_step, args, True)
    print(f'Train elapsed time => {str(datetime.now() - train_start)}')


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    args.stride = 32
    main(args)
from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.models.ego_autoencoder import EGOEncoder
from sense.models.ego_rnn import EgoRnn 
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
from sense.lib.nn import DataParallelWithCallback


def make_data_helper(path):
    pass

def data_cacher(path):
    cache_file_path = 'cache/ego_flow.pkl'
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_data = pickle.load(f)
            train_data = cached_data['train_data']
            test_data = cached_data['test_data']
    else:
        # data_list = os.listdir(path)
        train_data, test_data = make_data_helper(path)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(
                {
                    'train_data': train_data,
                    'test_data': test_data
                },
                f, pickle.HIGHEST_PROTOCOL)
    return train_data, test_data

def make_data_loader(dataset, args):
    pass

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
        
def save_checkpoint(model, optimizer, epoch, global_step, args):
    #SAVE
    now = datetime.now().strftime("%d-%m-%H-%M")
    save_dir = f"lstm_{now}"
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
    
    # Flow producer model
    holistic_scene_model = model_utils.make_model(args, do_flow=True, do_disp=True, do_seg=True)
    holistic_scene_model_path = 'data/pretrained_models/kitti2012+kitti2015_new_lr_schedule_lr_disrupt+semi_loss_v3.pth'
    ckpt = torch.load(holistic_scene_model_path)
    state_dict = ckpt['state_dict']
    holistic_scene_model.load_state_dict(state_dict)
    holistic_scene_model.eval()
    
    # EGO encoder model
    encoder_model = EGOEncoder("syncbn", "gelu")
    encoder_model = DataParallelWithCallback(encoder_model).cuda()
    encoder_model_path = '<path>'
    ckpt = torch.load(encoder_model_path)
    state_dict = ckpt['state_dict']
    encoder_model.load_state_dict(state_dict)
    encoder_model.eval()
    
    # Data load
    dataset = "E:/Thesis/content/flow_dataset"
    train_loader, validation_loader = make_data_loader(dataset, args)
    # train_loader, validation_loader, disp_test_loader = tjss.make_data_loader(args)
    
    # Make model
    model = EgoRnn(4096)
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
    # TODO Criteria
    criteria = None

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
        
        print(f'Epoch {epoch} elapsed time => {str(datetime.now() - epoch_start)}')
        # Save model
        save_checkpoint(model, optimizer, epoch, global_step, args)
    save_checkpoint(model, optimizer, epoch, global_step, args)
    print(f'Train elapsed time => {str(datetime.now() - train_start)}')


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    args.stride = 32
    main(args)
from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.datasets.ego_autoencoder_dataset import EGOFlowDataset, load_flow
from sense.utils.arguments import parse_autoencoder_args
from sense.lib.nn import DataParallelWithCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import os
import sys
import math
import pickle
import random
from datetime import datetime
import numpy as np

import sense.datasets.flow_transforms as flow_transforms

BASE_DIR='/content/dataset'

temp_save = None

def write_flo(filename, flow):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close() 
    
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
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def make_flow_data_helper(args):
    train_list = []
    val_list = []
    if args.dataset == "local":
        fly_train_dir = "E:/Thesis/Flow/train"
        for img in os.listdir(fly_train_dir):
            train_list.append(fly_train_dir + "/" + img)
        fly_train_dir = "E:/Thesis/Flow/val"
        for img in os.listdir(fly_train_dir):
            val_list.append(fly_train_dir + "/" + img)
            
    elif args.dataset == "sceneflow":
        scene_dir = os.path.join(BASE_DIR, "flow_dataset")
        # Flyingthings3d
        fly_train_dir = os.path.join(scene_dir, "flyingthings3d", "train")
        for img in os.listdir(fly_train_dir):
            train_list.append(os.path.join(fly_train_dir, img))

        fly_val_dir = os.path.join(scene_dir, "flyingthings3d", "val")
        for img in os.listdir(fly_val_dir):
            val_list.append(os.path.join(fly_val_dir, img))

        # Monkaa
        monkaa_dir = os.path.join(scene_dir, "monkaa")
        for dir in os.listdir(monkaa_dir):
            for img in os.listdir(os.path.join(monkaa_dir, dir)):
                train_list.append(os.path.join(monkaa_dir, dir, img))

        # Driving
        driving_dir = os.path.join(scene_dir, "driving")
        for focal in os.listdir(driving_dir):
            for direction in os.listdir(os.path.join(driving_dir, focal)):
                for speed in os.listdir(os.path.join(driving_dir, focal, direction)):
                    for img in os.listdir(os.path.join(driving_dir, focal, direction, speed)):
                        train_list.append(os.path.join(driving_dir, focal, direction, speed, img))
        
        # Sintel training
        sintel_dir = os.path.join(scene_dir, "sintel")
        for dir in os.listdir(os.path.join(sintel_dir, "training")):
            for img in os.listdir(os.path.join(sintel_dir, "training", dir)):
                train_list.append(os.path.join(sintel_dir, "training", dir, img))

        # Sintel stereo
        for dir in os.listdir(os.path.join(sintel_dir, "stereo")):
            for img in os.listdir(os.path.join(sintel_dir, "stereo", dir)):
                train_list.append(os.path.join(sintel_dir, "stereo", dir, img))
                
    elif args.dataset == "kittimalaga":
        raise NotImplementedError
        # TODO Implement Kitti_vo and Malaga dataset loader 
        #fly_train_dir = os.path.join(BASE_DIR, "flyingthings3d", "train")
        #for img in os.listdir(fly_train_dir):
        #    train_list.append(os.path.join(fly_train_dir, img))

        #fly_val_dir = os.path.join(BASE_DIR, "flyingthings3d", "val")
        #for img in os.listdir(fly_val_dir):
        #    val_list.append(os.path.join(fly_val_dir, img))
    
    else:
        raise f"Invalid dataset => {args.dataset}"
    
    return train_list, val_list

def data_cacher(args):
    if not os.path.exists("./cache"):
        os.mkdir("cache")
    cache_file_path = f'cache/{args.dataset}_ego_flow.pkl'
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_data = pickle.load(f)
            train_data = cached_data['train_data']
            test_data = cached_data['test_data']
    else:
        train_data, test_data = make_flow_data_helper(args)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(
                {
                    'train_data': train_data,
                    'test_data': test_data
                },
                f, pickle.HIGHEST_PROTOCOL)
    return train_data, test_data
    
def make_data_loader(args):
    height_new = args.flow_crop_imh
    width_new = args.flow_crop_imw
    transform = transforms.Compose([
        flow_transforms.NormalizeFlowOnly(mean=[0,0],std=[120.0, 120.0]),
        flow_transforms.ArrayToTensor(),
        transforms.RandomResizedCrop((height_new, width_new)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3),
    ]) 
 
    train_data, test_data = data_cacher(args)
    print("Train data sample size: ", len(train_data))
    print("Test data sample size: ", len(test_data))
    
    path = "E:/Thesis/content/flow_dataset" if args.dataset == "local" else "/content/flow_dataset"
    
    train_set = EGOFlowDataset(root=path, path_list=train_data, transform=transform)
    test_set = EGOFlowDataset(root=path, path_list=test_data, transform=transform)
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
    data = Variable(data).cuda()
    data_pred = model(data)
    loss = criteria(data_pred, data)
    optimizer.zero_grad()
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
    global temp_save
    now = datetime.now().strftime("%d-%m-%H-%M")
    if temp_save == None:
        temp_save = f"ego_autoencoder_{now}"
    save_dir = os.path.join(args.savemodel, temp_save)
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
    train_loader, validation_loader = make_data_loader(args)
    
    # Make model
    model = EGOAutoEncoder(args.bn_type)
    
    if args.bn_type == 'plain':
        model = torch.nn.DataParallel(model).cuda()
    elif args.bn_type == 'syncbn':
        model = DataParallelWithCallback(model).cuda()
    else:
        raise Exception('Not supported bn type: {}'.format(args.bn_type))
        
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

    # Train
    start_epoch = 1
    global_step = 0
    lr = args.lr
    train_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = datetime.now()
        batch_start = datetime.now()
        for batch_idx, batch_data in enumerate(train_loader):
            train_loss = train(model, optimizer, batch_data, criteria, args)
            global_step += 1
            if (batch_idx + 1) % args.print_freq == 0:
                print(print_format.format(
                    'Train', epoch, batch_idx, len(train_loader),
                    train_loss, str(datetime.now() - batch_start), lr))
                sys.stdout.flush()
                batch_start = datetime.now()

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
    print(f'Train elapsed time => {str(datetime.now() - train_start)}')

if __name__ == '__main__':
    parser = parse_autoencoder_args()
    args = parser.parse_args()
    args.stride = 32
    main(args)
import os
import sys
import pickle
import random
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms

from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.models.ego_rnn import EgoRnn 
from sense.utils.arguments import parse_args
from sense.rigidity_refine.io_utils import read_camera_data
import sense.models.model_utils as model_utils
from sense.lib.nn import DataParallelWithCallback
from sense.datasets import kitti_vo, malaga, visual_odometry_dataset
import sense.datasets.flow_transforms as flow_transforms
from sense.models.dummy_scene import SceneNet


temp_save = None

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def make_data_helper(path):
    if args.dataset == "kitti_vo":
        train_sequences = [0, 2, 8, 9]
        train_data, test_data = kitti_vo.kitti_vo_data_helper(path, train_sequences)
    elif args.dataset == "malaga":
        train_sequences = [1, 4, 6, 7, 8, 10, 11]
        train_data, test_data = malaga.malaga_data_helper(path, train_sequences)
    elif args.dataset == "mixed":
        kitti_train_sequences = [0, 2, 8, 9]
        kitti_val_sequences = [3, 10]
        malaga_train_sequences = [1, 4, 6, 7, 8, 10, 11]
        malaga_val_sequences = [2, 9]
        kitti_train, kitti_test = kitti_vo.kitti_vo_data_helper(path, kitti_train_sequences, kitti_val_sequences)
        malaga_train, malaga_test = malaga.malaga_data_helper(path, malaga_train_sequences, malaga_val_sequences)
        train_data = kitti_train + malaga_train
        test_data = kitti_test + malaga_test
    else:
        raise ValueError("Invalid dataset!")
    return train_data, test_data
    
def data_cacher(path, args):
    cache_file_path = f'cache/{args.dataset}_pose.pkl'
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'rb') as f:
            cached_data = pickle.load(f)
            train_data = cached_data['train_data']
            test_data = cached_data['test_data']
    else:
        train_data, test_data = make_data_helper(path)
        with open(cache_file_path, 'wb') as f:
            pickle.dump(
                {
                    'train_data': train_data,
                    'test_data': test_data
                },
                f, pickle.HIGHEST_PROTOCOL)
    return train_data, test_data

def make_data_loader(path, of_model, enc, args):
    height_new = 384
    width_new = 1280
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Resize((height_new, width_new)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3)])
    flow_transform = torchvision.transforms.Compose([
        flow_transforms.NormalizeFlowOnly(mean=[0,0],std=[-400.0, 400.0]),
    ])
    final_transform = torchvision.transforms.Compose([
        flow_transforms.NormalizeFlowOnly(mean=[0,0],std=[-60.0, 60.0]),
    ])
 
    train_data, test_data = data_cacher(path, args)
    print("Train data sequence size: ", len(train_data))
    print("Test data sequence size: ", len(test_data))
    
    train_set = visual_odometry_dataset.VODataset(train_data, input_transform, 5)
    test_set = visual_odometry_dataset.VODataset(test_data, input_transform, 5)
    
    collate_fn = visual_odometry_dataset.PreprocessingCollateFn(of_model, enc, flow_transform, final_transform, args)
    
    return torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=False,
            # worker_init_fn = lambda _: np.random.seed(int(torch.initial_seed()%(2**32 -1))),
            # collate_fn=collate_fn
        ), torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=False,
            # collate_fn=collate_fn
        ), collate_fn

def train(model, optimizer, data, criteria):
    input, targets = data
    model.train()
    optimizer.zero_grad()
    pose = model(input)
    loss = criteria(pose, targets)
    loss.backward()
    optimizer.step()
    loss = loss.item()
    return loss

def validation(model, data, criteria):
    input, targets = data
    model.eval()
    with torch.no_grad():
        pose = model(input)
        loss = criteria(pose, targets)
    return loss.item()
        
def save_checkpoint(model, optimizer, epoch, global_step, args):
    #SAVE
    global temp_save
    now = datetime.now().strftime("%d-%m-%H-%M")
    if temp_save == None:
        temp_save = f"lstm_{now}"
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

def adjust_learning_rate(optimizer, epoch, lr, rate):
    if epoch % rate[0] == 0:
        lr *= 0.316
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

def main(args):
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)   
    np.random.seed(args.seed)  
    random.seed(args.seed)
    
    # Flow producer model (PSMNexT)
    holistic_scene_model_path = '/content/model_0068.pth'
    
    # EGO encoder model
    ego_model_path = '/content/model_0012.pth'
    
    # Data load
    train_loader, validation_loader, preprocess = make_data_loader(args.base_dir, holistic_scene_model_path, ego_model_path, args)
    # train_loader, validation_loader, disp_test_loader = tjss.make_data_loader(args)
    
    # Make model
    model = EgoRnn(30)
    print('Number of LSTM parameters: {}'.format(
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
    criteria = nn.MSELoss()
    start_epoch = 1
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
    global_step = 0
    lr = args.lr
    
    train_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.lr_steps)
        epoch_start = datetime.now()
        batch_start = datetime.now()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_data = preprocess(batch_data)
            train_loss = train(model, optimizer, batch_data, criteria)
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
            batch_data = preprocess(batch_data)
            val_loss += validation(model, batch_data, criteria)
        print(print_format.format(
            'Val', epoch, 0, len(validation_loader),
            val_loss /  len(validation_loader), str(datetime.now() - val_start), lr))
        
        print(f'Epoch {epoch} elapsed time => {str(datetime.now() - epoch_start)}')
        # Save model
        save_checkpoint(model, optimizer, epoch, global_step, args)
    print(f'Train elapsed time => {str(datetime.now() - train_start)}')


if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    args.stride = 32
    main(args)
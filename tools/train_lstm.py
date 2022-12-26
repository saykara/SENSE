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
from sense.models.unet import UNet
from sense.utils.arguments import parse_args
from sense.rigidity_refine.io_utils import read_camera_data
import sense.models.model_utils as model_utils
from sense.lib.nn import DataParallelWithCallback
from .demo import run_holistic_scene_model, run_warped_disparity_refinement
from sense.datasets import kitti_vo, malaga, visual_odometry_dataset


temp_save = None

def pre_process(sequence, holistic_scene_model, warp_disp_ref_model, encoder_model):
    result = []
    pose = None
    seq = []
    for i in range(5):
        if pose is None:
            pose = sequence[i][4]
        else:
            pose = sequence[i][4] - pose
        flow_raw, flow_occ, disp0, disp1_unwarped, seg = run_holistic_scene_model(
	    	sequence[i][0], sequence[i][1],
	    	sequence[i][2], sequence[i][3],
	    	holistic_scene_model
	    )
        flow_rigid, disp1_raw, disp1_rigid, disp1_nn = run_warped_disparity_refinement(
        	sequence[i][0],
        	flow_raw, flow_occ,
        	disp0, disp1_unwarped,
        	seg,
        	sequence[5],
        	warp_disp_ref_model
        )
        seq.append(encoder_model(flow_rigid)[4])
    result.append(seq)
    result.append(pose)
    return result.cuda()

def make_data_helper(path):
    if args.dataset == "kitti_vo":
        train_sequences = [0, 2, 8, 9]
        train_data, test_data = kitti_vo.kitti_vo_data_helper(path, train_sequences)
    elif args.dataset == "malaga":
        train_sequences = [1, 4, 6, 7, 8, 10, 11]
        train_data, test_data = malaga.malaga_data_helper(path, train_sequences)
    elif args.dataset == "mixed":
        kitti_train_sequences = [0, 2, 8, 9]
        malaga_train_sequences = [1, 4, 6, 7, 8, 10, 11]
        kitti_train, kitti_test = kitti_vo.kitti_vo_data_helper(path, kitti_train_sequences)
        malaga_train, malaga_test = malaga.malaga_data_helper(path, malaga_train_sequences)
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

def make_data_loader(dataset, args):
    height_new = 384
    width_new = 1280
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((height_new, width_new)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
 
    train_data, test_data = data_cacher(args)
    print("Train data sequence size: ", len(train_data))
    print("Test data sequence size: ", len(test_data))
    
    train_set = visual_odometry_dataset.VODataset(train_data, transform, 5)
    test_set = visual_odometry_dataset.VODataset(test_data, transform, 5)
    
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
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=True
        )

def train(model, optimizer, data, criteria, hn, cn, holistic_scene_model, warp_disp_ref_model, encoder_model):
    model.train()
    data = data.cuda()
    data = pre_process(data, holistic_scene_model, warp_disp_ref_model, encoder_model)
    optimizer.zero_grad()
    data_pred = model(data[0], hn, cn)
    loss = criteria(data_pred, data[1])
    loss.backward()
    optimizer.step()
    loss = loss.item()
    return loss, hn, cn

def validation(model, data, criteria, holistic_scene_model, warp_disp_ref_model, encoder_model):
    model.eval()
    data = data.cuda()
    data = pre_process(data, holistic_scene_model, warp_disp_ref_model, encoder_model)
    with torch.no_grad():
        data_pred = model(data[0])
        loss = criteria(data_pred, data[1])
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
    
def main(args):
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)   
    np.random.seed(args.seed)  
    random.seed(args.seed)
    
    # Flow producer model (PSMNexT)
    holistic_scene_model = model_utils.make_model(args, do_flow=True, do_disp=True, do_seg=True)
    holistic_scene_model_path = 'data/pretrained_models/kitti2012+kitti2015_new_lr_schedule_lr_disrupt+semi_loss_v3.pth'
    ckpt = torch.load(holistic_scene_model_path)
    state_dict = ckpt['state_dict']
    holistic_scene_model.load_state_dict(state_dict)
    holistic_scene_model.eval()
    
    # Flow producer model
    warp_disp_ref_model = UNet()
    warp_disp_ref_model = nn.DataParallel(warp_disp_ref_model).cuda()
    warp_disp_ref_model_path = 'data/pretrained_models/kitti2015_warp_disp_refine_1500.pth'
    ckpt = torch.load(warp_disp_ref_model_path)
    state_dict = ckpt['state_dict']
    warp_disp_ref_model.load_state_dict(state_dict)
    warp_disp_ref_model.eval()
 
    # EGO encoder model
    ego_model = EGOAutoEncoder("syncbn", "gelu")
    ego_model_path = '<autoencoder path>'
    ckpt = torch.load(ego_model_path)
    state_dict = ckpt['state_dict']
    ego_model.load_state_dict(state_dict)
    encoder_model = ego_model.encoder
    encoder_model.eval()
    
    # Data load
    dataset = "E:/Thesis/content/flow_dataset"
    train_loader, validation_loader = make_data_loader(dataset, args)
    # train_loader, validation_loader, disp_test_loader = tjss.make_data_loader(args)
    
    # Make model
    model = EgoRnn(1024)
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
    
    hn, cn = model.init_states(args.batch_size)
    train_start = datetime.now()
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = datetime.now()
        for batch_idx, batch_data in enumerate(train_loader):
            batch_start = datetime.now()
            train_loss, hn, cn = train(model, optimizer, batch_data, criteria, hn, cn, 
                                       holistic_scene_model, warp_disp_ref_model, encoder_model)
            hn = hn.detach()
            cn = cn.detach()
            global_step += 1
            if (batch_idx + 1) % args.print_freq == 0:
                print(print_format.format(
                    'Train', epoch, batch_idx, len(train_loader),
                    train_loss, str(datetime.now() - batch_start), lr))
                sys.stdout.flush()

        val_start = datetime.now()
        val_loss = 0
        hn, cn = model.init_states(args.batch_size)
        for batch_idx, batch_data in enumerate(validation_loader):
            temp_val, hn, cn = validation(model, batch_data, criteria, hn, cn, 
                                          holistic_scene_model, warp_disp_ref_model, encoder_model)
            val_loss += temp_val
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
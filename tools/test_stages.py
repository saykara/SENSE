
import os
import pickle
import torchvision
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2
import time

import torch.nn.functional as F
from torchvision import transforms
from sense.utils.arguments import parse_args
from train_joint_synthetic_sceneflow import test_disp, test_flow
import sense.models.model_utils as model_utils
from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.lib.nn import DataParallelWithCallback
import sense.datasets.flow_transforms as flow_transforms
from sense.datasets import visual_odometry_dataset
from sense.datasets.ego_autoencoder_dataset import EGOAutoencoderImageDataset, PreprocessingCollateFn
from sense.models.ego_rnn import EgoRnn 

from sense.datasets.dataset_catlog import make_flow_disp_data
from sense.datasets.kitti_vo import kitti_vo_test_data_helper
from sense.datasets.test_dataset import Stage1TestDataset
from sense.datasets.dataset_utils import optical_flow_loader, sceneflow_disp_loader
from sense.datasets.flyingthings3d import make_flow_disp_data_simple_merge
from train_ego_autoencoder import make_flow_data_helper

def test_helper(stage, args):
    test_list = []
    if stage == 1:
        inputs, targets = make_flow_disp_data_simple_merge(args.base_dir, "val")
        test_list = (inputs, targets)
        #   [cur_im_path, nxt_im_path, cur_im_path, right_im_path], 
        #   [flow_path, flow_occ_path, disp_path, disp_occ_path]
    elif stage == 2:
        temp_list, _ = make_flow_disp_data_simple_merge(args.base_dir, "val")
        for i in temp_list:
            test_list.append([i[0], i[1]])
            #   [cur_im_path, nxt_im_path}
    elif stage == 3:
        test_sequences = [1]
        # 1, 7 will be test
        test_list = kitti_vo_test_data_helper(args.base_dir, test_sequences, 100)
    return test_list

####### STAGE 1 #######
def stage1_data_loader(args):
    test_data = test_helper(1, args)
    tmp_test_data = []
    for i in range(len(test_data[0])):
        tmp_test_data.append([test_data[0][i], test_data[1][i]])
    test_data = tmp_test_data
    crop_transform = flow_transforms.Compose([
		flow_transforms.CenterCrop((512, 960))
	])
    transform = transforms.Compose([
		flow_transforms.ArrayToTensor()
	])
    test_set = Stage1TestDataset(
        '',
		test_data,
		flow_loader=optical_flow_loader,
		disp_loader=sceneflow_disp_loader,
        crop_transform=crop_transform,
        transform=transform
	)
    return torch.utils.data.DataLoader(
		test_set,
		batch_size=1,
		shuffle=False,
		num_workers=0, 
		drop_last=False,
		pin_memory=True,
	)
    
    
def test_stage1(flow_prs, disp_prs, flow_gts, disp_gts, crit):
    flow = flow_prs[0][0].detach().cpu()
    flow_occ = flow_prs[1]
    disp = disp_prs[0][0].detach().cpu()
    disp_occ = disp_prs[1]
    flow_gt, flow_occ_gt = flow_gts
    flow_gt = flow_gt
    flow_occ_gt = flow_occ_gt.cuda()
    disp_gt, disp_occ_gt, _ = disp_gts
    disp_gt = disp_gt
    disp_occ_gt = disp_occ_gt.cuda()
    flow_occ_cr, disp_occ_cr = crit
    
    disp_mask = (disp_gt < 192)
    disp_mask.detach_()
    
    threshold = 3.0
    ### FLOW
    # Mean Square Error (MSE)
    mse = torch.mean((flow_gt - flow) ** 2)
    
    # End-point Error (EPE)
    epe = torch.mean(torch.sqrt(torch.sum(torch.square(flow_gt - flow), dim=1)))
    
    # Angular Error (AE)
    dot_product = torch.sum(flow_gt * flow, dim=0)
    norm_product = torch.norm(flow_gt, dim=0) * torch.norm(flow_gt, dim=0)
    cosine_similarity = torch.clamp(dot_product / norm_product, -1.0, 1.0)
    ae = torch.nanmean((torch.acos(cosine_similarity.float()) * (180.0 / 3.14159265359)))
    ###FLOW OCCLUSION
    # flow_occ_loss, _ = flow_occ_cr(flow_occ_gt, flow_occ[0])
    print(flow_occ[1])
    print(flow_occ_gt)
    time.sleep(60)
    flow_occ_loss = torch.mean((flow_occ[0] == flow_occ_gt).float())
    
    ### DISPARITY
    disp_error = torch.abs(disp_gt[disp_mask] - disp[disp_mask])
    # Mean Absolute Error (MAE)
    mae = torch.mean(disp_error)
    
    # Percentage of Bad Pixels (PBP)
    pbp = 100.0 * torch.sum(disp_error > threshold).float() / disp_error.numel()
    
    # 3 Pixel Error (3PE)
    err3 = 100.0 * torch.mean((disp_error > threshold).float())
    
    ###DISPARITY OCCLUSION
    # disp_occ_loss, _ = disp_occ_cr(disp_occ_gt, disp_occ)
    disp_occ_loss = None
    
    return mse, epe, ae, mae, pbp, err3, flow_occ_loss, disp_occ_loss

	    
def stage1(args):
    test_loader = stage1_data_loader(args)

    model = model_utils.make_model(
        args, 
        do_flow=True,
        do_disp=True,
        do_seg=False
    )
 
    ckpt = torch.load("E:/Thesis/Models/PSMNeXT_small/model_0068.pth")
    state_dict = ckpt['state_dict']
    model.load_state_dict(model_utils.patch_model_state_dict(state_dict))
    print(f'==> {args.enc_arch} model has been loaded.')
    model.eval()
    
    test_print_format = '{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'
    
    (flow_crit, flow_occ_crit), flow_down_scales, flow_weights = model_utils.make_flow_criteria(args)
    (disp_crit, disp_occ_crit), disp_down_scales, disp_weights = model_utils.make_disp_criteria(args)
    crit = (flow_occ_crit, disp_occ_crit)
    
    avg_mse = 0
    avg_ae = 0
    avg_epe = 0
    avg_mae = 0
    avg_pbp = 0
    avg_3pe = 0
    avg_foc = 0
    avg_doc = 0
    start_time = datetime.now()
    i = 0
    for idx, data in enumerate(test_loader):
        with torch.no_grad():
            flow_pred, disp_pred, _ = model(data[0][0].cuda(), data[0][1].cuda(), data[2][1].cuda())
        
        # flow_gt = data[1].permute(0, 3, 1, 2).numpy()
        # disp_gt = data[3].permute(0, 3, 1, 2).numpy()
        
        mse, epe, ae, mae, pbp, err3, flow_occ, disp_occ = test_stage1(flow_pred, disp_pred, data[1], data[3], crit)
        avg_mse += mse
        avg_epe += epe
        avg_ae += ae
        avg_mae += mae
        avg_pbp += pbp
        avg_3pe += err3
        avg_foc += flow_occ
        i += 1
        if i == 100:
            break
        #avg_foc += flow_occ
        #avg_doc += disp_occ
    print(f"Time elapsed: {datetime.now() - start_time}")
    print("avg_mse:", avg_mse.item())
    print("avg_epe:", avg_epe.item())
    print("avg_ae:", avg_ae.item())
    print("avg_mae:", avg_mae.item())
    print("avg_pbp:", avg_pbp.item())
    print("avg_3pe:", avg_3pe.item())
    print("avg_foc:", avg_foc.item())
    
    print(test_print_format.format(
        'Test',
        avg_mse.item() / 100,
        avg_epe.item() / 100  * args.div_flow, 
        avg_ae.item()  / 100, 
        avg_mae.item() / 100,
        avg_pbp.item() / 100,
        avg_3pe.item() / 100,
        avg_foc / 100,
        avg_doc / len(test_loader)
    ))


####### STAGE 2 #######
def stage2_data_loader(model, args):
    test_data = test_helper(2, args)
    
    transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Resize((512, 960))])

    flow_transform = transforms.Compose([
        flow_transforms.NormalizeFlowOnly(mean=[0,0],std=[-1000.0, 1000.0])])
    
    test_set = EGOAutoencoderImageDataset(root='', path_list=test_data, transform=transform)
    
    return torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=False
        ), PreprocessingCollateFn(model, flow_transform, args)
    
    
def test_stage2(flow_autoencoder, original):
    with torch.no_grad():
        original = original.unsqueeze(0)
        regenerated = flow_autoencoder(original)
    # Mean Square Error (MSE)
    mse = torch.mean((original - regenerated) ** 2)
    
    # End-point Error (EPE)
    epe = torch.mean(torch.sqrt(torch.sum(torch.square(original - regenerated), dim=1)))
    
    # Angular Error (AE)
    dot_product = torch.sum(original * regenerated, dim=0)
    norm_product = torch.norm(original, dim=0) * torch.norm(original, dim=0)
    cosine_similarity = torch.clamp(dot_product / norm_product, -1.0, 1.0)
    ae = torch.nanmean((torch.acos(cosine_similarity.float()) * (180.0 / 3.14159265359)))
    return mse, epe, ae


def stage2(args):
    # Flow estimator model
    flow_model = model_utils.make_model(
        args, 
        do_flow=not args.no_flow,
        do_disp=not args.no_disp,
        do_seg=(args.do_seg or args.do_seg_distill)
    )
    
    # Data loader
    flow_model = "E:/Thesis/Models/SENSE/model_0068.pth"
    test_loader, preprocess = stage2_data_loader(flow_model, args)
    
    # Flow encoder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    flow_autoencoder = DataParallelWithCallback(EGOAutoEncoder(args)).cuda()
    ckpt = torch.load("E:/Thesis/Models/Autoencoder/old/model_0009.pth")
    state_dict = ckpt['state_dict']
    flow_autoencoder.load_state_dict(state_dict)
    flow_autoencoder = flow_autoencoder.to(device)
    flow_autoencoder.eval()
    
    print(f'==> {args.enc_arch} flow encoder has been loaded.')
    
    # Print format
    print_format = '{}\t{:d}\t{:.6f}\t{:.6f}\t{:.6f}\t{}'

    test_start = datetime.now()
    mse_loss = 0
    epe_loss = 0
    ae_loss = 0
    for batch_idx, batch_data in enumerate(test_loader):
        batch_data = preprocess(batch_data)
        mse, epe, ae = test_stage2(flow_autoencoder, batch_data)
        mse_loss += mse
        epe_loss += epe
        ae_loss += ae
    print(print_format.format(
        'Val', len(test_loader), mse_loss / len(test_loader),  epe_loss / len(test_loader), 
        ae_loss /  len(test_loader), str(datetime.now() - test_start)))

###############
### STAGE 3 ###
###############
def stage3_data_loader(path, of_model, enc, args):
    height_new = 384
    width_new = 1280
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Resize((height_new, width_new)),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomVerticalFlip(0.3)])
    flow_transform = torchvision.transforms.Compose([
        flow_transforms.NormalizeFlowOnly(mean=[0,0],std=[-800.0, 800.0]),
    ])
    final_transform = torchvision.transforms.Compose([
        flow_transforms.NormalizeFlowOnly(mean=[0,0],std=[-60.0, 60.0]),
    ])
    test_data = test_helper(3, args)
    print("Test data sequence size: ", len(test_data))
    
    test_set = visual_odometry_dataset.VODataset(test_data, input_transform, 100)
    
    collate_fn = visual_odometry_dataset.PreprocessingCollateFn(of_model, enc, flow_transform, final_transform, args)
    
    return torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=True,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=False,
        ), collate_fn

def test_stage3(prediction, true):
    # Positional Error
    position_err = np.linalg.norm(true[:3] - prediction[:3])
    
    # Rotational Error
    trace = np.trace(np.dot(true[3:], prediction[3:].T))
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # ensure value is within the valid range for arccos
    angle_error = np.arccos(cos_angle)
    rotation_err = np.degrees(angle_error)
    return position_err, rotation_err

def stage3(args):
    # Flow producer model (PSMNexT)
    holistic_scene_model_path = "E:/Thesis/Models/SENSE/model_0068.pth"
    
    # EGO encoder model
    ego_model_path = "E:/Thesis/Models/Autoencoder/old/model_0009.pth"
    
    # Data load
    test_loader, preprocess = stage3_data_loader(args.base_dir, holistic_scene_model_path, ego_model_path, args)
    
    # Make model
    model = EgoRnn(30720)
    model = model.cuda()
    model.eval()
    
    if args.loadmodel is not None:
        ckpt = torch.load("E:/Thesis/Models/LSTM/old/model_0006.pth")
        state_dict = ckpt['state_dict']
        model.load_state_dict(state_dict)
        print(f'==> {args.enc_arch} pose model has been loaded.')
    
    # Print format
    print_format = '{}\t{:d}\t{:.6f}\t{:.6f}\t{}'
    
    test_start = datetime.now()
    total_pos_err = 0.
    total_rot_err = 0.
    for batch_idx, batch_data in enumerate(test_loader):
        batch_data = preprocess(batch_data)
        input, targets = batch_data
        with torch.no_grad():
            pose = model(input)
            pos_err, rot_err = test_stage3(pose, targets)
        total_pos_err += pos_err
        total_rot_err += rot_err
    print(print_format.format(
        'Val', len(test_loader), total_pos_err /  len(test_loader), 
        total_rot_err / len(test_loader), str(datetime.now() - test_start)))
        

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    args.stride = 32
    if args.test_stage == "stage1":
        stage1(args)
    elif args.test_stage == "stage2":
        stage2(args)
    elif args.test_stage == "stage3":
        stage3(args)
    else:
        raise Exception
    
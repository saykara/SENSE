
import os
import pickle
import torchvision
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import cv2
import time
import random
import math

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
from sense.datasets.kitti_vo import kitti_vo_test_data_helper, kitti_vo_data_helper
from sense.datasets.test_dataset import Stage1TestDataset
from sense.datasets.dataset_utils import optical_flow_loader, sceneflow_disp_loader
from sense.datasets.flyingthings3d import make_flow_disp_data_simple_merge
from train_ego_autoencoder import make_flow_data_helper

total_pos_err = 0.
total_rot_err = 0.
total_pos_err_pct = 0.
error_pose = 0

def test_helper(stage, args, seq):
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
        test_sequences = [seq[2]]
        # 1, 7 will be test
        test_list = kitti_vo_test_data_helper(args.base_dir, test_sequences, seq[0], seq[1])
        test_list, _ = kitti_vo_data_helper(args.base_dir, test_sequences, [])
    return test_list

####### STAGE 1 #######
def stage1_data_loader(args):
    test_data = test_helper(1, args, None)
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
    test_data = test_helper(2, args, None)
    
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
def stage3_data_loader(path, of_model, enc, begin, seq_l, s, args):
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
    test_data = test_helper(3, args, (begin, seq_l, s))
    print("Test data sequence size: ", len(test_data))
    
    test_set = visual_odometry_dataset.VODataset(test_data, input_transform, seq_l)
    
    collate_fn = visual_odometry_dataset.PreprocessingCollateFn(of_model, enc, flow_transform, final_transform, seq_l, args)
    
    return torch.utils.data.DataLoader(
            test_set,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            drop_last=True,
            pin_memory=False,
        ), collate_fn

def test_stage3(prediction, true):
    prediction = prediction.cpu().numpy().squeeze()
    true = true.cpu().numpy().squeeze()
    
    # Positional Error
    pos_err = np.linalg.norm(true[:3] - prediction[:3])
    print(true[:3])
    pos_err_pct = (np.linalg.norm(prediction[:3] - true[:3]) / np.linalg.norm(true[:3])) * 100
    mse =  np.mean((true[:3] - prediction[:3]) ** 2)
    # Rotational Error
    # define the order of rotations
    order = "ZYX"

    # compute the rotation matrices from the Euler angles
    def euler_angles_to_matrix(euler_angles, order):
        # compute the sin and cos values of the angles
        s1 = torch.sin(euler_angles[0])
        c1 = torch.cos(euler_angles[0])
        s2 = torch.sin(euler_angles[1])
        c2 = torch.cos(euler_angles[1])
        s3 = torch.sin(euler_angles[2])
        c3 = torch.cos(euler_angles[2])

        # compute the rotation matrix
        if order == "ZYX":
            r = torch.tensor([[c1*c2, c1*s2*s3 - c3*s1, s1*s3 + c1*c3*s2],
                              [c2*s1, c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3],
                              [-s2,   c2*s3,            c2*c3           ]])
        elif order == "XYZ":
            r = torch.tensor([[c2*c3, -c2*s3, s2 ],
                              [c1*s3 + c3*s1*s2, c1*c3 - s1*s2*s3, -c2*s1],
                              [s1*s3 - c1*c3*s2, c3*s1 + c1*s2*s3, c1*c2]])
        return r

    gt_R = euler_angles_to_matrix(torch.from_numpy(true[3:]), order)
    pred_R = euler_angles_to_matrix(torch.from_numpy(prediction[3:]), order)

    # compute the rotational error
    def rotation_error(R1, R2):
        eps = 1e-8
        trace = torch.sum(R1 * R2)
        trace = torch.clamp(trace, -1.0 + eps, 3.0 - eps)
        angle = torch.acos((trace - 1.0) / 2.0)
        return angle

    rotation_err = rotation_error(gt_R, pred_R)
    return pos_err, pos_err_pct, rotation_err.item()

def stage3(args, begin, seq_l, s):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    global total_pos_err
    global total_rot_err
    global error_pose
    global total_pos_err_pct
    # Flow producer model (PSMNexT)
    holistic_scene_model_path = "/content/dataset/models/SENSE/model_0068.pth"
    
    # EGO encoder model
    ego_model_path = "/content/dataset/models/encoder/old/model_0009.pth"
    
     
    # Data load
    test_loader, preprocess = stage3_data_loader(args.base_dir, holistic_scene_model_path, ego_model_path, begin, seq_l, s, args)
    
    # Make model
    model = EgoRnn(30720)

    ckpt = torch.load("/content/dataset/models/lstm/old/model_0035.pth")
    state_dict = ckpt['state_dict']
    model.load_state_dict(state_dict)
    print(f'==> {args.enc_arch} pose model has been loaded.')
    model = model.cuda()
    model.eval()
    
    # Print format
    print_format = '{}\t{:d}\t{:.6f}\t{:.6f}\t{:.6f}\t{}'
    
    test_start = datetime.now()
    total_time = 0
    pose_change = torch.empty(1, 6).cuda()
    
    for batch_idx, batch_data in enumerate(test_loader):
        batch_data = preprocess(batch_data)
        input, pose_gt = batch_data
        with torch.no_grad():
            input = input.cuda()
            pose_change = model(input)
            pos_err, pos_err_pct, rot_err = test_stage3(pose_change, pose_gt)
            if pos_err_pct > 1000 or pos_err == math.nan:
                error_pose += 1
            else:
                total_pos_err += pos_err
                total_pos_err_pct += pos_err_pct
            total_rot_err += rot_err
    print(print_format.format(
        'Val', len(test_loader), total_pos_err, total_pos_err_pct, 
        total_rot_err, str(datetime.now() - test_start)))
    #print(f'Test size = {seq_l}')
    #print(f'LSTM elapsed time = {total_time}')
    #print(f'LSTM elapsed in ms = {total_time * 1000}')
    #print(f'Average LSTM time = {total_time / seq_l}')
    #print(f'Average LSTM in ms = {(total_time * 1000) / seq_l}')
    #preprocess.print_time()
        

if __name__ == '__main__':
    parser = parse_args()
    args = parser.parse_args()
    args.stride = 32
    
    if args.test_stage == "stage1":
        stage1(args)
    elif args.test_stage == "stage2":
        stage2(args)
    elif args.test_stage == "stage3":
        seq_list = range(0, 1050, 75)
        k = 2192
        for s in [1, 7]:
            stage3(args, 0, 5, s)
        print_format = '{}\t{:d}\t{:.6f}\t{:.6f}\t{:.6f}'
        print(print_format.format(
        'Test', k, total_pos_err / (k - error_pose), total_pos_err_pct / (k - error_pose), total_rot_err / k))
    else:
        raise Exception
    
from sense.models.ego_autoencoder import EGOAutoEncoder
from sense.datasets.ego_autoencoder_dataset import EGOFlowDataset
from sense.utils.arguments import parse_args

import torch
import torch.nn as nn
import torch.optim as optim

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

def make_flow_data_helper(base_dir, split):
	data_list = []
	# flow_dir = os.path.join(base_dir, split, 'flow')
	flow_dir = base_dir + "/" + split + "/" + 'flow'
	for lr in ['left', 'right']:
		for fb in ['into_future', 'into_past']:
			for it in os.listdir(os.path.join(flow_dir, lr, fb)):
				# data_list.append(os.path.join(flow_dir, lr, fb, it))
				data_list.append(flow_dir + "/" + lr + "/" + fb + "/" + it)
	return data_list

def data_cacher(path):
	cache_file_path = 'cache/ego_flow.pkl'
	if os.path.exists(cache_file_path):
		with open(cache_file_path, 'rb') as f:
			cached_data = pickle.load(f)
			train_data = cached_data['train_data']
			test_data = cached_data['test_data']
	else:
		# data_list = os.listdir(path)
		train_data = make_flow_data_helper(path, "train")
		test_data  = make_flow_data_helper(path, "val")
		# train_data, test_data = np.split(data_list, [int(len(data_list)*0.8)])
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
		
def save_checkpoint(model, optimizer, epoch, global_step, args):
	#SAVE
	now = datetime.now().strftime("%d-%m-%H-%M")
	save_dir = f"ego_autoencoder_{now}"
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
	
	# Data load
	dataset = "E:/Thesis/SceneFlow/FlyingThings3D_subset"
	# train_loader, validation_loader = make_data_loader(dataset, args)
	train_loader, validation_loader, disp_test_loader = tjss.make_data_loader(args)
	
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
			cur_im, nxt_im = batch_data[0] 
			left_im, right_im = batch_data[2]
			batch_start = datetime.now()
			flow_data, _, _, _, _ = demo.run_holistic_scene_model(cur_im, nxt_im, 
															left_im, right_im, 
															holistic_scene_model)
			imageio.imsave('data/results/ego_flow.png', kitti_viz.flow_to_color(flow_data.cpu().numpy().transpose(1, 2, 0)))
			train_loss = train(model, optimizer, flow_data, criteria, args)
			global_step += 1
			if (batch_idx + 1) % args.print_freq == 0:
				print(print_format.format(
					'Train', epoch, batch_idx, len(train_loader),
					train_loss, str(datetime.now() - batch_start), lr))
				sys.stdout.flush()

		val_start = datetime.now()
		val_loss = 0
		for batch_idx, batch_data in enumerate(validation_loader):
			cur_im, nxt_im = batch_data[0] 
			left_im, right_im = batch_data[2]
			flow_data, _, _, _, _ = demo.run_holistic_scene_model(cur_im, nxt_im, 
															left_im, right_im, 
															holistic_scene_model)
			val_loss += validation(model, flow_data, criteria)
		print(print_format.format(
			'Val', epoch, 0, len(validation_loader),
			val_loss /  len(validation_loader), str(datetime.now() - val_start), lr))
			
		# Save model
		print(f'Epoch {epoch} elapsed time => {str(datetime.now() - epoch_start)}')
		save_checkpoint(model, optimizer, epoch, global_step, args)
	print(f'Train elapsed time => {str(datetime.now() - train_start)}')
	
	
	im = cv2.imread("C:/Users/Utkua/OneDrive/SENSE/SENSE/data/results", 0)
	
	zeros = np.zeros((384, 1280))
	zeros[:im.shape[0], :im.shape[1]] = im
	zeros = zeros.astype(np.float32) / 255.0
	
	model.eval()
	zeros = zeros.cuda()
	
	with torch.no_grad():
		data_pred = model(zeros)
		imageio.imsave('data/results/ego_flow.png', kitti_viz.flow_to_color(data_pred.cpu().numpy().transpose(1, 2, 0)))


if __name__ == '__main__':
	parser = parse_args()
	args = parser.parse_args()
	args.stride = 32
	main(args)
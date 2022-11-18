"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import os
import glob
import numpy as np

def make_flow_dataset(kitti_dir, split_id=-1):
	if split_id > 0:
		val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
		assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
		val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
		trn_idxes = [idx for idx in range(200) if idx not in val_idxes]
	else:
		trn_idxes = [idx for idx in range(194)]
		val_idxes = []

	im_dir = os.path.join(kitti_dir, 'training', 'colored_0')
	flow_dir = os.path.join(kitti_dir, 'training', 'flow_occ')

	def make_file_list(idxes, allow_no_flow_gt=False):
		paths = []
		for idx in idxes:
			cur_im_name = '{:06d}_10.png'.format(idx)
			nxt_im_name = '{:06d}_11.png'.format(idx)
			cur_im_path = os.path.join(im_dir, cur_im_name)
			nxt_im_path = os.path.join(im_dir, nxt_im_name)
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			flow_path = os.path.join(flow_dir, cur_im_name)
			if not allow_no_flow_gt:
				assert os.path.exists(flow_path), flow_path
			paths.append([
				[cur_im_path, nxt_im_path], [flow_path, None]
			])
		return paths

	train_data = make_file_list(trn_idxes)

	if split_id > 0:
		test_data = make_file_list(val_idxes)
	else:
		im_dir = os.path.join(kitti_dir, 'testing', 'colored_0')
		flow_dir = os.path.join(kitti_dir, 'testing', 'flow_occ')
		test_idxes = [idx for idx in range(195)]
		test_data = make_file_list(test_idxes, allow_no_flow_gt=True)

	return train_data, test_data

def make_disparity_dataset(kitti_dir, split_id=-1):
	left_dir  = os.path.join(kitti_dir, 'training/colored_0')
	right_dir = os.path.join(kitti_dir, 'training/colored_1')
	disp_dir = os.path.join(kitti_dir, 'training/disp_occ')

	if split_id > 0:
		val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
		assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
		val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
		val = ['%06d_10.png' % idx for idx in val_idxes]
		train = ['%06d_10.png' % idx for idx in range(200) if idx not in val_idxes]
	else:
		train = ['%06d_10.png' % idx for idx in range(194)]
		val = []

	def make_file_list(im_names, allow_no_disp_gt=False):
		left_im_paths = [os.path.join(left_dir, n) for n in im_names]
		right_im_paths = [os.path.join(right_dir, n) for n in im_names]
		disp_paths = [os.path.join(disp_dir, n) for n in im_names]
		paths = []
		for i in range(len(left_im_paths)):
			assert os.path.exists(left_im_paths[i]), left_im_paths[i]
			assert os.path.exists(right_im_paths[i]), right_im_paths[i]
			if not allow_no_disp_gt:
				assert os.path.exists(disp_paths[i]), disp_paths[i]
			paths.append([
				[left_im_paths[i], right_im_paths[i]], [disp_paths[i], None]
			])
		return paths

	train_data = make_file_list(train)
	if split_id > 0:
		test_data = make_file_list(val)
	else:
		left_dir  = os.path.join(kitti_dir, 'testing/colored_0')
		right_dir = os.path.join(kitti_dir, 'testing/colored_1')
		disp_dir = os.path.join(kitti_dir, 'testing/disp_occ')
		test = ['%06d_10.png' % idx for idx in range(195)]
		test_data = make_file_list(test, allow_no_disp_gt=True)
	return train_data, test_data

def make_flow_disp_dataset(kitti_dir, split_id=1, pseudo_gt_dir=None):
	left_dir  = os.path.join(kitti_dir, 'training/colored_0')
	right_dir = os.path.join(kitti_dir, 'training/colored_1')
	disp_dir = os.path.join(kitti_dir, 'training/disp_occ')
	flow_dir = os.path.join(kitti_dir, 'training/flow_occ')

	# val_idxes_file = os.path.join(kitti_dir, 'val_idxes_split{}.txt'.format(split_id))
	# assert os.path.exists(val_idxes_file), 'Val indexes file not found {}'.format(val_idxes_file)
	# val_idxes = np.loadtxt(val_idxes_file, delimiter=',').astype(int).tolist()
	# trn_idxes = [idx for idx in range(200) if idx not in val_idxes]
	trn_idxes = [idx for idx in range(194)]
	val_idxes = []

	def make_file_list(idxes):
		paths = []
		for idx in idxes:
			cur_im_name = '{:06d}_10.png'.format(idx)
			nxt_im_name = '{:06d}_11.png'.format(idx)
			cur_im_path = os.path.join(left_dir, cur_im_name)
			nxt_im_path = os.path.join(left_dir, nxt_im_name)
			right_im_path = os.path.join(right_dir, cur_im_name)
			assert os.path.exists(cur_im_path), cur_im_path
			assert os.path.exists(nxt_im_path), nxt_im_path
			assert os.path.exists(right_im_path), right_im_path
			flow_path = os.path.join(flow_dir, cur_im_name)
			assert os.path.exists(flow_path), flow_path
			disp_path = os.path.join(disp_dir, cur_im_name)
			assert os.path.exists(disp_path), disp_path

			# occlusion mask path generated by a pre-trained model
			if pseudo_gt_dir is not None:
				flow_occ_path = os.path.join(pseudo_gt_dir, 'flow_occ', 'training/colored_0', cur_im_name)
				assert os.path.exists(flow_occ_path), flow_occ_path
				disp_occ_path = os.path.join(pseudo_gt_dir, 'disp_occ', 'training/colored_0', cur_im_name)
				assert os.path.exists(disp_occ_path), disp_occ_path
			else:
				flow_occ_path = None
				disp_occ_path = None
			paths.append([
				[cur_im_path, nxt_im_path, cur_im_path, right_im_path], 
				[flow_path, flow_occ_path, disp_path, disp_occ_path]
			])
		return paths

	train_data = make_file_list(trn_idxes)
	test_data = make_file_list(val_idxes)

	return train_data, test_data

if __name__ == '__main__':
	kitti2012_dir = os.path.join('/content/bucket-data', 'KITTI_Stereo_2012')
	train_data, test_data = make_flow_disp_dataset(kitti2012_dir)
	print(len(train_data), len(test_data))
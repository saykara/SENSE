from ..sense.models.ego_autoencoder import EGOAutoEncoder

import torch
import torch.nn as nn
import torch.optim as optim

import sys
import math
from datetime import datetime


def RMSLELoss(input, output):
	loss = 0.
	for i in range(input.len()):
		loss = loss + math.pow(math.log(output[i] + 1) - math.log(input[i] + 1), 2)
	loss = loss / input.len()
	return math.sqrt(loss) 

def make_data_loader():
    pass

def train(model, optimizer, data, criteria, args):
	pass 

def validation(model, optimizer, data, criteria, args):
	pass

def save_checkpoint(model, optimizer, epoch, global_step, args):
	pass

def main(args):
	# TODO Data load
	train_loader, validation_loader = make_data_loader()
	
	# Make model
	model = EGOAutoEncoder(args.bn_type, args.act_type)
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
	criteria = RMSLELoss
	
	# Save & Load model
	
	# Print format
	print_format = '{}\t{:d}\t{:d}\t{:d}\t{:.3f}\t{}\t{:.6f}'
	
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
			val_loss += validation(model, optimizer, batch_data, criteria, args)
		print(print_format.format(
			'Val', epoch, 0, len(validation_loader),
			val_loss /  len(validation_loader), str(datetime.now() - val_start), lr))
		# Save model
		print(f'Epoch {epoch} elapsed time => {str(datetime.now() - epoch_start)}')
	print(f'Train elapsed time => {str(datetime.now() - train_start)}')
 

if __name__ == '__main__':
	args = None
	args.bn_type = 'syncbn'
	args.act_type = "gelu"
	main(args)
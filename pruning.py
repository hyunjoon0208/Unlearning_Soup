import argparse
import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

import arg_parser
from pruner import *
from trainer import train, validate
from pruner import *
from utils import *


best_acc = 0
best_mia = 0

def main():
	global args, best_acc
	args = arg_parser.parse_args()
	for arg in vars(args):
		print('{}: {}'.format(arg, getattr(args, arg)))
	
	torch.cuda.set_device(args.gpu)
	os.makedirs(args.save_dir, exist_ok=True)
	if args.seed:
		setup_seed(args.seed)
	
	if args.dataset != 'imagenet':
		model, train_loader, val_loader, test_loader, marked_loader = setup_model_dataset(args)
	else:
		model, train_loader, val_loader, test_loader, marked_loader = setup_model_dataset(args)

	model.cuda()

	criterion = nn.CrossEntropyLoss().cuda()
	decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

	try:
		if args.prune_type == 'pt':
			print('Pruning type: pt')
			initialization = None
		elif args.prune_type == 'lt':
			print('Pruning type: lt')
			initialization = deepcopy(model.state_dict())
		elif args.prune_type == 'rewind_lt':
			print('Pruning type: rewind_lt')
			initialization = None

	except:
		raise ValueError('Invalid pruning type')


	if args.imagenet_arch:
		lambda0 = (
			lambda cur_iter: (cur_iter + 1) / args.warmup
			if cur_iter < args.warmup
			else (
				0.5
				* (
					1.0
					+ np.cos(
						np.pi * ((cur_iter - args.warmup) / (args.epochs - args.warmup))
					)
				)
			)
		)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
	else:
		scheduler = torch.optim.lr_scheduler.MultiStepLR(
			optimizer, milestones=decreasing_lr, gamma=0.1
		)  # 0.1 is fixed

	if args.resume:
		print('Resuming from checkpoint {}'.format(args.checkpoint))

		checkpoint = torch.load(args.checkpoint, map_location='cuda:{}'.format(args.gpu))
		best_acc = checkpoint['best_acc']
		print('Best accuracy: {}'.format(best_acc))
		start_epoch = checkpoint['epoch']
		all_results = checkpoint['results']
		start_state = checkpoint['state']
		if start_state:
			current_mask = extract_mask(checkpoint['state_dict'])
			prune_model_custom(model, current_mask)
			check_sparsity(model)
			
		model.load_state_dict(checkpoint['state_dict'],strict=False)
		x_rand = torch.rand(1,3,args.image_size,args.image_size).cuda()
		model.eval()
		with torch.no_grad():
			model(x_rand)

		optimizer.load_state_dict(checkpoint['optimizer'])
		scheduler.load_state_dict(checkpoint['scheduler'])
		initialization = checkpoint['initialization']
	
	else:
		all_results = {}
		all_results['train_acc'] = []
		all_results['test_acc'] = []
		all_results['val_acc'] = []

		start_epoch = 0
		start_state = 0

	for state in range(start_state,args.pruning_times):
		check_sparsity(model)
		for epoch in range(start_epoch, args.epochs):
			start_time = time.time()
			print(optimizer.state_dict()['param_groups'][0]['lr'])
			train_acc = train(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, epoch=epoch, args=args)
			if args.prune_type == 'rewind_lt':
				rewind_state = deepcopy(model.state_dict())

			val_acc = validate(model = model, val_loader = val_loader, criterion = criterion, args = args)
			all_results['train_acc'].append(train_acc)
			all_results['val_acc'].append(val_acc)
			is_best = train_acc > best_acc
			best_acc = max(train_acc, best_acc)

			save_checkpoint(
				{
					"state": state,
					'result': all_results,
					"epoch": epoch + 1,
					'state_dict': model.state_dict(),
					"best_acc": best_acc,
					"optimizer": optimizer.state_dict(),
					"scheduler": scheduler.state_dict(),
					'init_weight': initialization,
				},
				args.save_dir,
				state,
				is_best,
			)

			plt.plot(all_results['train_acc'], label='train_acc')
			plt.plot(all_results['val_acc'], label='val_acc')
			plt.legend()
			plt.savefig(os.path.join(args.save_dir,str(state)+ 'train.png'))
			plt.close()
			print('Epoch {} took {} seconds'.format(epoch, time.time() - start_time))
		torch.save(
			model.state_dict(),os.path.join(args.save_dir, 'epoch_{}_weight.pt'.format(epoch + 1))
		)

		check_sparsity(model)
		print('Performance on test set')
		test_acc = validate(model = model, val_loader = test_loader, criterion = criterion, args = args)
		if len(all_results['test_acc']) != 0:
			val_pick_best_epoch = np.argmax(all_results['val_acc'])
			print(
				'Best validation accuracy: {} at epoch {}'.format(all_results['val_acc'][val_pick_best_epoch], val_pick_best_epoch)
			)

		all_results = {}
		all_results['train_acc'] = []
		all_results['test_acc'] = []
		all_results['val_acc'] = []
		best_acc = 0
		start_epoch = 0

		if args.prune_type == 'pt':
			print('loading pretrained model')
			initialization = torch.load(
				os.path.join(args.save_dir, '0model_best.pth.tar'),
				map_location='cuda:{}'.format(str(args.gpu)),
			)['state_dict']

		if args.random_prune:
			print('Random pruning')
			model = pruning_model_random(model, args.rate)
		else:
			print('L1 pruning')
			model = pruning_model(model, args.rate)
		print('model.named_parameters()')
		for name, param in model.named_parameters():
			print(name, param.size())
		print('len(list(model.parameters())) :', len(list(model.parameters())))
		remain_weight = check_sparsity(model)
		current_mask = extract_mask(model.state_dict())
		remove_prune(model)

		model.load_state_dict(initialization, strict=False)
		prune_model_custom(model, current_mask)
		optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

		if args.imagenet_arch:
			lambda0 = (
				lambda cur_iter: (cur_iter + 1) / args.warmup
				if cur_iter < args.warmup
				else (
					0.5
					* (
						1.0
						+ np.cos(
							np.pi
							* ((cur_iter - args.warmup) / (args.epochs - args.warmup))
						)
					)
				)
			)
			scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
		else:
			scheduler = torch.optim.lr_scheduler.MultiStepLR(
				optimizer, milestones=decreasing_lr, gamma=0.1
			)  # 0.1 is fixed
		if args.rewind_epoch:
			# learning rate rewinding
			for _ in range(args.rewind_epoch):
				scheduler.step()

if __name__ == '__main__':
	main()
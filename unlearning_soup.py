import arg_parser
import os
import time
from copy import deepcopy
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils import *
from evaluation import mia_ver3 as mia
from trainer import train, validate
def main():
    global args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = arg_parser.parse_args()
    print(args)

    torch.cuda.set_device(int(args.gpu))
    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        setup_seed(args.seed)

    # prepare dataset
    if args.dataset == "imagenet":
        args.class_to_replace = None
        model, train_loader, val_loader = setup_model_dataset(args)
    else:
        (
            model, 
            retain_train_loader,
            val_loader, 
            retain_test_loader, 
            test_loader, 
            marked_loader,
            forget_loader
        ) = setup_model_dataset(args)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=decreasing_lr, gamma=0.1
    )
    # 경로에 있는 모든 .pth.tar 파일을 불러와서 파일이름으로 dictionary에 저장.
    models_dict = {}
    for file in os.listdir(args.models_dir):
        if file.endswith(".pth.tar"):
            checkpoint = torch.load(os.path.join(args.models_dir, file))
            models_dict[file] = checkpoint["state_dict"]
    
            

    # 모델들을 MIA순으로 내림차순 정렬
    mias = {}
    for i, model_name in enumerate(models_dict):
        new_state_dict = OrderedDict()
        for k, v in models_dict[model_name].items():
            if 'mask' in k:
                name = k[:-5]
                new_state_dict[name] = v
        tmp_model = deepcopy(model)
        tmp_model.load_state_dict(new_state_dict, strict=False)
        print('model: ', model_name)
        mias[model_name] = mia(tmp_model,test_loader,retain_train_loader,forget_loader,1)
    for i in mias:
        print(i, mias[i])
    mias = dict(sorted(mias.items(), key=lambda x: x[1], reverse=True))

    mia_ = []
    acc_ = []
    models_emsemble = []
    ## mias 순으로 모델을 ensemble하여 성능을 평가
    for i, model_name in enumerate(mias):
        print('i: ', i)
        if i == 0:
            print('first model')
            print('model: ', )
            ensemble_model = deepcopy(model)
            new_state_dict = OrderedDict()
            for k, v in models_dict[model_name].items():
                if 'mask' in k:
                    name = k[:-5]
                    new_state_dict[name] = v
            ensemble_model.load_state_dict(new_state_dict, strict=False)
            best_mia = mias[model_name]
            models_emsemble.append(i)
        else:
            print('{}th model'.format(i+1))
            print('model: ', model_name)
            tmp_model = deepcopy(ensemble_model)
            org_state_dict = deepcopy(tmp_model.state_dict())
            new_state_dict = OrderedDict()
            for k, v in models_dict[model_name].items():
                if 'mask' in k:
                    name = k[:-5]
                    new_state_dict[name] = v
            average_state_dict = OrderedDict()
            for key in new_state_dict:
                average_state_dict[key] = (org_state_dict[key] + new_state_dict[key]) / 2
            
            tmp_model.load_state_dict(average_state_dict, strict=False)

            tmp_mia = mia(tmp_model,test_loader,retain_train_loader,forget_loader,1)
            acc_.append(validate(test_loader, tmp_model, criterion, args))
            mia_.append(tmp_mia)
            print('best_mia: ', best_mia)
            print('mia: ', tmp_mia)
            print('acc: ', acc_[-1])
            if tmp_mia > best_mia:
                print("Update Ensemble Model")
                ensemble_model = tmp_model
                best_mia = tmp_mia
                models_emsemble.append(model)
    print("Best MIA: ", best_mia)
    print("Best Accuracy: ", max(acc_))


    plt.plot(mias.values())
    plt.plot(mia_)
    plt.plot(acc_)
    plt.legend(["MIA", "MIA_ensemble", "Accuracy"])
    plt.savefig("MIA.png")


if __name__ == "__main__":
    main()

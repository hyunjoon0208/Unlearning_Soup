import torch
from collections import OrderedDict
from models import *
import torch.nn.utils.prune as prune
from utils import *
from trainer import train, validate
import arg_parser
def main():
    state_dict_path = ['/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/imp/ckpt/FTcheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/imp/ckpt/GAcheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/imp/ckpt/retraincheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/imp/ckpt/fisher_newcheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/omp/ckpt/FTcheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/omp/ckpt/GAcheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/omp/ckpt/retraincheckpoint.pth.tar',
                          '/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/unlearning/omp/ckpt/wfishercheckpoint.pth.tar'

    ]
    args = arg_parser.parse_args()
    (
            model, 
            retain_train_loader,
            val_loader, 
            retain_test_loader, 
            test_loader, 
            marked_loader,
            forget_loader
        ) = setup_model_dataset(args)
    for i in range(len(state_dict_path)):
        # model.load_state_dict(torch.load('/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/pruning/imp/epoch_182_weight.pt'))
        model.load_state_dict(torch.load('/home/hyunjoon/2024/lab/Unlearning/Unlearning_Soup/result/pruning/omp/epoch_182_weight.pt'))
        # model.load_state_dict(torch.load(state_dict_path)['state_dict'])
        state_dict = torch.load(state_dict_path[i])['state_dict']
        print(state_dict_path[i])
        new_state_dict = OrderedDict()
        cnt = 0
        for k, v in state_dict.items():
            # print('k:', k)
            if 'mask' in k:
                cnt += 1
                name = k[:-5] 
                new_state_dict[name] = v
        if cnt:
            print('new_state_dict')
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=decreasing_lr, gamma=0.1
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_acc = validate(test_loader, model, criterion, args)
        print(val_acc)
if __name__ == '__main__':
    main()

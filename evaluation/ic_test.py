import torch
import numpy as np
from sklearn.metrics import confusion_matrix

from utils import seed_everything, seed_worker

def collect_predictions(model, dataloader, device='cuda'):
    criterion = torch.nn.CrossEntropyLoss()
    dataloader = torch.utils.data.DataLoader(dataloader.dataset, num_workers=dataloader.num_workers, batch_size=128, shuffle=False)
    model.eval()
    reals=[]
    predicts=[]
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        probs = torch.nn.functional.softmax(output, dim=1)
        predict = np.argmax(probs.cpu().detach().numpy(),axis=1)
        reals = reals + list(target.cpu().detach().numpy())
        predicts = predicts + list(predict)

    return reals, predicts

def interclass_confusion(model, data_loader, seed, class_to_forget, device):
    seed_everything(seed)
    '''
        ic_err_test, fgt_test = interclass_confusion(model, test_loader, class_to_forget, 'cuda')
        ic_err_retain, fgt_retain = interclass_confusion(model, retain_loader, class_to_forget, 'cuda')
    '''
    reals, predicts = collect_predictions(model, data_loader, device=device)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # is this needed?
    cm = confusion_matrix(reals, predicts, labels=classes)
    counts = 0
    for i in range(len(cm)):
        if i != class_to_forget[0]:
            counts += cm[class_to_forget[0]][i]
        if i != class_to_forget[1]:
            counts += cm[class_to_forget[1]][i]
    
    ic_err = counts / (np.sum(cm[class_to_forget[0]]) + np.sum(cm[class_to_forget[1]]))
    fgt = cm[class_to_forget[0]][class_to_forget[1]] + cm[class_to_forget[1]][class_to_forget[0]]
    #print (cm)
    return ic_err, fgt
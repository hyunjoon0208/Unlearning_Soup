'''
MIA Train (label: data)
  - Member: Dr
  - Non-Member: Dt
MIA Eval (label: data)
  - Df
'''

# From Forgetting Outside the Box https://github.com/shash42/Evaluating-Inexact-Unlearning/blob/master/src/membership.py#L1
import torch
import torch.nn.functional as F

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN

def entropy(p, dim = -1, keepdim = False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)
    #return (p * p.log()).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model, seed):
    torch.manual_seed(seed)
    sampler = torch.utils.data.RandomSampler(data_loader.dataset, replacement=False, num_samples=4000)
    data_loader_small = torch.utils.data.DataLoader(data_loader.dataset, batch_size=128, num_workers=data_loader.num_workers, sampler=sampler, shuffle=False)
    prob = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader_small):
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)

def get_membership_attack_data(retain_loader, forget_loader, test_loader, model, seed):    
    retain_prob = collect_prob(retain_loader, model, seed)
    forget_prob = collect_prob(forget_loader, model, seed)
    test_prob = collect_prob(test_loader, model, seed)
    
    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])
    
    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])    
    return X_f, Y_f, X_r, Y_r

def get_membership_attack_prob(retain_loader, forget_loader, test_loader, model, seed):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, model, seed)
    clf = SVC(C=3,gamma='auto',kernel='rbf', random_state=seed)
    #clf = LogisticRegression(class_weight='balanced',solver='lbfgs',multi_class='multinomial', random_state=seed)
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    results1 = clf.predict(X_r)
    acc = accuracy_score(results, Y_f)
    train_acc = accuracy_score(results1, Y_r)
    TP, FP, TN, FN = perf_measure(Y_r, results1)
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)
    
    print (f"TP:{TP}, FP{FP}, TN{TN}, FN{FN}")
    print (f"false negative rate: {FN/(FN+TP)}")
    print (f"false positive rate: {FP/(FP+TN)}")
    return acc, train_acc, FPR, FNR 

def membership_inference_attack(model, data_loader, seed):
    t_loader, r_loader, f_loader = data_loader['test'], data_loader['retain'], data_loader['forget']
    prob, train_acc, FPR, FNR = get_membership_attack_prob(r_loader, f_loader, t_loader, model, seed)
    print("Attack prob: ", prob)
    print(f"Train Acc: {train_acc}")
    return prob * 100
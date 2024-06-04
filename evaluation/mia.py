'''
Unlearn-Sparse MIA 사용 예정
USE case
if "SVC_MIA_forget_efficacy" not in evaluation_result:
    test_len = len(test_loader.dataset)
    forget_len = len(forget_dataset)
    retain_len = len(retain_dataset)

    # remove augmentation
    utils.dataset_convert_to_test(retain_dataset, args)
    utils.dataset_convert_to_test(forget_loader, args)
    utils.dataset_convert_to_test(test_loader, args)

    shadow_train = torch.utils.data.Subset(retain_dataset, list(range(test_len)))
    shadow_train_loader = torch.utils.data.DataLoader(
        shadow_train, batch_size=args.batch_size, shuffle=False
    )

    evaluation_result["SVC_MIA_forget_efficacy"] = evaluation.SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=test_loader,
        target_train=None,
        target_test=forget_loader,
        model=model,
    )
    unlearn.save_unlearn_checkpoint(model, evaluation_result, args)

'''
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

import numpy as np
from sklearn.svm import SVC

def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)

def collect_prob(data_loader, model):
    if data_loader is None:
        return torch.zeros([0, 10]), torch.zeros([0])

    prob = []
    targets = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
            data, target = batch
            output = model(data)
            prob.append(F.softmax(output, dim=-1).data)
            targets.append(target)

    return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test, seed):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf", random_state=seed)
    clf.fit(X_shadow, Y_shadow)

    accs = []

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.append(acc_train)

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.append(acc_test)

    return np.mean(accs)

def membership_inference_attack(model, test_loader,retain_loader, forget_loader, seed):
    # t_loader, r_loader, f_loader = data_loader['test'], data_loader['retain'], data_loader['forget']
    t_loader, r_loader, f_loader = test_loader, retain_loader, forget_loader
    torch.manual_seed(seed)
    sampler = RandomSampler(r_loader.dataset, num_samples=len(t_loader.dataset))
    shadow_train = DataLoader(r_loader.dataset, batch_size=r_loader.batch_size, num_workers=r_loader.num_workers, shuffle=False, sampler=sampler)
    shadow_test = t_loader
    target_train = None
    target_test = f_loader

    shadow_train_prob, shadow_train_labels = collect_prob(shadow_train, model)
    shadow_test_prob, shadow_test_labels = collect_prob(shadow_test, model)

    target_train_prob, target_train_labels = collect_prob(target_train, model)
    target_test_prob, target_test_labels = collect_prob(target_test, model)

    # shadow_train_corr = (
    #     torch.argmax(shadow_train_prob, axis=1) == shadow_train_labels
    # ).int()
    # shadow_test_corr = (
    #     torch.argmax(shadow_test_prob, axis=1) == shadow_test_labels
    # ).int()
    # target_train_corr = (
    #     torch.argmax(target_train_prob, axis=1) == target_train_labels
    # ).int()
    # target_test_corr = (
    #     torch.argmax(target_test_prob, axis=1) == target_test_labels
    # ).int()

    shadow_train_conf = torch.gather(shadow_train_prob, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_prob, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_prob, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_prob, 1, target_test_labels[:, None])

    # shadow_train_entr = entropy(shadow_train_prob)
    # shadow_test_entr = entropy(shadow_test_prob)

    # target_train_entr = entropy(target_train_prob)
    # target_test_entr = entropy(target_test_prob)

    # shadow_train_m_entr = m_entropy(shadow_train_prob, shadow_train_labels)
    # shadow_test_m_entr = m_entropy(shadow_test_prob, shadow_test_labels)
    # if target_train is not None:
    #     target_train_m_entr = m_entropy(target_train_prob, target_train_labels)
    # else:
    #     target_train_m_entr = target_train_entr
    # if target_test is not None:-
    #     target_test_m_entr = m_entropy(target_test_prob, target_test_labels)
    # else:
    #     target_test_m_entr = target_test_entr

    # acc_corr = SVC_fit_predict(
    #     shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr, seed
    # )
    acc_conf = SVC_fit_predict(
        shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf, seed
    )
    # acc_entr = SVC_fit_predict(
    #     shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr, seed
    # )
    # acc_m_entr = SVC_fit_predict(
    #     shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr, seed
    # )
    # acc_prob = SVC_fit_predict(
    #     shadow_train_prob, shadow_test_prob, target_train_prob, target_test_prob, seed
    # )
    # m = {
    #     "correctness": acc_corr,
    #     "confidence": acc_conf,
    #     "entropy": acc_entr,
    #     "m_entropy": acc_m_entr,
    #     "prob": acc_prob,
    # }

    # print(m)
    return acc_conf * 100 # returning confidence
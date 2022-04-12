import torch
import timm
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F
from avalanche.evaluation.metrics.accuracy import Accuracy
from torch.utils.data import DataLoader
from stat_tracker import *
import numpy as np
import random
from dataset import *
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cub', choices=['cub', 'cifar', 'flowers', 'caltech'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    dataset = args.dataset
    seed = args.seed
    max_epoch = args.max_epoch
    batch_size = args.batch_size


    model_name = 'resnet18'
    resnet = timm.create_model(model_name, pretrained=True)
    add_tracker(resnet)

    ds = get(resnet, dataset)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    train_stream = ds.train_stream
    test_stream = ds.test_stream
    n_experiences = ds.n_experiences
    n_classes = ds.n_classes
    if n_experiences == 11: n_experiences = 10
    resnet.reset_classifier(n_classes)

    criteria = nn.CrossEntropyLoss()
    acc = Accuracy()
    resnet = resnet.cuda()

    r_mean = [[] for _ in range(n_experiences + 1)]
    r_var = [[] for _ in range(n_experiences + 1)]
    weight = [[] for _ in range(n_experiences + 1)]
    bias = [[] for _ in range(n_experiences + 1)]
    t_mean = [[] for _ in range(n_experiences + 1)]
    layers = get_layer(resnet)
    for _ in layers:
        if type(_) == nn.BatchNorm2d:
            r_mean[0].append(deepcopy(_.running_mean.cpu().data))
            r_var[0].append(deepcopy(_.running_var.cpu().data))
            weight[0].append(deepcopy(_.weight.cpu().data))
            bias[0].append(deepcopy(_.bias.cpu().data))
        elif type(_) == StatTracker:
            t_mean[0].append(deepcopy(_.getmean().cpu().data))

    for t, experience in enumerate(train_stream[:n_experiences]):
        opt = SGD(resnet.parameters(), lr=1e-2)
        y2class = torch.tensor(experience.classes_in_this_experience, dtype=torch.long).cuda()
        class2y = torch.zeros(n_classes, dtype=torch.long).cuda()
        for i, _ in enumerate(y2class):
            class2y[_] = i
        print("Start of task ", t)
        print('Classes in this task:', y2class)
        current_training_set = experience.dataset
        dl = DataLoader(current_training_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
        for epoch in range(max_epoch):
            resnet.train()
            if epoch == 0:
                for name, _ in resnet.named_parameters():
                    if 'fc' in name:
                        _.requires_grad = True
                    else:
                        _.requires_grad = False
            elif epoch == max_epoch // 5:
                for name, _ in resnet.named_parameters():
                    if 'bn' in name or 'downsample.1' in name:
                        _.requires_grad = True
            elif epoch == max_epoch // 2:
                for _ in resnet.parameters():
                    _.requires_grad = True
            pbar = tqdm(dl)
            for x, y, z in pbar:
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                out = resnet(x)[:, y2class]
                loss = criteria(out, class2y[y])
                opt.zero_grad()
                loss.backward()
                pbar.set_description("epoch %d, loss %f" % (epoch, loss.cpu().item()))
                opt.step()

        acc.reset()
        resnet.eval()
        r_mean[t + 1] = []
        r_var[t + 1] = []
        weight[t + 1] = []
        bias[t + 1] = []
        t_mean[t + 1] = []
        for _ in layers:
            if type(_) == nn.BatchNorm2d:
                r_mean[t + 1].append(deepcopy(_.running_mean.cpu().data))
                r_var[t + 1].append(deepcopy(_.running_var.cpu().data))
                weight[t + 1].append(deepcopy(_.weight.cpu().data))
                bias[t + 1].append(deepcopy(_.bias.cpu().data))
            elif type(_) == StatTracker:
                t_mean[t + 1].append(deepcopy(_.getmean().cpu().data))

        with torch.no_grad():
            for i in range(t + 1):
                j = 0
                y2class = torch.tensor(test_stream[i].classes_in_this_experience, dtype=torch.long).cuda()
                class2y = torch.zeros(n_classes, dtype=torch.long).cuda()
                for k, _ in enumerate(y2class):
                    class2y[_] = k
                for _ in layers:
                    if type(_) == nn.BatchNorm2d:
                        _.running_mean = deepcopy(r_mean[i + 1][j]).cuda()
                        _.running_var = deepcopy(r_var[i + 1][j]).cuda()
                        _.weight.data += deepcopy(weight[i + 1][j]).cuda() - _.weight.data
                        _.bias.data += deepcopy(bias[i + 1][j]).cuda() - _.bias.data
                        j += 1
                    elif type(_) == StatTracker:
                        _.setmean(deepcopy(t_mean[i + 1][j]).cuda())
                if i != t:
                    correct_shift(resnet)
                current_test_set = test_stream[i].dataset
                test_dl = DataLoader(current_test_set, batch_size=128, shuffle=False, drop_last=False,
                                     num_workers=4)
                for x, y, z in tqdm(test_dl):
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    out = resnet(x)
                    acc.update(out[:, y2class].argmax(dim=1).view(-1), class2y[y], i)
            print(acc.result())
            print(sum(acc.result().values()) / len(acc.result()))


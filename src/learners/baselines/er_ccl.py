import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import random as r
import numpy as np
import os
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.learners.ccldc.baseccl import BaseCCLLearner
from src.utils.losses import SupConLoss
from src.buffers.reservoir import Reservoir
from src.models.resnet import ResNet18, ImageNet_ResNet18
from src.utils.metrics import forgetting_line   
from src.utils.utils import get_device

from copy import deepcopy

device = get_device()
scaler = amp.GradScaler()

LR_MIN = 5e-4
LR_MAX = 5e-2
class _ECELoss(torch.nn.Module):
    def __init__(self, n_bins=10):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_accs, bin_confs = [], []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                bin_accs.append(accuracy_in_bin.item())
                bin_confs.append(avg_confidence_in_bin.item())
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)
        return ece.item(), bin_confs, bin_accs
class ERCCLLearner(BaseCCLLearner):
    def __init__(self, args):
        super().__init__(args)
        self.buffer = Reservoir(
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method,
        )
        self.previous_model = None
        self.kd_lambda = self.params.kd_lambda
        self.iter = 0
        self.tf_ema = nn.Sequential(
            RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2),
        ).to(device)  # 在训练过程中增强输入图像x^

    def load_model(self, **kwargs):
        if self.params.dataset == 'cifar10' or self.params.dataset == 'cifar100' or self.params.dataset == 'tiny':
            return ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)
        elif self.params.dataset == 'imagenet' or self.params.dataset == 'imagenet100':
            return ImageNet_ResNet18(
                dim_in=self.params.dim_in,
                nclasses=self.params.n_classes,
                nf=self.params.nf
            ).to(device)

    
    def load_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, dataloader, **kwargs):
        task_name  = kwargs.get('task_name', 'unknown task')
        task_id    = kwargs.get('task_id', 0)
        dataloaders = kwargs.get('dataloaders', None)
        self.model1 = self.model1.train()
        self.model2 = self.model2.train()

        for j, batch in enumerate(dataloader):
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x)
            
            for _ in range(self.params.mem_iters):
                mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)

                if mem_x.size(0) > 0:

                    # Combined batch
                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    # Augment
                    bs = combined_x.size(0)
                    combined_aug = self.transform_train(combined_x)
                    # Inference

                    feat, logits = self.model1.full(combined_aug)
                    feat2, logits2 = self.model2.full(combined_aug)


                    loss_ce = self.criterion(logits, combined_y.long())
                    loss_ce2 = self.criterion(logits2, combined_y.long())

                    combined_x, combined_y = self.combine(batch_x, batch_y, mem_x, mem_y)  # (batch_size, nb_channel, img_size, img_size)
                    # Augment
                    combined_aug1 = self.transform_1(combined_x)
                    combined_aug2 = self.transform_2(combined_aug1)
                    combined_aug = self.transform_3(combined_aug2)
                    # Inference

                    logits1 = self.model1.logits(combined_aug)
                    logits2 = self.model2.logits(combined_aug)

                    logits1_vanilla = self.model1.logits(combined_x)
                    logits2_vanilla = self.model2.logits(combined_x)

                    logits1_step1 = self.model1.logits(combined_aug1)
                    logits2_step1 = self.model2.logits(combined_aug1)

                    logits1_step2 = self.model1.logits(combined_aug2)
                    logits2_step2 = self.model2.logits(combined_aug2)
                    # Cls Loss
                    loss_ce += self.criterion(logits1, combined_y.long()) + self.criterion(logits1_vanilla, combined_y.long()) + self.criterion(logits1_step1, combined_y.long()) + self.criterion(logits1_step2, combined_y.long())
                    loss_ce2 += self.criterion(logits2, combined_y.long()) + self.criterion(logits2_vanilla, combined_y.long()) + self.criterion(logits2_step1, combined_y.long()) + self.criterion(logits2_step2, combined_y.long())

                    # Distillation Loss
                    loss_dist = self.kl_loss(logits1, logits2.detach()) + self.kl_loss(logits1_vanilla, logits2_step1.detach()) + self.kl_loss(logits1_step1, logits2_step2.detach()) + self.kl_loss(logits1_step2, logits2.detach()) 
                    loss_dist2 = self.kl_loss(logits2, logits1.detach()) + self.kl_loss(logits2_vanilla, logits1_step1.detach()) + self.kl_loss(logits2_step1, logits1_step2.detach()) + self.kl_loss(logits2_step2, logits1.detach())

                    # Total Loss
                    loss = 0.5 * loss_ce  + self.kd_lambda * loss_dist 
                    loss2 = 0.5 * loss_ce2 + self.kd_lambda * loss_dist2 

                    self.loss = loss.item()
                    self.loss2 = loss2.item()
                    print(f"Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   batch {j}", end="\r")
                    self.optim1.zero_grad()
                    loss.backward()
                    self.optim1.step()

                    self.optim2.zero_grad()
                    loss2.backward()
                    self.optim2.step()

                    self.iter += 1
            # Update reservoir buffer
            self.buffer.update(imgs=batch_x, labels=batch_y)

            if (j == (len(dataloader) - 1)) and (j > 0):
                print(
                    f"Task : {task_name}   batch {j}/{len(dataloader)}   Loss (Peer1) : {loss.item():.4f}  Loss (Peer2) : {loss2.item():.4f}   time : {time.time() - self.start:.4f}s"
                )


    def print_results(self, task_id):
        n_dashes = 20
        pad_size = 8
        print('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        print('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for line in self.results:
            print('Acc'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line), f"{np.nanmean(line):.4f}")
    

    def combine(self, batch_x, batch_y, mem_x, mem_y):
        mem_x, mem_y = mem_x.to(self.device), mem_y.to(self.device)
        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
        combined_x = torch.cat([mem_x, batch_x])
        combined_y = torch.cat([mem_y, batch_y])
        if self.params.memory_only:
            return mem_x, mem_y
        else:
            return combined_x, combined_y
        
    def evaluate_ece_curve(self, dataloader, task_id, n_bins=10):
        self.model1.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                logits = self.model1.logits(self.tf_ema(inputs))
                all_logits.append(logits)
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        ece_criterion = _ECELoss(n_bins=n_bins)
        ece, bin_confs, bin_accs = ece_criterion(all_logits, all_labels)

        print(f"Task {task_id} ECE = {ece:.4f}")
        self.plot_reliability_curve(bin_confs, bin_accs, task_id, n_bins)


    def plot_reliability_curve(self, confidences, accuracies, task_id, n_bins=10):   
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = bin_edges[:-1]
        bin_width = bin_edges[1] - bin_edges[0]
        #bins = np.linspace(0, 1, n_bins)
        width = 1 / n_bins
        plt.figure(figsize=(5, 5))
        plt.bar(bin_centers, accuracies, width=width,alpha=1, label='Accuracy',align='edge',edgecolor='black',zorder=3)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration',linewidth=2,zorder=4)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, linestyle='-.', linewidth=0.5, zorder=0)
        plt.savefig(f"./ece_task{task_id}.png")  
        plt.close()
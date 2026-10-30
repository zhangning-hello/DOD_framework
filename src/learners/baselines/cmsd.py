import torch
import time
import torch.nn as nn
import sys
import logging as lg
import random as r
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import torch.cuda.amp as amp
import random
import wandb

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import matplotlib.pyplot as plt
from src.utils.utils import maybe_cuda, AverageMeter
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from copy import deepcopy
from src.learners.base import BaseLearner
from src.learners.baselines.er import ERLearner
from src.utils.losses import WKDLoss 
from src.models.resnet import ResNet18
from src.utils import name_match
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device, filter_labels
from src.utils.utils import get_device
from src.utils.augment import MixupAdaptative, ZetaMixup
from torch.distributions import Categorical
device = get_device()
scaler = amp.GradScaler()

LR_MIN = 5e-4
LR_MAX = 5e-2
class CMSDLearner(ERLearner):
    def __init__(self, args):
        super().__init__(args)
        self.wkdloss = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb= not self.params.no_wandb,
            alpha_kd=self.params.alpha_kd
        )

        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        
        self.ema_models = {}
        self.ema_alphas = {}
        if self.params.alpha_min is None or self.params.alpha_max is None:
            # Manually set 5 teachers
            self.ema_models[0] = deepcopy(self.model)
            self.ema_models[1] = deepcopy(self.model)
            self.ema_alphas[0] = self.params.ema_alpha3
            self.ema_alphas[1] = self.params.ema_alpha4   
        else:
            # Automatically set a specific number of teachers
            for i in range(self.params.n_teacher):
                self.ema_models[i] = deepcopy(self.model)
                self.ema_alphas[i] = 10**(
                    np.log10(self.params.alpha_min) + 
                    (np.log10(self.params.alpha_max) - np.log10(self.params.alpha_min))*i/(max(1,self.params.n_teacher - 1))
                    )
        print(self.ema_alphas)
        
        self.tf_ema = nn.Sequential(
            RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)),
            RandomHorizontalFlip(),
            ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            RandomGrayscale(p=0.2),
        ).to(device)  # 在训练过程中增强输入图像x^

        self.update_ema(init=True)
        self.previous_model = None
        if self.params.measure_drift >= 0:
            self.drift = []
            self.previous_model = None
    def update_ema(self, init=False):
        """
        Update the Exponential Moving Average (EMA) of the group of pytorch models
        """
        for i, ema_model in enumerate(self.ema_models.values()):
            alpha = self.ema_alphas[i]
            for param, ema_param in zip(self.model.parameters(), ema_model.parameters()):
                p = deepcopy(param.data.detach())
                
                if init:
                    ema_param.data.mul_(0).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))
                else:
                    ema_param.data.mul_(1 - alpha).add_(p * alpha / (1 - alpha ** max(1, self.stream_idx // (self.params.batch_size * self.params.ema_correction_step))))        
    # @profile
    def train(self, dataloader, **kwargs):
        task_name = kwargs.get("task_name", "Unknown")
        task_id = kwargs.get('task_id', None)
        losses = AverageMeter()
        acc_batch = AverageMeter()
        # virtual update trackers
        losses_vm1 = AverageMeter()
        losses_vm2 = AverageMeter()
        acc_batch_vm1 = AverageMeter()
        acc_batch_vm2 = AverageMeter()
        for j, batch in enumerate(dataloader):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                # Stream batch
                batch_x, batch_y = batch[0], batch[1]
                self.stream_idx += len(batch_x)
                # self.ema_model1.train()
                self.stream_idx += len(batch_x)
                batch_x = maybe_cuda(batch_x, self.cuda)
                batch_y = maybe_cuda(batch_y, self.cuda)
                batch_x_copy = batch_x.clone()
                batch_y_copy = batch_y.clone()
                self.model = self.model.train()
                for _ in range(self.params.mem_iters):
                    self.optim.zero_grad()
                    # Iteration over memory + stream
                    #UMD, Compute the predictive entropy of different teachers
                    vm1 = self.ema_models[0]
                    vm2 = self.ema_models[1]
                    with torch.no_grad():
                        mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                        if mem_x.size(0) > 0:
                            mem_x = mem_x.to(device)
                            batch_x = batch_x.to(device)
                            combined_x = torch.cat([mem_x, batch_x]).to(device)  # 结合后的x
                            mem_y = mem_y.to(device)
                            batch_y = batch_y.to(device)
                            combined_y = torch.cat([mem_y, batch_y]).to(device)
                            combined_aug = self.tf_ema(combined_x)
                            
                            logits_vm1 = torch.cat([vm1.logits(combined_x), vm1.logits(combined_aug)], dim=1)
                            logits_orig_vm1 = vm1.logits(combined_x)  
                    
                            probs_orig_vm1 = F. softmax(logits_orig_vm1, dim=1)

                            batch_size = combined_x.shape[0]
                            
                            entropy = Categorical(probs=probs_orig_vm1).entropy()
                            entropy_vm1 = entropy.sum().item() 
                            entropy_avg_vm1 = entropy_vm1 / batch_size

                            acc_vm1 = entropy_avg_vm1
                        else:
                            acc_vm1 = 0
                        # Compare the predictive entropy of different teachers
                        if mem_x.size(0) > 0:
                            logits_vm2 = torch.cat([vm2.logits(combined_x), vm2.logits(combined_aug)], dim=1)
                            logits_orig_vm2 = vm2.logits(combined_x)  
                            logits_aug_vm2  = vm2.logits(combined_aug)
                            probs_orig_vm2 = F.softmax(logits_orig_vm2, dim=1)
                            probs_aug_vm2  = F.softmax(logits_aug_vm2, dim=1) 
                            avg_probs = 0.5 * (probs_orig_vm2 + probs_aug_vm2)
                            probs_orig_vm2 = F.softmax(logits_orig_vm2, dim=1) 
                            batch_size = combined_x.shape[0]
                        
                            entropy = Categorical(probs=probs_orig_vm2).entropy()
                            entropy_vm2 = entropy.sum().item() 
                            entropy_avg_vm2 = entropy_vm2 / batch_size
                            acc_vm2 = entropy_avg_vm2
                            
                        else:
                            acc_vm2 = 0
                    if acc_vm2 > acc_vm1 :
                           
                        self.teacher = vm2
                    elif acc_vm1 > acc_vm2 :
                        self.teacher = vm2
                    else:
                        self.teacher = vm2
                    #RMD
                    mem_x, mem_y = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    if mem_x.size(0) > 0:
                        mem_x = mem_x.to(device)
                        batch_x = batch_x.to(device)
                        combined_x = torch.cat([mem_x, batch_x]).to(device)
                        mem_y = mem_y.to(device)
                        batch_y = batch_y.to(device)
                        combined_y = torch.cat([mem_y, batch_y]).to(device)
                        combined_y = combined_y.long()
                        # Augment
                        combined_aug = self.transform_train(combined_x)
                        combined_aug_list = [self.transform_train(combined_x) for _ in range(4)]

                        # logits
                        logits_stu = self.model.logits(combined_aug)
                        logits_stu_raw = self.model.logits(combined_x)
                        
                        loss_dist = 0
                        loss_dist1 = 0
                        logits_tea_x =self.teacher.logits(combined_x)
                        logits_tea = self.teacher.logits(combined_aug)
                        # PR
                        teacher_model_prob1 = F.softmax(logits_tea_x/3, 1)
                        teacher_model_prob2 = F.softmax(logits_tea/3, 1)
                        label_mask = F.one_hot(combined_y, num_classes=logits_tea.shape[-1]) > 0
                        adaptive_weight_1 = teacher_model_prob1[label_mask]
                        adaptive_weight_2 = teacher_model_prob2[label_mask]
                        #caculate overall loss
                        loss_dist += (adaptive_weight_2 * self.wkdloss(logits_tea.detach(), logits_stu) +adaptive_weight_1 *  self.wkdloss(logits_tea.detach(), logits_stu_raw))/2  #RMD loss
                        loss_dist1 = 3*self.wkdloss(logits_tea_x.detach(), logits_stu_raw) # UMD loss
                        loss_dist = loss_dist.mean()
                        loss_ce = self.criterion(logits_stu, combined_y.long())
                        loss = self.params.kd_lambda*loss_dist + 3*loss_ce+loss_dist1
                            
                        loss = loss.mean()

                        # Backprop
                        self.loss = loss.item()
                        
                        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False):
                            scaler.scale(loss).backward()
                            scaler.step(self.optim)
                            scaler.update()
                        self.update_ema()
                        if self.params.annealing:
                            self.scheduler.step()
                        self.optim.zero_grad()
                        
                        if self.params.measure_drift >=0 and task_id > 0:
                            self.measure_drift(task_id)
                        
                        if not self.params.no_wandb:
                            wandb.log({
                                "loss_dist": loss_dist.item(),
                                "loss": loss.item()
                            })
                        print(f"Phase: {task_name}  Loss:{loss.item():.3f}  Loss dist:{loss_dist.item():.3f} batch {j}", end="\r")
                self.buffer.update(imgs=batch_x, labels=batch_y)
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if self.params.tsne and task_id == 4:
                        self.tsne()
                    print(
                        f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s",
                        end="\r"
                    )
                    self.save(model_name=f"ckpt_{task_name}.pth")
    
    def train_uni(self, dataloader, **kwargs):
        raise NotImplementedError

    def encode(self, dataloader, model_tag=0, nbatches=-1, **kwargs):
        self.init_agg_model()
        if not self.params.drop_fc:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)

                    logits = self.model_agg.logits(self.transform_test(inputs))
                    preds = nn.Softmax(dim=1)(logits).argmax(dim=1)
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_preds = preds.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_preds = np.hstack([all_preds, preds.cpu().numpy()])
                    i += 1
            
            return all_preds, all_labels
        else:
            i = 0
            with torch.no_grad():
                for sample in dataloader:
                    if nbatches != -1 and i >= nbatches:
                        break
                    inputs = sample[0]
                    labels = sample[1]
                    
                    inputs = inputs.to(self.device)
                    features, _ = self.model_agg(self.transform_test(inputs))
                    
                    if i == 0:
                        all_labels = labels.cpu().numpy()
                        all_feat = features.cpu().numpy()
                    else:
                        all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                        all_feat = np.vstack([all_feat, features.cpu().numpy()])
                    i += 1
            return all_feat, all_labels
    
    def init_agg_model(self):
        
        teacher_1 = list(self.ema_models.values())[0]
        teacher_2 = list(self.ema_models.values())[1]
        # 使用其中一个模型作为结构模板
        self.model_agg = deepcopy(teacher_1)

        # 对每个参数执行逐元素平均
        with torch.no_grad():
            for param_agg, param_1, param_2 in zip(
                self.model_agg.parameters(),
                teacher_1.parameters(),
                teacher_2.parameters()
            ):
                param_agg.copy_((param_1.detach() + param_2.detach()) / 2.0)
        
        #self.model_agg = deepcopy(self.model)
        self.model_agg.eval()


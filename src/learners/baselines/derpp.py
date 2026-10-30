
import torch
import time
import torch.nn as nn
import sys
import logging as lg
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import torchvision

from sklearn.metrics import accuracy_score

from src.learners.baselines.er import ERLearner
from src.buffers.logits_res import LogitsRes
from src.models.resnet import ResNet18
from src.utils.metrics import forgetting_line

from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import get_device

device = get_device()

class DERppLearner(ERLearner):
    def __init__(self, args):
        super().__init__(args) #调用父类 ERLearner 的初始化方法
        self.buffer = LogitsRes( #用作动态缓冲区，用于存储之前任务的样本
            max_size=self.params.mem_size,
            img_size=self.params.img_size,
            nb_ch=self.params.nb_channels,
            n_classes=self.params.n_classes,
            drop_method=self.params.drop_method
        )

    def load_criterion(self):
        return F.cross_entropy #交叉熵函数

    
    def train(self, dataloader, **kwargs):
        """
        :param dataloader:数据加载器
        :return:
        """
        self.model = self.model.train()#将模型设置为训练模式
        if self.params.training_type == 'inc':#增量训练
            self.train_inc(dataloader, **kwargs)
        elif self.params.training_type == "uni":#统一训练，同时处理所有任务的数据
            self.train_uni(dataloader, **kwargs)
        elif self.params.training_type == "blurry": #模糊训练
            self.train_blurry(dataloader, **kwargs)

    def train_uni(self, dataloader, **kwargs): #统一训练
        raise NotImplementedError
        
    def train_inc(self, dataloader, task_name, **kwargs):
        """Adapted from https://github.com/aimagelab/mammoth/blob/master/models/derpp.py
        """
        """
        dataloader: 提供当前任务的数据的迭代器。
task_   name: 当前任务的名称。
        """
        task_id = kwargs.get("task_id", None) #从kwargs中获取任务ID
        for j, batch in enumerate(dataloader): #遍历数据加载器中的每个批次
            self.model = self.model.train() #训练模式
            # Stream data
            batch_x, batch_y = batch[0], batch[1] #提取数据和标签
            self.stream_idx += len(batch_x) #更新处理的样本数量
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad()
                outputs = self.model.logits(self.transform_train(batch_x.to(device))) #前向传播
                loss = self.criterion(outputs, batch_y.long().to(device)) #计算损失
                mem_x, _, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size) #从缓冲区中随机检索样本
                #计算记忆损失
                if mem_x.size(0) > 0:
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    
                    # Loss
                    loss += self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits.to(device))
                    
                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    loss += self.params.derpp_beta * self.criterion(mem_outputs, mem_y.long().to(device))

                self.loss = loss.mean().item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                loss.backward() #反向传播计算梯度
                self.optim.step() #优化器根据梯度更新网络参数
                
                if self.params.measure_drift >=0 and task_id > 0:
                    self.measure_drift(task_id) #是否发生漂移

            # Update buffer
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=outputs.detach()) #更新缓冲区
            if (j == (len(dataloader) - 1)) and (j > 0): #如果是最后一个批次，用来衡量特征判别
                if self.params.tsne and task_id == self.params.n_tasks - 1:
                    self.tsne()
                lg.info(
                    f"Phase : {task_name}   batch {j}/{len(dataloader)}   Loss : {self.loss:.4f}    time : {time.time() - self.start:.4f}s"
                )
    
    def train_blurry(self, dataloader, **kwargs): #模糊训练
        self.model.train()
        for j, batch in enumerate(dataloader):
            self.model = self.model.train()
            # Stream data
            batch_x, batch_y = batch[0], batch[1]
            self.stream_idx += len(batch_x) #更新处理的样本数量
            
            for _ in range(self.params.mem_iters):
                self.optim.zero_grad() #清空优化器的梯度信息
                outputs = self.model.logits(self.transform_train(batch_x.to(device))) #前向传播
                loss = self.criterion(outputs, batch_y.long().to(device)) #计算当前批次的损失
                mem_x, _, mem_logits = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size) #采样
                if mem_x.size(0) > 0:#求损失
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    
                    # Loss
                    loss += self.params.derpp_alpha * F.mse_loss(mem_outputs, mem_logits.to(device))
                    
                    mem_x, mem_y, _ = self.buffer.random_retrieve(n_imgs=self.params.mem_batch_size)
                    mem_outputs = self.model.logits(self.transform_train(mem_x.to(device)))
                    loss += self.params.derpp_beta * self.criterion(mem_outputs, mem_y.long().to(device))

                self.loss = loss.mean().item()
                print(f"Loss {self.loss:.3f}  batch {j}", end="\r")
                loss.backward()
                self.optim.step()
                    
            # Update buffer
            self.buffer.update(imgs=batch_x.detach(), labels=batch_y.detach(), logits=outputs.detach()) #更新缓冲区

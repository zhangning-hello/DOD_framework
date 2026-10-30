from random import random
import torch
import logging as lg
import os
import pickle
import time
import os
import datetime as dt
import numpy as np
import pandas as pd
import torch.nn as nn
import random as r
import json
import torchvision
import wandb
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
from torchvision import transforms
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
from src.utils.utils import maybe_cuda, AverageMeter
import copy
from copy import deepcopy
from src.utils.utils import save_model
from src.models import resnet
from src.utils.metrics import forgetting_line 
from src.utils.utils import get_device

device = get_device()

class BaseLearner(torch.nn.Module):
    def __init__(self, args):
        """Learner abstract class
        Args:
            args: argparser arguments
        """
        super().__init__()
        self.params = args # 存储参数对象
        self.device = get_device()  #获取设备（CPU或GPU
        self.init_tag() #初始化标签或标识符
        self.model = self.load_model() #加载模型
        self.optim = self.load_optim() #加载优化器
        self.buffer = None #缓冲区
        self.start = time.time()
        self.criterion = self.load_criterion() #加载损失函数
        self.old_labels = []
        self.new_labels = []
        self.task_seen = 0
        self.error_list = []
        self.new_class_score = []
        self.old_class_score = []
        self.fc_norm_new = []
        self.fc_norm_old = []
        self.bias_norm_new = []
        self.bias_norm_old = []
        self.lbl_inv_map = {}
        self.class_task_map = {}
        if self.params.tensorboard:
            self.writer = self.load_writer()
        # self.classifiers_list = ['logreg', 'knn', 'linear', 'svm', 'ncm']  # Classifiers used for evaluating representation
        self.classifiers_list = ['ncm']  # Classifiers used for evaluating representation
        self.loss = 0 #初始化损失值
        self.stream_idx = 0 #初始化流索引
        self.init_results() #初始化结果记录
        
        # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if self.params.dataset == 'tiny' else nn.Identity()
        normalize = nn.Identity() #标准化
        if self.params.tf_type == 'partial':
            self.transform_train = nn.Sequential(
                torchvision.transforms.RandomCrop(self.params.img_size, padding=4), #RandomCrop: 随机裁剪图像
                RandomHorizontalFlip(), #随机水平翻转
                normalize #标准化图像
            ).to(device) #定义训练数据的预处理步骤
        elif self.params.tf_type == 'full':
            self.transform_train = nn.Sequential(
                RandomResizedCrop(size=(self.params.img_size, self.params.img_size), scale=(self.params.min_crop, 1.)), #随机调整图像大小并裁剪
                RandomHorizontalFlip(), #随机水平翻转图像
                ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8), #随机改变图像的亮度、对比度和饱和度
                RandomGrayscale(p=0.2), #随机将图像转换为灰度
                normalize #标准化
            ).to(device)
        
        self.transform_test = nn.Sequential(
            normalize
        ).to(device) #定义测试数据的预处理步骤

    def before_train(self, x_train, y_train):
        phase_new_labels = list(set(y_train.tolist()))

        # print('before train')
        # print('self.new_labels before addition: ', self.new_labels)
        overlapped_labels = list(set(phase_new_labels).intersection(
            set(self.old_labels)))
        new_non_overlap_labels = list(set(phase_new_labels).difference(set(overlapped_labels)))
        self.new_labels += phase_new_labels
        for i, lbl in enumerate(new_non_overlap_labels):
            self.lbl_inv_map[lbl] = len(self.old_labels) + i

        for i in phase_new_labels:
            self.class_task_map[i] = self.task_seen

    def after_train(self):
        overlapped_labels = list(set(self.new_labels).intersection(set(self.old_labels)))  #recalculate overlapped and non-overlapped labels after each phase
        new_non_overlap_labels = list(set(self.new_labels).difference(set(overlapped_labels)))
        self.old_labels += new_non_overlap_labels #add only non-overlapped labels to old labels after each phase
        self.new_labels_zombie = copy.deepcopy(self.new_labels) #for zombie copy, should i include overlapped labels?
        self.new_labels.clear()
        self.task_seen += 1
        self.model.train()
        mem_x = self.buffer.buffer_imgs[:self.buffer.n_seen_so_far]
        mem_y = self.buffer.buffer_labels[:self.buffer.n_seen_so_far]
        if mem_x.size(0) > 0:
            rv_dataset = TensorDataset(mem_x, mem_y)
            rv_loader = DataLoader(rv_dataset, batch_size=self.params.eps_mem_batch, shuffle=True, num_workers=0,
                                       drop_last=True)
            for ep in range(1):
                for i, batch_data in enumerate(rv_loader):
                        # batch update
                    batch_x, batch_y = batch_data
                    batch_x = maybe_cuda(batch_x, self.cuda)
                    batch_y = maybe_cuda(batch_y, self.cuda)
                    print("batch_x:", batch_x.shape)
                    print("batch_y:",batch_y.shape)
                    logits = self.model.forward(batch_x)
                    logits = torch.cat([self.model.forward(batch_x).unsqueeze(1),
                                                  self.model.forward(self.transform(batch_x)).unsqueeze(1)], dim=1)
                    print("logits:", logits.shape)
                    loss = self.criterion(logits, batch_y)
                    self.opt.zero_grad()
                    loss.backward()
                    params = [p for p in self.model.parameters() if p.requires_grad and p.grad is not None]
                    grad = [p.grad.clone()/10. for p in params]
                    for g, p in zip(grad, params):
                        p.grad.data.copy_(g)
                    self.opt.step()


    def init_results(self):
        """Initialise results placeholder，初始化用于存储实验结果和遗忘率（forgetting rate）的字典
        """
        self.results = {}
        self.results_forgetting = {}
        for clf_name in self.classifiers_list: #遍历 self.classifiers_list 中的每个分类器名称
            #对于每个分类器，初始化一个空列表来存储结果和遗忘率
            self.results[clf_name] = []
            self.results_forgetting[clf_name] = []

    def init_tag(self):
        """Initialise tag for experiment 唯一的标签（tag），用于标识和记录实验的配置
        """
        if self.params.training_type == 'inc': #增量学习
            self.params.tag = f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}{self.params.tag}"  #标签包括学习器类型、数据集、记忆体大小、记忆体批次大小和标准批次大小
        elif self.params.training_type == "blurry": #模糊训练
            self.params.tag =\
                 f"{self.params.learner},{self.params.dataset},m{self.params.mem_size}mbs{self.params.mem_batch_size}sbs{self.params.batch_size}blurry{self.params.blurry_scale}{self.params.tag}" #标签包括学习器类型、数据集、记忆体大小、记忆体批次大小、标准批次大小、模糊训练特有的参数（例如模糊尺度）。
        else: #统一训练
            self.params.tag = f"{self.params.learner},{self.params.dataset},{self.params.epochs}b{self.params.batch_size},uni{self.params.tag}" #标签包括学习器类型、数据集、训练周期数、批次大小和统一训练的标识。
        print(f"Using the following tag for this experiment : {self.params.tag}") #打印标签，确认实验使用的标签
    
    def load_writer(self):
        """Initialize tensorboard summary writer初始化TensorBoard的摘要写入器
        """
        from torch.utils.tensorboard import SummaryWriter #导入 SummaryWriter
        writer = SummaryWriter(os.path.join(self.params.tb_root, self.params.tag)) #日志保存在 self.params.tb_root 和 self.params.tag 指定的路径下
        return writer

    def save(self, model_name):
        """Save model and buffer.
        Args:
            model_name (str): Name of the model file
        """
        if self.params.save_ckpt: #保存检查点
            save_dir = os.path.join(self.params.ckpt_root, self.params.tag, str(self.params.run_id)) #创建保存目录
            save_model(
                self.model.state_dict(),
                model_name,
                dir=save_dir
                ) #保存模型的状态字典到指定的路径
            if self.buffer is not None: #保存缓冲区到文件
                lg.debug("Saving memory buffer")
                with open(os.path.join(save_dir, f"memory_{self.buffer.n_seen_so_far}.pkl"), 'wb') as memory_file:
                    pickle.dump(self.buffer, memory_file)
    
    def resume(self, model_path=None, buffer_path=None):
        """Resume model and buffer.
        Args:
            model_path (str, optional): Path to the model to load. Defaults to None.
            buffer_path (str, optional): Path to the buffer to load. Defaults to None.
        """
        if model_path is not None: #如果提供了 model_path
            self.model.load_state_dict(torch.load(model_path)) #加载模型的状态字典
        if buffer_path is not None: #如果提供了 buffer_path
            with open(buffer_path, 'rb') as f: #打开文件并加载缓冲区
                self.buffer = pickle.load(f)
            f.close()
        if model_path is None or buffer_path is None: #如果 model_path 或 buffer_path 任一为 None，则抛出警告
            raise Warning("Invalid model and/or buffer state for resume or test argument.")
        torch.cuda.empty_cache() #清空CUDA缓存

    def load_model(self):
        """Load model 加载backbone模型
        Returns:
            untrained torch backbone model 加载一个未训练的 PyTorch 骨干网络模型
        """
        model = resnet.SupConResNet(
            head=self.params.head,
            dim_in=self.params.dim_in,
            dim_int=self.params.dim_int,
            proj_dim=self.params.proj_dim,
            input_channels=self.params.nb_channels,
            instance_norm=self.params.instance_norm,
            nf=self.params.nf
        ) #创建一个 resnet.SupConResNet 模型实例，配置其头、维度、输入通道数等参数
        model = nn.DataParallel(model) #nn.DataParallel 包装模型以支持多GPU训练
        model.to(device) #将模型移至指定的设备（如GPU）
        return model

    def load_optim(self):
        """Load optimizer for training 加载训练的优化器
        Returns:
            torch.optim: torch optimizer
        """
        if self.params.optim == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay) #使用 torch.optim.Adam
        elif self.params.optim == 'SGD':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.params.learning_rate,
                momentum=self.params.momentum,
                weight_decay=self.params.weight_decay
                ) #使用 torch.optim.SGD
        else: 
            raise Warning('Invalid optimizer selected.')
        return optimizer

    def load_scheduler(self):
        raise NotImplementedError

    def load_criterion(self):
        raise NotImplementedError
    
    def train(self, dataloader, task, **kwargs):
        raise NotImplementedError

    def evaluate(self, dataloaders, task_id, **kwargs):
        """Evaluate the model after training ona specified task
        Args:
            dataloaders (Dictonnary of dataloaders): Dictionnary containing every train and test loader for each task
            task_id (int): id of the current task 当前任务的ID
        Returns:
            NCM current average accuracy and average forgetting
        """
        with torch.no_grad():
            self.model.eval() #将模型设置为评估模式

            # Train classifier on labeled data
            step_size = int(self.params.n_classes/self.params.n_tasks)
            if self.params.eval_mem: #是否评估记忆 (self.params.eval_mem)，，获取记忆表示和标签
                mem_representations, mem_labels = self.get_mem_rep_labels(use_proj=self.params.eval_proj)
            else:
                mem_representations, mem_labels = self.sample_rep_label(
                    dataloader=dataloaders['train'],
                    labels_list=self.params.labels_order[:(task_id+1)*step_size],
                    data_per_class=self.params.lab_pc,
                    output_format="torch"
                )
            classifiers = self.init_classifiers() #初始化分类器
            classifiers = self.fit_classifiers(classifiers=classifiers, representations=mem_representations, labels=mem_labels) #拟合分类器从、
            
            accs = {}
            representations = {}
            targets = {}
            # Init representations - labels pairs 计算每个测试任务的表示和目标
            for j in range(task_id + 1):
                test_representation, test_targets = self.encode(dataloaders[f"test{j}"], use_proj=self.params.eval_proj) #提取表示
                representations[f"test{j}"] = test_representation
                targets[f"test{j}"] = test_targets
            # Compute accuracies
            for clf, clf_name in zip(classifiers, self.classifiers_list):#遍历分类器
                accs[clf_name] = [] #空列表记录准确率
                for j in range(task_id + 1): #遍历所有任务
                    acc = accuracy_score(targets[f"test{j}"], clf.predict(representations[f'test{j}'])) #计算准确率
                    accs[clf_name].append(acc)
                    # Wandb logs
                    if not self.params.no_wandb:
                        wandb.log({
                            f"{clf_name}_{j}": acc, #记录每个分类器在每个测试任务上的准确率
                            "task_id": task_id
                        })
            for clf_name in accs:
                for j in range(self.params.n_tasks - task_id - 1): #对于每个分类器，如果还有未执行的任务
                    accs[clf_name].append(np.nan) #填充缺失的任务的准确率为 NaN
                self.results[clf_name].append(accs[clf_name])
            # Compute forgettings
            for clf_name in self.results: #遍历所有分类器的结果
                line = forgetting_line(pd.DataFrame(self.results[clf_name]), task_id=task_id, n_tasks=self.params.n_tasks) #计算遗忘率
                line = line[0].to_numpy().tolist() #将遗忘率结果转换为 NumPy 数组，然后转换为列表
                self.results_forgetting[clf_name].append(line)
            
            self.print_results(task_id) # 调用 print_results 方法打印当前任务的结果

            return np.nanmean(self.results['ncm'][-1]), np.nanmean(self.results_forgetting['ncm'][-1]) #返回最后一个任务的平均准确率和平均遗忘率。

    def evaluate_offline(self, dataloaders, epoch):
        """Evaluate the model on test set at a specified epoch. Train loader is used for training
        a classifier on some random images selected from train set
        Args:
            dataloaders (dict): Dictionnary containing dataloaders 'train' and 'test'
        """
        with torch.no_grad():
            self.model.eval()

            # Train classifier on labeled data
            if self.params.eval_mem:
                mem_representations, mem_labels = self.get_mem_rep_labels() #获取缓冲区的表示、标签
            else:
                mem_representations, mem_labels = self.sample_rep_label(
                    dataloader=dataloaders['train'],
                    labels_list=self.params.labels_order,
                    data_per_class=self.params.lab_pc,
                    output_format="torch"
                )
            classifiers = self.init_classifiers()
            classifiers = self.fit_classifiers(
                classifiers=classifiers,
                representations=mem_representations,
                labels=mem_labels
            ) # 使用记忆表示和标签拟合分类器

            test_representations, test_targets = self.encode(dataloaders['test']) #获取测试数据的表示和目标标签

            for clf, clf_name in zip(classifiers, self.classifiers_list):
                acc = accuracy_score(test_targets, clf.predict(test_representations))
                self.results[clf_name].append(acc) #计算每个分类器的准确率
            
            if not ((epoch + 1) % 5):
                self.save(model_name=f"{self.params.tag}_{epoch}.pth")
        
        lg.info(f"ACCURACY {self.results['ncm'][-1]}")
        return self.results['ncm'][-1] #最后一个任务的准确率

    def sample_rep_label(self, dataloader, labels_list, data_per_class, output_format='torch', return_proj=False, return_raw=False):
        """Sample labeled data randomly from dataloader. Returns data representation and labels such that every 
        label is equally present.  从数据加载器中随机抽取每个类别固定数量的数据，以确保每个类别的样本数量相等
        Exemple : labels_list = [0,1], data_per_class = 3
        returns labels [1,1,0,1,0,0] and reps of shape (6, rep_dim)
        Args:
            dataloader (torch dataloader):      Dataloader for iterating over labeled data 
            labels_list (list):                 List of requested labels
            data_per_class (_type_):            Number of data per class
            output_format (str, optional):      Datatype fo the output image. Defaults to 'torch'
            return_proj (bool, optional):       Whether to return the projection of the representation of the model
            return_raw (bool, optional)         Whether to return the raw data instead of the projection
        Returns:
            out_rep   (numpy array or torch tensor):     List of representations
            out_labels (numpy array):                     List of labels
        """
        #初始化表示（out_rep）和标签（out_labels）的列表
        out_rep = []
        out_labels = []
        for sample in dataloader:
            imgs, labels = sample[0], sample[1]
            labels = np.array(labels)
            # Loop over batch
            for img, label in zip(imgs, labels):
                if np.isin(label, labels_list): #检查标签是否在请求的标签列表中
                    nb_data_per_label = np.bincount(out_labels, minlength=np.max(labels_list)+1) #np.bincount 函数来计算 out_labels 数组中每个唯一标签的出现次数
                    if nb_data_per_label[int(label)] < data_per_class: #根据return_raw或return_proj添加表示和标签
                        if return_raw:
                            out_rep.append(img.cpu().numpy())
                        else:
                            i = 1 if return_proj else 0
                            out_rep.append(np.array(self.model(self.transform_test(img.to(self.device).unsqueeze(0)))[i][0].cpu()))
                        out_labels.append(label)
                    if len(out_labels) >= (len(labels_list) * data_per_class):
                        if output_format == "numpy":
                            if return_raw:
                                temp = np.array(out_rep)
                                out_rep = np.ascontiguousarray(np.swapaxes(np.swapaxes(temp, 1, 2), 2, 3)*255)
                            out_labels = np.array(out_labels)
                        elif output_format == "torch":
                            out_rep = torch.Tensor(np.array(out_rep))
                            out_labels = torch.Tensor(np.array(out_labels))
                        return out_rep, out_labels #检查是否已经收集了足够的数据，如果是，则根据output_format返回数据

    def encode(self, dataloader, use_proj=False, nbatches=-1):
        """Compute representations - labels pairs for a certain number of batches of dataloader.
            Not really optimized. 计算数据加载器中一定数量批处理的数据表示和标签
        Args:
            dataloader (torch dataloader): dataloader to encode
            nbatches (int, optional): Number of batches to encode. Use -1 for all. Defaults to -1.
        Returns:
            representations - labels pairs
        """
        i = 0
        with torch.no_grad():
            for sample in dataloader: #要编码的数据加载器
                if nbatches != -1 and i >= nbatches:
                    break
                inputs = sample[0]
                labels = sample[1]
                
                inputs = inputs.to(self.device)
                if use_proj: #是否使用模型的投影层
                    _, features = self.model(self.transform_test(inputs))
                else: #使用模型基础层
                    features, _ = self.model(self.transform_test(inputs))
                if i == 0: #收集所有批次的特征和标签。+
                    all_labels = labels.cpu().numpy()
                    all_feat = features.cpu().numpy()
                else:
                    all_labels = np.hstack([all_labels, labels.cpu().numpy()])
                    all_feat = np.vstack([all_feat, features.cpu().numpy()])
                i += 1
        return all_feat, all_labels

    def print_results(self, task_id):
        """Log learner's results as accuracy and forgetting打印结果
        Example : 
                root - INFO - --------------------FORGETTING--------------------
                root - INFO - ncm     0.0000   nan      nan      nan      nan      0.0000
                root - INFO - ncm     0.2885   0.0000   nan      nan      nan      0.2885
                root - INFO - ncm     0.2935   0.2225   0.0000   nan      nan      0.2580
                root - INFO - ncm     0.4615   0.3190   0.0370   0.0000   nan      0.2725
                root - INFO - ncm     0.5815   0.2155   0.1795   0.0250   0.0000   0.2504
                root - INFO - --------------------ACCURACY--------------------
                root - INFO - ncm     0.7750   nan      nan      nan      nan      0.7750
                root - INFO - ncm     0.4865   0.5260   nan      nan      nan      0.5062
                root - INFO - ncm     0.4815   0.3035   0.5150   nan      nan      0.4333
                root - INFO - ncm     0.3135   0.2070   0.4780   0.2875   nan      0.3215
                root - INFO - ncm     0.1935   0.3105   0.3355   0.2625   0.3045   0.2813
        Args:
            task_id (int):      Number of the task where the learner was just trained on.
        """
        n_dashes = 20
        pad_size = 8
        lg.info('-' * n_dashes + f"TASK {task_id + 1} / {self.params.n_tasks}" + '-' * n_dashes)
        
        lg.info('-' * n_dashes + "FORGETTING" + '-' * n_dashes)
        for clf_name in self.results:
            for line in self.results_forgetting[clf_name]:
                lg.info(f'{clf_name}'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line) + f" {np.nanmean(line):.4f}")
        
        lg.info('-' * n_dashes + "ACCURACY" + '-' * n_dashes)        
        for clf_name in self.results:
            for line in self.results[clf_name]:
                lg.info(f'{clf_name}'.ljust(pad_size) + ' '.join(f'{value:.4f}'.ljust(pad_size) for value in line) + f" {np.nanmean(line):.4f}")

    def save_results(self):
        """Save results as csv.
        """
        results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}") #定义结果保存目录
        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir): #如果目录不存在，则创建目录
            os.makedirs(results_dir, exist_ok=True)
        
        df_avg = pd.DataFrame() #存储平均结果
        cols = [f'task {i}' for i in range(self.params.n_tasks)] #构造列名列表
        # Loop over classifiers results
        for clf_name in self.results:
        # Each classifier has a value for every task. NaN if future task遍历分类器的结果
            df_clf = pd.DataFrame(self.results[clf_name], columns=cols)
            # Average accuracy over all tasks with value not NaN
            df_avg[clf_name] = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, f'{clf_name}.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg.csv'), index=False)
        #将平均结果保存为 CSV 文件
        df_avg = pd.DataFrame()
        lg.info(f"Saving forgetting results in : {results_dir}")
        cols = [f'task {i}' for i in range(self.params.n_tasks)]
        for clf_name in self.results_forgetting:
            df_clf = pd.DataFrame(self.results_forgetting[clf_name], columns=cols)
            df_avg[clf_name] = df_clf.mean(axis=1)
            df_clf.to_csv(os.path.join(results_dir, f'{clf_name}_forgetting.csv'), index=False)
        df_avg.to_csv(os.path.join(results_dir, 'avg_forgetting.csv'), index=False)

        self.save_parameters() #保存参数
    
    def save_results_offline(self):
        """Save results as csv for offline training
        """
        #定义结果保存目录 results_dir
        results_dir = os.path.join(self.params.results_root, self.params.tag, f"run{self.params.seed}")

        lg.info(f"Saving accuracy results in : {results_dir}")
        if not os.path.exists(results_dir): #创建目录（如果不存在
            os.makedirs(results_dir, exist_ok=True)
        df_all = pd.DataFrame() #初始化一个空的 DataFrame df_all
        for clf_name in self.results: #遍历 self.results 中的每个分类器名称和结果
            df_all[clf_name] = pd.DataFrame(self.results[clf_name]) #将每个分类器的结果添加到 df_all
        df_all.to_csv(os.path.join(results_dir, 'acc.csv'), index=False) #保存为一个 CSV 文件 'acc.csv'
        self.save_parameters() # 保存训练过程中使用的参数

    def save_parameters(self):
        """Save parameters used for training  保存训练过程中使用的参数
        """
        filename = os.path.join(self.params.results_root, self.params.tag, f'run{self.params.seed}/params_used.json') #构造参数文件的路径
        with open(filename, 'w') as f:
            json.dump(self.params.__dict__, f, indent=2) #将参数对象转换为字典并保存为 JSON 文件

    def get_mem_rep_labels(self, eval=True, use_proj=False):
        """Compute every representation -labels pairs from memory 从内存中计算所有表示-标签对
        Args:
            eval (bool, optional): Whether to turn the mdoel in evaluation mode. Defaults to True.
        Returns:
            representation - labels pairs
        """
        if eval: self.model.eval() #eval: 是否将模型设置为评估模式
        mem_imgs, mem_labels = self.buffer.get_all() #获取内存中的所有图像和标签
        batch_s = 10 #将图像分批处理并转换为模型的表示
        n_batch = len(mem_imgs) // batch_s #总批次数
        all_reps = []
        for i in range(n_batch): #循环遍历批次的数量
            mem_imgs_b = mem_imgs[i*batch_s:(i+1)*batch_s].to(self.device) #从mem_imgs（可能是一个包含所有图像的列表或张量）中取出一个批次的图像
            mem_imgs_b = self.transform_test(mem_imgs_b) #对当前批次的图像进行测试时的转换
            if use_proj:
                _, mem_representations_b = self.model(mem_imgs_b)
            else:
                mem_representations_b, _ = self.model(mem_imgs_b)
            all_reps.append(mem_representations_b)
        mem_representations = torch.cat(all_reps, dim=0) #将当前批次的图像表示添加到all_reps列表中
        return mem_representations, mem_labels  #返回图像表示和对应的标签

    def init_classifiers(self): # 初始化用于表示迁移学习的分类器
        """Initiliaze every classifier for representation transfer learning
        Returns:
            List of initialized classifiers
        """
        # logreg = LogisticRegression(random_state=self.params.seed)
        # knn = KNeighborsClassifier(3)
        # linear = MLPClassifier(hidden_layer_sizes=(200), activation='identity', max_iter=500, random_state=self.params.seed)
        # svm = SVC()
        ncm = NearestCentroid()  #最近质心分类器（NearestCentroid）的实例
        # return [logreg, knn, linear, svm, ncm]
        return [ncm]
    
    @ignore_warnings(category=ConvergenceWarning)
    def fit_classifiers(self, classifiers, representations, labels):
        """Fit every classifiers on representation - labels pairs 在表示-标签对上训练分类器
        Args:
            classifiers : List of sklearn classifiers分类器列表
            representations (torch.Tensor): data representations 数据表示
            labels (torch.Tensor): data labels
        Returns:
            List of trained classifiers
        """
        for clf in classifiers:
            clf.fit(representations.cpu(), labels)
        return classifiers

    def after_eval(self, **kwargs):
        pass

    def before_eval(self, **kwargs): #在模型评估之前
        if self.params.review:
            model_backup = deepcopy(self.model) #使用 deepcopy 函数创建模型的一个深拷贝
            mem_x, mem_y = self.buffer.get_all() #从缓冲区（buffer）中获取所有的数据

            for e in range(self.mem_epochs):
                idx = np.random.permutation(len(mem_x)).tolist() #生成一个随机排列的索引列表
                mem_x = mem_x[idx]
                mem_y = mem_y[idx]
                self.model = self.model.train() #将模型设置为训练模式
                batch_size = 64 #设置批次大小为64
                
                for j in range(len(mem_y) // batch_size): #遍历批次
                    batch_x = mem_x[batch_size * j:batch_size * (j + 1)].to(self.device)  #从特征数据中取出当前批次的数据，并将其移动到指定的设备上
                    aug1, aug2 = self.transform_train(batch_x), self.transform_train(batch_x) #对当前批次的数据进行两次数据增强，得到 aug1 和 aug2
                    _, p1 = self.model(aug1)
                    _, p2 = self.model(aug2)
                    p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1) #将 p1 和 p2 在第二个维度上拼接起来
                    loss = self.criterion(p, labels=mem_y[batch_size * j:batch_size * (j + 1)].to(self.device)) #计算损失函数，将拼接后的结果 p 和当前批次的标签进行比较
                    self.loss = loss.item()

                    # Optim
                    self.optim.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0) #对梯度进行裁剪，防止梯度爆炸
                    self.optim.step()

            self.model = model_backup #将模型恢复到原始状态

    def augment(self, combined_x, mem_x, batch_x, **kwargs): #数据增强
        with torch.no_grad():
            augmentations = [] #用于存储增强后的数据
            for key in self.tf_seq: #遍历 self.tf_seq 字典中的所有键
                if 'aug' in key:
                    augmentations.append(self.tf_seq[key](combined_x))
                else:
                    n_repeat = self.get_n_repeat(key) #获取当前键对应的重复次数
                    batch1 = combined_x # 将 combined_x 赋值给 batch1
                    batch2 = batch_x if self.params.daa else combined_x #根据 self.params.daa 的值决定 batch2 是 batch_x 还是 combined_x
                    for _ in range(n_repeat): #根据重复次数进行循环
                        augmentations.append(self.tf_seq[key](batch1, batch2, model=self.model)) # batch1 和 batch2 进行数据增强操作，并将结果添加到 augmentations 列表中
            augmentations.append(combined_x)
            return augmentations

    def train_inc(self, **kwargs):
        raise NotImplementedError

    def train_blurry(self, **kwargs):
        raise NotImplementedError
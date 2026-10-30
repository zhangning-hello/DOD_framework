import os
import torch
import pandas as pd
import numpy as np
import sys
import logging as lg
import datetime as dt
import random as r
import ssl
import wandb
ssl._create_default_https_context = ssl._create_unverified_context
from src.utils.utils import maybe_cuda, AverageMeter
from src.utils.data import get_loaders
from src.utils import name_match
from src.utils.early_stopping import EarlyStopper
from config.parser import Parser
import warnings
import copy
warnings.filterwarnings("ignore")
from argparse import Namespace

old_labels = []
new_labels = []
lbl_inv_map = {}
class_task_map = {}
task_seen = 0
def before_train(x_train, y_train):
    global old_labels
    global new_labels
    global task_seen
    global class_task_map
    global lbl_inv_map
    phase_new_labels = list(set(y_train.tolist()))  # 从当前训练集y_train中提取所有独特的标签
    overlapped_labels = list(set(phase_new_labels).intersection(
        set(old_labels)))  # recalculate overlapped and non-overlapped labels after each phase 计算新标签phase_new_labels和老标签self.old_labels的交集，即两个集合中共有的元素
    new_non_overlap_labels = list(set(phase_new_labels).difference(set(overlapped_labels)))  # 计算新标签中不在老标签集合中的元素，即非重叠的新标签
    new_labels += phase_new_labels  # 新元素标签
    for i, lbl in enumerate(new_non_overlap_labels):
        lbl_inv_map[lbl] = len(old_labels) + i  # 对于每个非重叠的新标签，将其添加到self.lbl_inv_map字典中，并分配一个唯一的索引
    for i in phase_new_labels:
        class_task_map[i] = task_seen  # 每个标签都与它首次出现的任务相关联
    overlapped_labels = list(set(new_labels).intersection(set(old_labels)))
    new_non_overlap_labels = list(set(new_labels).difference(set(overlapped_labels)))
    cumm_new_labels = old_labels + new_non_overlap_labels
    return overlapped_labels, new_non_overlap_labels, cumm_new_labels


def after_train():
    global old_labels
    global new_labels
    global task_seen
    overlapped_labels = list(set(new_labels).intersection(
        set(old_labels)))  # recalculate overlapped and non-overlapped labels after each phase
    new_non_overlap_labels = list(set(new_labels).difference(set(overlapped_labels)))
    old_labels += new_non_overlap_labels  # add only non-overlapped labels to old labels after each phase
    new_labels_zombie = copy.deepcopy(new_labels)  # for zombie copy, should i include overlapped labels?
    new_labels.clear()
    task_seen += 1



def main():
    runs_accs = []
    runs_fgts = [] #用于存储每次运行的准确率和遗忘率
    
    parser = Parser() #解析命令行参数
    args = parser.parse() #参数
    # 设置日志记录的格式和处理器
    cf = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
    ch = lg.StreamHandler()
    
    for run_id in range(args.start_seed, args.start_seed + args.n_runs): # 循环执行指定次数的运行
        # Re-parse tag. Useful when using multiple runs.
        args = parser.parse() #重新解析参数
        args.run_id = run_id

        if args.sweep: # 如果进行参数扫描，则初始化 Weights & Biases (wandb)
            wandb.init() #初始化 wandb
            for key in wandb.config.keys(): #从 wandb 配置中获取参数
                setattr(args, key, wandb.config[key])
            parser.check_args()# 检查参数
            for key in wandb.config.keys():
                setattr(args, key, wandb.config[key])

        # Seed initilization
        if args.n_runs > 1: args.seed = run_id
        np.random.seed(args.seed) #设置随机种子以确保实验的可重复性
        r.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Learner
        if args.learner is not None: #如果指定了学习器，则创建学习器实例
            learner = name_match.learners[args.learner](args)
            if args.resume: learner.resume(args.model_state, args.buffer_state) #从先前的状态恢复学习器
        else:
            raise Warning("Please select the desired learner.")
        
        # logs
        # Define logger and timstamp
        logfile = f'{args.tag}.log'  #定义日志文件的名称
        if not os.path.exists(args.logs_root): os.mkdir(args.logs_root)

        ff = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s') #设置文件日志记录的格式

        logger = lg.getLogger() #获取日志记录器
        fh = lg.FileHandler(os.path.join(args.logs_root, logfile)) #设置文件日志处理器
        ch.setFormatter(cf)
        fh.setFormatter(ff)
        logger.addHandler(fh)
        logger.addHandler(ch) #将处理器添加到日志记录器
        if args.verbose:
            logger.setLevel(lg.DEBUG)
            logger.warning("Running in VERBOSE MODE.")
        else:
            logger.setLevel(lg.INFO)
        # 记录日志信息
        lg.info("=" * 60)
        lg.info("=" * 20 + f"RUN N°{run_id} SEED {args.seed}" + "=" * 20)
        lg.info("=" * 60)        
        lg.info("Parameters used for this training")
        lg.info("=" * 20)
        lg.info(args)

        # Dataloaders
        dataloaders = get_loaders(args) # 获取数据加载器


        # wandb initilization 如果没有禁用 wandb 并且不是参数扫描，则初始化 wandb
        if not args.no_wandb and not args.sweep:
            wandb.init(
                project=f"{args.learner}",
                config=args.__dict__
            )
            
        # Training
        # Class incremental training
        if args.training_type == 'inc': #增量学习
            for task_id in range(args.n_tasks):#每个任务
                #train_loader = dataloaders['train']
                #test_loader = dataloaders['test']
                #x_train, y_train = [], []
                #for images, labels,_ in train_loader:
                #    x_train.append(images)
                #    y_train.append(labels)
                #x_train = torch.cat(x_train, dim=0)
                #y_train = torch.cat(y_train, dim=0)
                #before_train(x_train, y_train)

                #acc_batch = AverageMeter()
                #acc_batch_vm1 = AverageMeter()
                #acc_batch_vm2 = AverageMeter()
                #overlapped_labels, new_non_overlap_labels, cumm_new_labels = before_train(x_train, y_train)
                for e in range(args.epochs): #循环执行指定次数的训练周期（epochs）
                    task_name = f"train{task_id}"
                    if args.train:
                        learner.train(
                            dataloader=dataloaders[task_name],
                            task_name=task_name,
                            task_id=task_id,
                            dataloaders=dataloaders
                            ) #训练学习器
                    else:
                        model_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/ckpt_train{task_id}.pth")  #构建模型状态文件的路径
                        mem_idx = int(len(dataloaders['train']) * args.batch_size / args.n_tasks) * (task_id + 1) #计算记忆索引
                        buffer_state = os.path.join(args.ckpt_root, f"{args.tag}/{args.run_id}/memory_{mem_idx}.pkl") #构建缓冲区状态文件的路径
                        learner.resume(model_state, buffer_state) #恢复模型和缓冲区状态
                #after_train()
                    learner.before_eval() #在评估之前执行一些准备工作
                    avg_acc, avg_fgt = learner.evaluate(dataloaders, task_id) #调用学习器的 evaluate 方法评估当前任务的平均准确率和遗忘率
                    if not args.no_wandb:
                        wandb.log({
                            "avg_acc": avg_acc,
                            "avg_fgt": avg_fgt,
                            "task_id": task_id
                        }) #使用 wandb 记录平均准确率、遗忘率和任务ID
                        if args.wandb_watch:
                            wandb.watch(learner.model, learner.criterion, log="all", log_freq=1) #使用 wandb 监视模型和损失函数
                    learner.after_eval() # 在评估之后执行一些后续工作
                #after_train()
            learner.save_results() #保存当前任务的结果

        # Training with blurry boundaries
        elif args.training_type == 'blurry': #模糊
            learner.train(dataloaders['train'])  #调用学习器的 train 方法进行训练，使用指定的数据加载器
            avg_acc = learner.evaluate_offline(dataloaders, epoch=1) #使用学习器的 evaluate_offline 方法评估模型的准确率
            avg_fgt = 0 #设置遗忘率（avg_fgt）为0
            if not args.no_wandb:
                wandb.log({
                        "avg_acc": avg_acc,
                    })
                if args.wandb_watch:
                    wandb.watch(learner.model, learner.criterion, log="all", log_freq=1)
            learner.save_results_offline()
        # Uniform training (offline)
        elif args.training_type == 'uni': #统一模式
            # early_stopper = EarlyStopper(patience=args.es_patience, min_delta=args.es_delta)
            for e in range(args.epochs):
                learner.train(dataloaders['train'], epoch=e)
                avg_acc = learner.evaluate_offline(dataloaders, epoch=e) #使用学习器的 evaluate_offline 方法评估模型的准确率
                avg_fgt = 0 #设置遗忘率（avg_fgt）为0
                # if early_stopper.early_stop(avg_acc):
                #     break
                if not args.no_wandb:
                    wandb.log({
                            "Accuracy": avg_acc,
                            "loss": learner.loss
                        })
            learner.save_results_offline()
        runs_accs.append(avg_acc) #当前训练的平均准确率添加到 runs_accs 列表中
        runs_fgts.append(avg_fgt) #将当前训练的遗忘率添加到 runs_fgts 列表中
        if not args.no_wandb:
            wandb.finish() #结束当前的 wandb 运行
    

    # Save runs accs and forgettings
    if args.n_runs > 1:
        df_acc = pd.DataFrame(runs_accs)
        df_fgt = pd.DataFrame(runs_fgts)
        results_dir = os.path.join(args.results_root, args.tag)
        lg.info(f"Results for the aggregated runs are save in : {results_dir}")
        df_acc.to_csv(os.path.join(results_dir, 'runs_accs.csv'), index=False)
        df_fgt.to_csv(os.path.join(results_dir, 'runs_fgts.csv'), index=False)

    # Exits the program
    sys.exit(0)

threshold_values = [0.5, 0.55, 0.6, 0.65, 0.7,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.25,1.3,1.35,1.45,1.55,1.65,1.7,1.75,1.8,1.85,1.9,1.95,2.0]
def run_training_and_evaluate(threshold):
    args = Namespace(**your_default_args)
    args.threshold = threshold
    learner = DERppscrLearner(args)
    # ... 运行训练和评估 ...
    return validation_performance


if __name__ == '__main__':
    main()
  
import os
import math
import argparse
import sys
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
import json
import pandas as pd
import torch.nn as nn
import numpy as np
from PIL import Image

from Test11_efficientnetV2.model import efficientnetv2_s as create_model
from utils import  read_split_data,plot_data_loader_image,evaluate,train_one_epoch
from options import Options
from torch.utils.data import Dataset
from candock_master.creatnet import CreatNet
import candock_master.util as util
#------------------------------------nirs-----------------------------------#
from datetime import datetime
from functools import partial
from PIL import Image
from Preprocessing.Preprocessing import *
import numpy as np
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torchvision.models import resnet
from tqdm import tqdm
import json
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import DatasetFolder
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt
from options import Options
import candock_master.transformer as transformer
import candock_master.statistics as statistics
import candock_master.heatmap as heatmap

class DynamicSparseRouter(nn.Module):
    """改进的动态稀疏路由模块（支持双输入）"""
    def __init__(self, channels, sparsity_ratio=0.5):
        super().__init__()
        self.channels = channels
        self.sparsity_ratio = sparsity_ratio
        
        # 双输入门控参数
        self.gate_x = nn.Conv1d(channels, channels, 1)
        self.gate_y = nn.Conv1d(channels, channels, 1)
        
        # 动态路由参数
        self.routing_temp = 2.0  # 初始温度参数
        
    def forward(self, x, y):
        batch_size, channels, length = x.size()
        
        # 分别计算门控权重
        gate_x = torch.sigmoid(self.gate_x(x))  # [B, C, L]
        gate_y = torch.sigmoid(self.gate_y(y))  # [B, C, L]
        
        # 合并门控信号
        combined_gate = gate_x + gate_y  # [B, C, L]
        
        # 动态稀疏化过程
        with torch.no_grad():
            # 计算联合重要性指标（L1范数）
            importance = torch.sum(torch.abs(combined_gate), dim=1, keepdim=True)  # [B, 1, L]
            threshold = torch.quantile(importance, self.sparsity_ratio, dim=-1, keepdim=True)
            
        # 生成稀疏掩码
        sparse_mask = (combined_gate >= threshold).float()  # [B, 1, L]
        
        # 路由选择（直通估计器）
        selected_x = torch.bernoulli(gate_x * sparse_mask) / (gate_x * sparse_mask + 1e-7)
        selected_y = torch.bernoulli(gate_y * sparse_mask) / (gate_y * sparse_mask + 1e-7)
        
        return x * selected_x, y * selected_y, selected_x.mean(dim=-1) + selected_y.mean(dim=-1)

class DynamicSparseCrossAttention(nn.Module):
    """支持双输入的动态稀疏交叉注意力模块"""
    def __init__(self, in_channels=1, reduction_ratio=16, sparsity_ratio=0.7):
        super().__init__()
        self.channels = in_channels
        
        # 稀疏卷积模块
        self.conv_x = nn.Conv1d(in_channels, in_channels//reduction_ratio, 3, padding=1)
        self.conv_y = nn.Conv1d(in_channels, in_channels//reduction_ratio, 3, padding=1)
        
        # 动态稀疏路由层
        self.dynamic_router = DynamicSparseRouter(
            channels=in_channels//reduction_ratio,
            sparsity_ratio=sparsity_ratio
        )
        
        # 交叉注意力计算模块
        self.cross_att = nn.MultiheadAttention(
            embed_dim=in_channels//reduction_ratio,
            num_heads=4,
            batch_first=True
        )

    def forward(self, x, y):
        """
        双输入动态稀疏交叉注意力
        Args:
            x: Input tensor 1 [B, C, L]
            y: Input tensor 2 [B, C, L]
            
        Returns:
            fused_output: 融合后的输出 [B, C, L]
            sparse_rates: 稀疏率 [B]
        """
        # 特征压缩
        x_red = F.gelu(self.conv_x(x))  # [B, C_red, L]
        y_red = F.gelu(self.conv_y(y))  # [B, C_red, L]
        
        # 动态稀疏处理
        x_sparse, y_sparse, sparse_rate = self.dynamic_router(x_red, y_red)
        
        # 交叉注意力计算
        attn_output, _ = self.cross_att(
            x_sparse.transpose(1,2),  # query: [B, L, C_red]
            y_sparse.transpose(1,2),  # key: [B, L, C_red]
            y_sparse.transpose(1,2)   # value: [B, L, C_red]
        )
        
        # 特征融合
        fused_output = torch.cat([
            x_sparse + attn_output.transpose(1,2),
            y_sparse - attn_output.transpose(1,2)
        ], dim=1)  # [B, 2*C_red, L]
        
        return fused_output, sparse_rate

def image(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_data_path, train_data_label, val_data_path, val_data_label = read_split_data(args.data_path)

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.RandomResizedCrop(img_size[num_model][0]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(data_path=train_data_path,
                              data_class=train_data_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(data_path=val_data_path,
                            data_class=val_data_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    # 如果存在预训练权重则载入
    model = create_model(num_classes=args.num_classes).to(device)
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # 是否冻结权重
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    return model,optimizer,optimizer,device,train_loader,val_loader
    
#----------------------------------nirs-------------------------------
class MyDataSet(Dataset):
    def __init__(self, data_path, data_class, transform=None):
        self.data_path = data_path
        self.data_class = data_class
        self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        data_item = self.data_path[idx]
        if len(data_item)!=2:
        	 raise ValueError(f"Expected a tuple of length 2, but got {len(data_item)} items: {data_item}")
        img_path,spectrum = data_item
        label = self.data_class[idx]

        # 读取图像
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # 处理光谱数据  
        spectrum = np.expand_dims(spectrum, axis=0)  # 增加一个通道维度
        spectrum = Preprocessing("SNV",spectrum)
        spectrum = Preprocessing("D1",spectrum)
        spectrum = Preprocessing("SG",spectrum)
        spectrum = spectrum[:,3151:]
        spectrum = np.squeeze(spectrum)  # 移除增加的维度

        return image, spectrum, label

    @staticmethod
    def collate_fn(batch):
        images, spectra, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        spectra=np.array(spectra)
        spectra = torch.tensor(spectra, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        return images, spectra, labels


    
def adjust_learning_rate(optimizer, epoch, opt):

    """Decay the learning rate based on schedule"""

    lr = opt.nirlr

    if opt.cos:  # cosine lr schedule

        lr *= 0.5 * (1. + math.cos(math.pi * epoch / opt.nirepochs))
        #print("__________cos___________")
	
    else:  # stepwise lr schedule

        for milestone in opt.schedule:

            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:

        param_group['lr'] = lr


#mlp
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

 
       
        

if __name__ == '__main__':
    #--------------------------------image_parameter---------------------------#
    opt = Options().getparse()
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model,optimizer,optimizer,device,train_loader,val_loader = image(opt)
    cross_model = DynamicSparseCrossAttention(in_channels=128)
    #---------------------------------cig---------------------------------#
    nirnum_classes=opt.nirnum_classes
    net=CreatNet(opt.model_name)
    util.show_paramsnumber(net)
    
    tb_writer = SummaryWriter()
    print(net)
    if not opt.no_cuda:
        print("-------------------------------")
        net.cuda()
    if opt.pretrained:
        net.load_state_dict(torch.load('./checkpoints/pretrained/'+opt.dataset_name+'/'+opt.model_name+'.pth'))
    if opt.continue_train:
        net.load_state_dict(torch.load('./checkpoints/last.pth'))
    if not opt.no_cudnn:
        torch.backends.cudnn.benchmark = True
    
    # 初始化alpha和beta为可学习参数
    alpha = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
    beta = nn.Parameter(torch.tensor(0.5).to(device), requires_grad=True)
    
    mlp_model = MLP(input_dim=128, hidden_dim=64, output_dim=6).cuda()
    
    pg = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD((list.net.parameters())+pg, lr=opt.lr, momentum=0.9, weight_decay=1E-4)
    optimizer = torch.optim.Adam(list(net.parameters())+pg+list(mlp_model.parameters())+[alpha,beta],lr=opt.lr,  weight_decay=1E-4)
    criterion = nn.CrossEntropyLoss()
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
#----------------------------------------train-------------------------------------#
    print('begin to train ...')
    #final_confusion_mat = np.zeros((num_classes,num_classes), dtype=int)
    plot_result={'train':[1.],'test':[1.]}
    #confusion_mats = []
    

    for epoch in range(opt.epochs):
    #(model,net, optimizer, imgdata_loader,nirdata_loader, device, epoch,opt):
        train_loss, train_acc  = train_one_epoch(model=model,net=net ,mlp_model=mlp_model,cross_model, optimizer=optimizer, data_loader=train_loader, device=device, epoch=epoch,opt=opt,alpha=alpha,beta=beta)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,net=net ,mlp_model=mlp_model, data_loader=val_loader, device=device, epoch=epoch,alpha=alpha,beta=beta)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    
    

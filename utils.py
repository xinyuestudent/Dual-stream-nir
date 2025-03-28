import os
import sys
import json
import pickle
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt
import pandas as pd


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_data_path = []  # 存储训练集的所有图片路径
    train_data_label = []  # 存储训练集图片对应索引信息
    
    val_data_path = []  # 存储验证集的所有图片路径
    val_data_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        
        #===========================================================================
        # 读取光谱数据
        spectra_path = os.path.join("D:/文件/烟草研究/multimodel1/pines.csv")
        spectra = pd.read_csv(spectra_path, header=None).values
        #print(spectra)
        assert len(images) == len(spectra), "Mismatch between number of images and spectra in class {}".format(cla)
        #===========================================================================
        
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_samples = random.sample(list(zip(images, spectra)), k=int(len(images) * val_rate))
        #print(val_samples)
        for img_path , spectrum in zip (images , spectra):
            spectrum_tuple = tuple(spectrum)
            if any((v_img_path, tuple(v_spectrum)) == (img_path, spectrum_tuple) for v_img_path, v_spectrum in val_samples):  # 如果该路径在采样的验证集样本中则存入验证集
                #print("val")
                #print(spectrum)
            #if (img_path, spectrum) in val_samples:  # 如果该路径在采样的验证集样本中则存入验证集
                val_data_path.append((img_path,spectrum))
                val_data_label.append(image_class)
            else:  # 否则存入训练集
                #print("train")
                #print(spectrum)
                train_data_path.append((img_path,spectrum))
                train_data_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_data_path)))
    print("{} images for validation.".format(len(val_data_path)))
    #print()
    assert len(train_data_path) > 0, "number of training images must greater than 0."
    assert len(val_data_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(data_classes)), every_class_num, align='center')
        plt.xticks(range(len(data_classes)), data_classes)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('class distribution')
        plt.show()

    return train_data_path, train_data_label, val_data_path, val_data_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


#-----------------------------train_fuction-----------------------   
def train_one_epoch(model,net,mlp_model,cross_model, optimizer, data_loader, device, epoch,opt,alpha,beta):
    model.train()
    mlp_model.train()
    net.train()
    cross_model.train()
	
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_labels = []
    all_preds = []
    
    for step, data in enumerate(data_loader):
        #print(data[1][0].shape)
        images, spectra,labels = data
        sample_num += images.shape[0]
        #print(images.shape) (32,3,300,300)
        images = images.to(device)
        spectra=spectra.to(device).float()
        labels=labels.to(device)
        #print(spectra) #(32,0)
        spectra = spectra.reshape(spectra.shape[0],1,spectra.shape[1]).float()
        
        #print(images.shape)
        img_ftr = model(images)
        spc_ftr = net(spectra)
        
        #feature = net(spectra.cuda(non_blocking=True))
        #print(feature.shape) #(32,128)
        #print("===========================")
        #print("===========================")
        #(32,256)
		################
		#
		#动态稀疏交叉注意力
		#需要加一个模块
		#
		################
        combined_features = (alpha/(alpha+beta)) * img_ftr + (beta/(alpha+beta)) *spc_ftr
		#combined_features = cross_model(img_ftr,spc_ftr)
        #print(alpha,beta)
        #print(combined_features.shape)
        #label=alpha * labels.to(device) + beta *target.to(device)
        outputs = mlp_model(combined_features).cuda()
        #print(labels.shape)
        loss = loss_function(outputs,labels)
        
        pred_classes = torch.max(outputs, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels).sum().item()
        accu_loss += loss.detach()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(pred_classes.cpu().numpy())
        
        # 反向传播和参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        data_loader.desc = f"[train epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}"
        #print(alpha,beta)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
        #imgdata_loader.desc = f"[train epoch {epoch}] loss: {accu_loss / (step + 1):.3f}, acc: {accu_num / sample_num:.3f}"

    #return accu_loss.item() / (step + 1), accu_num.item() / sample_num
    return accu_loss/ len(data_loader), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model,net,mlp_model, data_loader, device, epoch,alpha,beta):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()
    net.eval()
    mlp_model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    all_labels = []
    all_preds = []
    for step, data in enumerate(data_loader):
        images, spectra,labels = data
        sample_num += images.shape[0]
        
        
        # 将数据移动到设备上
        images = images.to(device)
        spectra = spectra.to(device).float()
        labels = labels.to(device)
        pred = model(images.to(device))
        #print("===================")
        #print(spectra.shape)
        #print(image.shape)
        #print("===================")
        spectra = spectra.reshape(spectra.shape[0],1,spectra.shape[1]).float()
        
        with torch.no_grad():
            img_ftr = model(images)
            spc_ftr = net(spectra)

            combined_features = alpha * img_ftr + beta * spc_ftr
            outputs = mlp_model(combined_features).cuda()

            loss = loss_function(outputs, labels)

            pred_classes = torch.max(outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, labels).sum().item()
            accu_loss += loss.detach()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred_classes.cpu().numpy())


            data_loader.desc = f"[valid epoch {epoch}] loss: {accu_loss.item() / (step + 1):.3f}, acc: {accu_num.item() / sample_num:.3f}"

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
        #data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                       accu_loss.item() / (step + 1),
         #                                                                      accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

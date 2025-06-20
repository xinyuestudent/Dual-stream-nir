## A Lightweight Two-Stream Model for Real-Time Near-Infrared Spectroscopy Classification on Portable Devices Using Gramian Angular Field and Dynamic Sparse Cross Attention

### Xinyue Wang, Xiangdong  Chen, Jun Jiang, Ronggao  Gong, Ya Deng

#### Required Environment

```python
torch
torchvision
pandas
numpy
```

### Abstract

Near-infrared (NIR) spectroscopy is a vital tool for rapid, non-invasive analysis in food safety, agriculture, and environmental monitoring. Portable NIR spectrometers allow for on-site assessments. However, the high dimensionality of NIR spectral data can complicate accurate classification, especially in resource-limited settings. Conventional techniques such as Partial Least Squares Discriminant Analysis (PLS-DA) and machine learning methods including Support Vector Machines (SVM) encounter challenges with overfitting and generalization in high-dimensional datasets. To address these issues, a lightweight two-stream model is proposed, which integrates NIR spectral data with Gram Angular Field (GAF) matrix images to accentuate intricate relationships frequently overlooked by conventional methods. The model employs a 1D convolutional neural network (CNN) for the NIR spectrum and a lightweight CNN with an attention mechanism for GAF images, ensuring efficient feature representation suitable for mobile applications. The architecture incorporates a novel cross-attention mechanism to enhance feature integration across modalities, leading to improved generalization. Notwithstanding the intricacy of data integration, the model maintains efficiency and attains state-of-the-art classification accuracy on datasets comprising mango cultivars, pine species, and tree organs, thereby providing a pragmatic solution for real-time classification. 

### The Architecture of Proposed Method

![](G:/%E8%AE%BA%E6%96%87-%E9%A1%B9%E7%9B%AE-%E6%96%87%E4%BB%B6/%E7%83%9F%E8%8D%89%E7%A0%94%E7%A9%B6/%E5%B0%8F%E8%AE%BA%E6%96%87/multimodal%20classification/code/fig/fig1.png)

#### NIR stream Network

<img src="G:\论文-项目-文件\烟草研究\小论文\multimodal classification\code\fig\image-20250618230546882.png" alt="image-20250618230546882" style="zoom:33%;" />

#### Designed Attention module

<img src="G:\论文-项目-文件\烟草研究\小论文\multimodal classification\code\fig\image-20250618230622911.png" alt="image-20250618230622911" style="zoom:33%;" />

#### The GAF stream Network

<img src="G:\论文-项目-文件\烟草研究\小论文\multimodal classification\code\fig\image-20250618230703330.png" alt="image-20250618230703330" style="zoom:33%;" />

#### The Dynamic-Sparse Cross Attention

<img src="G:\论文-项目-文件\烟草研究\小论文\multimodal classification\code\fig\image-20250618230747959.png" alt="image-20250618230747959" style="zoom:33%;" />

## How to use these codes:

multimodal.py : Mainly the definition of the model and the attention module, as well as extracting data and defining the format of data

utils.py: Mainly the operation process code of the entire network

options.py: Parameter definition of model network and hyperparameters of data

```python
run multimodal.py
```


# 深度生成模型课程大作业：PointAugment复现与改进

本仓库包含了对CVPR 2020的paper [PointAugment: an Auto-Augmentation Framework for Point Cloud Classification](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_PointAugment_An_Auto-Augmentation_Framework_for_Point_Cloud_Classification_CVPR_2020_paper.pdf)进行复现的一些实验与结论。代码在原作者的[开源代码](https://github.com/liruihui/PointAugment)基础上修改得到。

## 思路与实验流程
我们最开始的思路是跑通原作者的代码，然后在此基础上进行改进。第一项改进是对Augmentor结构的改进，尝试将Augmentor的backbone替换为基于Attention机制的PCT网络，发现效果没有明显提升。第二项改进试图简化Augmentor的loss，我们打算先尝试将Augmentor的loss直接设置为classifier的loss的相反数（即两个模块直接对抗），直觉上这个训练很可能是不稳定的，但我们写代码时误将loss写成了与classifier相同而不是相反数（即两个模块同向训练），却惊讶地发现训练结果没有怎么掉点，这让我们怀疑Augmentor是否真的有用。
于是我们改变了实验思路，做了以下一系列实验：
* 将Augmentor的loss只保留第一项，即与classifier同向训练
* 将Augmentor去掉，只训练classifier（使用传统数据增强）
* 将Augmentor去掉，只训练classifier（去掉传统数据增强）
* 将Augmentor中的M矩阵和D矩阵全部用噪音生成，而不是用原始点云的特征

## 实验结果与分析

### Baseline与本文方法的复现结果
实验：MN40数据集，点云分类任务

| 方法 | 论文结果 | 复现结果 |
| :-: | :-: | :-: |
| PointNet(+DA) | 89.2 | 90.56 |
| PointNet++(+DA) | 90.7 | 93.03 |
| PointNet(+PA) | 90.9 (1.7↑) | 90.92 (0.36↑) |
| PointNet++(+PA) | 92.9 (2.2↑) | 93.11 (0.08↑) |
| PointNet(w/o DA) |  | 90.07 |

（由于PointNet++的运行速度很慢，我们大部分实验都是在PointNet上做的）

分析：
在仅使用传统数据增强的条件下，PointNet和PointNet++的复现结果均高于原始论文，使用PointAugment做数据增强的结果与本文基本一致。复现结果中PointAugment在PointNet和PointNet++上取得的提升分别为0.36和0.08个点，远低于本文中报告的数值。即使去掉传统数据增强，PointNet的复现结果（90.07）也比文中报告的数据（89.2）要高不少。

### 对Augmentor进行的消融实验
实验：MN40数据集，点云分类任务，Classifier使用PointNet

- 使用Classifier loss更新Augmentor：90.72 (0.20↓)
- 只使用文中Augmentor loss的第一项更新Augmentor：90.62 (0.30↓)
-- 分析：这两种训练方式本质上都是让Augmentor与Classifier同向训练，按理说Augmentor会试图学出一个恒等映射，这样对classifier分类难度是最低的，这样Augmentor就没有起到任何作用，但实验表明准确率只下降了一点。
- 在Augmentor网络中，不使用点云特征，而是只用噪声作为M矩阵和D矩阵的输入：90.64 (0.28↓)
-- 分析：这更加证明了Augmentor网络没有起到原文所说的作用，即能够对每个点云生成特异性的增强样本，以解决传统增强方法不灵活的问题。

### 对Augmentor网络结构进行的改进
实验：MN40数据集，点云分类任务，Classifier使用PointNet

- Augmentor改用PCT backbone：91.00 (0.08↑)
-- 分析：没有明显改进。这个实验是我们最开始做的，既然后续实验已经证明Augmentor没有太大作用，那这个实验也就没有意义了，这里仅仅贴一下结果。

## 结论
我们通过一些实验证明了原文提出的方法（使用Augmentor进行对抗训练做为数据增强）没有明显效果。原文列举的实验结果通过将他的方法和baseline原文报告的结果比较以证明准确率的提升，但这个提升主要源于实现上的改进（以及一些不明的trick）提升了baseline的表现。

## 代码结构
```
train_PA.py     训练入口代码
Augment
- |- augmentor.py  Augmentor网络
- |- augmentor2.py 我们改进的两个Augmentor网络，采用PCT做backbone，以及不使用点云特征，随机生成M和D
- |- config.py     配置参数
- |- model.py      完整模型和训练流程
- |- pointnet.py   PointNet网络以及我们加入的PointNet++网络实现
- |- pointnet2.py  和pointnet.py内容相同，在训pointnet++的时候方便备份
Common
- |- data_utils.py 传统数据增强
- |- loss_utils.py 各个模块的Loss函数
- |- ModelNetDataLoader.py MN40数据集的DataLoader
- |- point_operation.py    传统数据增强的点变换
```

## 代码运行方式

### 运行环境
Pytorch 1.7。

运行时如果报错import error，需要安装所缺少的库（如TensorBoardX等）。

训练前需要先下载[MN40数据集](https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip)到本地文件夹中。

### 各项实验
验证原文的PointAugment算法（MN40, 点云分类任务）

```python3 train_PA.py --data_dir ModelNet40_Folder```

如果运行时报错，建议使用单卡训练。

每次运行会将所有记录存入到log下的文件夹中，包括：
- log_train.txt  训练时的完整输出，包括最终的结果
- args.txt   命令行参数
- events.xxx 训练过程中tensorboard记录的summary，包含val_acc等
- xxx.py.backup  是本次运行时各代码文件的备份

由于我们实验过程中频繁修改代码，因此要运行以下实验的话，需要将对应log目录中的代码备份文件（xxx.py.backup）覆盖到Augment和Common文件夹中。

具体实验包括：
- 原文的PointAugment算法
-- log目录20210611-2107
- 用Classifier Loss更新Augmentor
-- log目录20210612-1525
- Augmentor Loss只保留第一项
-- log目录20210612-1637
- 只训练Classifier
-- log目录20210612-2152
- 只训练Classifier，并且去掉传统的数据增强
-- log目录20210615-1024
- 用PointNet++做Classifier，训练原文的PointAugment算法
-- log目录20210613-1820
-- 改用指令```python3 train_PA.py --data_dir ModelNet40_Folder --batch_size 12 --model_name pointnet2```
- 用PointNet++做Classifier，只训练Classifier
-- log目录20210614-1015
-- 改用指令```python3 train_PA.py --data_dir ModelNet40_Folder --batch_size 12 --model_name pointnet2```
- Augmentor改用PCT做backbone
-- log目录20210513-1012
- Augmentor改成不使用点云特征，随机生成M和D矩阵
-- log目录20210615-1551

列表中最后两项实验需要在pytorch 1.1的库版本下运行。这是因为我们两个组员分别尝试了各种实验，但彼此的pytorch版本不同。pytorch新旧版本的代码互相不兼容，本repo中放的代码大部分都是pytorch 1.7的，但最后两项实验中log目录下的代码是1.1版本的，运行时需要切换pytorch版本。
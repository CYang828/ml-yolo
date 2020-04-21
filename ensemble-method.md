# 集成模型(Ensemble Model)

降低不确定性，使模型更稳定。

集成模型的稳定性高的说明。

稳定性的基础是多样性(Diversity)。
- 训练样本的随机化
- 特征选择的随机化

比如说推荐系统中推荐的多样性。
<div align="center"> <img src="assets/image.png" rel="image.png" /></div>

## Bagging(Bootstrap Aggregating)

<div align="center"> <img src="assets/image.png" rel="image.png" /></div>

<div align="center"> <img src="assets/image.png" rel="image.png" /></div>

<div align="center"> <img src="assets/image.png" rel="image.png" /></div>

## Random Forests

## Boosting方法
Bagging方法和Boosting方法的异同：
- Bagging是使用weak learners，由于每个模型都是过拟合的
- Boosting也是使用weak learners，由于每个模型都是欠拟合的

###  AdaBoosting(Adaptive Boosting)
根据数据的正确和错误对数据进行加权。

### GBDTBoosting提升树
使用残差迭代训练。最终的预测值为所有模型预测之和。

### XGBoost
在GBDT的基础上，提升在大数据上的并行能力。

整个推导的过程。

## Majority Voting
<div align="center"> <img src="assets/image.png" rel="image.png" /></div>

## Soft Voting

这里不仅考虑了每个模型的分类结果，同时也考虑分类的强度，并且给每个模型赋予一定的权重。

使用不同模型求众数。

有放回的随机采样。

假设数据是均匀分布的，一条数据不被采样的概率是0.368.

**Bias-Variance分解**

Loss = Bias + Variance + Noise

调整训练数据中的权重。

把很多不同的树结合起来，在训练这些树的时候，生成节点使用所有特征的子集。

## Stacking

把第一层预测的输出当作是第二层的输入来用。

## 加入噪音

Dropout/Dropconnect

DeNoising Autoencoder

## Bayesian Model

### 随机森林
对于每一棵树能看到的样本维度进行限制

对于样本进行限制

每棵树最大限度的生长，不做任何修剪

优点：
- 适用数据集广
- 高维数据(可并行计算)
- Feature重要性排序
- 训练速度快，可并行

缺点
- 噪音较大时容易过拟合
- 级别划分较多的属性影响大

### Boost算法
#### Adaboost
主要用来解决分类问题。

训练多弱分类器，可以使每棵树的深度都小一些，这样可以使得训练的更快速，效果也很好。

关注被错分的样本，器重性能好的分类器。

关注->增加样本分错样本权重
器重->好的分类器全重大

#### GBDT(Gradient Bossting Decision Tree)

GBDT几乎可以用于所有的回归问题(线性和非线性)。

也可以用于解决二分类问题(设定阈值)，不太适合做多分类问题

广告推荐和各种竞赛

使用残差迭代训练。最终的预测值为所有模型预测之和。

与随机森林的比较：
- 精度上GBDT > RF 
- 过拟合RF harder > GBDT
- 建模能力 GBDT > RF
- 并行化 RF > GBDT

Facebook 使用Boosting的方法进行特征组合，进行CRT预估

需要学习的三样东西：
- 树的形状(hard work)
- 每一个决策的阈值
- 页节点的值

学习一个模型的形状是相对难的一个问题。

这个问题就可以被定义为从树的顶点到各层不确定性越来越小的决策，或者说特征的好坏程度递减的。

评判不确定性：信息熵(entropy)

entropy 计算

连续特征的转换

回归问题怎么做：标准差

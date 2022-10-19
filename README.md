# gPINNs pytorch 复现

- ## 论文：

  ### 原文：[gPINNs](https://www.sciencedirect.com/science/article/pii/S0045782522001438?via%3Dihub)

###   源码：[Deepxde](https://github.com/lu-group/gpinn)



- ## 复现情况：



#### 1. 正向问题(Forward question)

- **Function approximation via a gradient-enhanced neural network** 

  $$
  u(x) = −(1.4 − 3x) sin(18x), x \in [0, 1]
  $$


  - **Loss Function:** 
  $$
    \mathcal{L}  = \frac{1}{n}\sum_{i = 1}^{n}\left|u(x_{i})-\hat{u}(x_{i})\right |^{2} + w_{g}\frac{1}{n}\sum_{i = 1}^{n}\left|\bigtriangledown u(x_{i})-\bigtriangledown \hat{u}(x_{i})\right |^{2}
    $$


### 预测对比：

| Figure.1 C & D | ![Figure.1 C](https://github.com/konanl/gPINNs_pytorch/blob/main/paper%20figure/figure1%20C.png) | ![Figure.1 D](https://github.com/konanl/gPINNs_pytorch/blob/main/paper%20figure/figure1%20D.png) |
| :------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|    15-u/u`     | ![PINNs 15 u](./result/figure/function/u-pinn-15.png)![gPINNs 15 u](./result/figure/function/u-gpinn-15.png) | ![gPINNs 15 u_g](./result/figure/function/u_g-pinn-15.png)![gPINNs 15 u_g](./result/figure/function/u_g-gpinn-15.png) |
|    20-u/u`     | ![PINNs 15 u](./result/figure/function/u-pinn-20.png)![gPINNs 15 u](./result/figure/function/u-gpinn-20.png) | ![PINNs 15 u](./result/figure/function/u_g-pinn-20.png)![gPINNs 15 u](./result/figure/function/u_g-gpinn-20.png) |
|                |                                                              |                                                              |



### Figure.1 A & B 对比：



| Figure.1 A & B   | ![figure1 A](./paper%20figure/figure1%20A.png)     | ![figure1 B](./paper%20figure/figure1%20B.png)      |
| ---------------- | ---------------------------------------------- | ----------------------------------------------- |
| Figure.1  NN/gNN | ![f1 NN](./result/figure/function/L2%200f%20u.png) | ![f1 NN](./result/figure/function/L2%200f%20u`.png) |
|                  |                                                |                                                 |



- **3.2.1 Poisson equation**

  $$
  \Delta u = \sum_{i=1}^{4} isin(ix) + 8sin(8x),  x\in [0, \pi]
  $$
  
  
  - **Loss Function:**
  
    $$
    \mathcal{ L = L_{f} + wL_{g} }
    $$



### **预测对比：**

| Figure.2 D & E | ![figure 2 D](./paper%20figure/figure2%20D.png)                  | ![figure 2 D](./paper%20figure/figure2%20E.png)                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 15-u/u'        | ![figure2 15 D](./result/figure/poisson-1D/u-pinn-15.png)![figure2 15 D](./result/figure/poisson-1D/u-gpinn-15.png) | ![figure2 15 E](./result/figure/poisson-1D/u_g-pinn-15.png)![figure2 15 E](./result/figure/poisson-1D/u_g-gpinn-15.png) |
| 20-u/u`        | ![figure2 20 D](./result/figure/poisson-1D/u-pinn-20.png)![figure2 20 D](./result/figure/poisson-1D/u-gpinn-20.png) | ![figure2 20 E](./result/figure/poisson-1D/u_g-pinn-20.png)![figure2 20 E](./result/figure/poisson-1D/u_g-gpinn-20.png) |



### Figure.2 A,B & C 对比：

| Figure.2 A,B & C | ![A](./paper%20figure/figure2%20A.png)           | ![A](./paper%20figure/figure2%20B.png)            | ![A](./paper%20figure/figure2%20C.png)             |
| ---------------- | -------------------------------------------- | --------------------------------------------- | ---------------------------------------------- |
| 复现             | ![A](./result/figure/poisson-1D/L2%20of%20u.png) | ![A](./result/figure/poisson-1D/L2%20of%20u`.png) | ![A](./result/figure/poisson-1D/pde%20error.png) |                                 



### Figure.2 F & G:



| Figure.2 F & G | ![F](./paper%20figure/figure2%20F.png)          | ![F](./paper%20figure/figure2%20G.png)           |
| -------------- | ------------------------------------------- | -------------------------------------------- |
| 复现           | ![F](./result/figure/poisson-1D/u_of_w.png) | ![G](./result/figure/poisson-1D/u`_of_w.png) |

- 注：复现的图与论文原图情况不一致的原因是，作者是训练10次取平均的结果，而由于设备的原因，我只run了3次取平均的结果，所以可能是计算平台的差异导致的不一样的结果；但是，总体来说，在合适的权重情况下，还是gPINN的效果更好一点。

- **3.2.2 Diffusion-reaction equation**
  
  
  $$
  \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} + R(x, t), \qquad x \in [-\pi, \pi], t \in [0, 1]
  $$
  
  
  $$
  R(x, t) = e^{-t}[\frac{3}{2}sin(2x)+\frac{8}{3}sin(3x)+\frac{15}{4}sin(4x)+\frac{63}{8}sin(8x)]
  $$
  
  
  
  - **Loss Function:**
    
    
    $$
    \mathcal{L = L_{f}+wL_{gx}+wL_{gt}}
    $$



### 预测对比：

| ![figure bc](./paper%20figure/figure4%20BC.png)![figure bc](./paper%20figure/figure4%20DE.png) |
| ------------------------------------------------------------ |



| ![figure.4](./result/figure/diffusion-reaction/figure4.png) |
| ----------------------------------------------------------- |

- 注：这里的gPINNs第一幅图g_wight=0.1， 第二幅图g_weight=0.01
----



### Figure.3 A,B & C,D 对比：

| Figure.3 | ![figure bc](./paper%20figure/figure3.png) |
| :------- | ------------------------------------------ |



| <img src="./result/figure/diffusion-reaction/figure3-A.png" alt="figure.3" style="zoom:10%;" /> | <img src="./result/figure/diffusion-reaction/figure3-B.png" alt="figure.3" style="zoom:10%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./result/figure/diffusion-reaction/figure3-C.png" alt="figure.3" style="zoom:10%;" /> | <img src="./result/figure/diffusion-reaction/figure3-D.png" alt="figure.3" style="zoom:10%;" /> |



- 对于这里当gPINN的weight分别等于1，0.1，0.01时，出现的不稳定情况，原因可能如下：
  1. 作者这里是，对于每一种模型分别训练了10次，然后取平均情况，所以最后的结果相对来说也叫稳定；而由于没有足够的计算资源，我只训练了2次取平均，所以结果相对来说也比较不稳定
  2. 梯度的计算没有优化，模型在计算梯度的过程中，存在一些累计的误差导致gPINN的权重较大时，效果不如原始的PINN







#### 2. 反向问题(Inverse problem)

- **3.3.1 Brinkman-Forchheimer model**

### case 1


$$
 -\frac{\nu_{e} }{\epsilon } \nabla^{2}u + \frac{\nu u}{K} = g, \qquad x \in [0, H],
$$



##### Loss Function:


$$
\mathcal{L = L_{f}+wL_{g}+L_{data}}
$$



### 预测对比：

| Figure 6. D                        | ![figure.6 D](./paper%20figure/figure6%20D.png)                  |
| ---------------------------------- | ------------------------------------------------------------ |
| 10 train points - 5 observe points | ![nn](./result/figure/BF/case%201/u-pinn-10.png) ![nn](./result/figure/BF/case%201/u-gpinn-10.png) |
| 20 train points - 5 observe points | ![nn](./result/figure/BF/case%201/u-pinn-20.png) ![nn](./result/figure/BF/case%201/u-gpinn-20.png) |
| 30 train points - 5 observe points | ![nn](./result/figure/BF/case%201/u-pinn-30.png) ![nn](./result/figure/BF/case%201/u-gpinn-30.png) |





### Figure.6 A,B,C



| ![figure.6 ABC](./paper%20figure/figure6%20ABC.png) |
| --------------------------------------------------- |





| <img src="./result/figure/BF/case%201/figure6_A.png" alt="figure.6 A" style="zoom:25%;" /> | <img src="./result/figure/BF/case%201/figure6_B.png" alt="figure.6 B" style="zoom:25%;" /> | <img src="./result/figure/BF/case%201/figure6_C.png" alt="figure.6 C" style="zoom:25%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |



### Figure.6 E

| ![Figure.6 E](./paper%20figure/figure6%20E.png) | <img src="./result/figure/BF/case%201/figure6_E.png" alt="re figure.6 E" style="zoom: 10%;" /> |
| ----------------------------------------------- | ------------------------------------------------------------ |





### case 2



### 预测对比：

| Figure.7 A                                          | ![figure.7 A](./paper%20figure/figure7%20A.png)              |
| --------------------------------------------------- | ------------------------------------------------------------ |
| 10 train points - 5 observe points pinn             | ![figure.7 A NN](./result/figure/BF/case%202/u-NN-10.png)    |
| 10 train points - 5 observe points gpinn w=0.1/0.01 | ![figure.7 A NN](./result/figure/BF/case%202/u-gNN,%20w=0.1-10.png)![figure.7 A NN](./result/figure/BF/case%202/u-gNN,%20w=0.01-10.png) |



### Figure.7 B

| ![figure.7 B](./paper%20figure/figure7%20B.png) | <img src="./result/figure/BF/case%202/figure7_B.png" alt="figure7 B" style="zoom:10%;" /> |
| ----------------------------------------------- | ------------------------------------------------------------ |
|                                                 |                                                              |

 



### Figure.7 C

| ![figure.7 B](./paper%20figure/figure7%20C.png) | <img src="./result/figure/BF/case%202/figure7_C.png" alt="figure7 C" style="zoom:10%;" /> |
| ----------------------------------------------- | ------------------------------------------------------------ |





### When we add Gaussian noise

### 预测对比：

| Figure.8 A | ![figure.8 A](./paper%20figure/figure8%20A.png)              |
| ---------- | ------------------------------------------------------------ |
| 复现       | <img src="./result/figure/BF/case%202/u-NN-12_add_GS_noise.png" alt="figure8 A" style="zoom:30%;" /><img src="./result/figure/BF/case%202/u-gNN, w=0.1-12_add_GS_noise.png" alt="figure8 A" style="zoom:30%;" /> |





### Figure.8 B

| ![figure.8 B](./paper%20figure/figure8%20B.png) | <img src="./result/figure/BF/case%202/figure8_B.png" alt="figure8 A" style="zoom:10%;" /> |
| ----------------------------------------------- | ------------------------------------------------------------ |





### Figure.8 C

| ![figure.8 C](./paper%20figure/figure8%20C.png) | <img src="./result/figure/BF/case%202/figure8_C.png" alt="figure8 C" style="zoom:10%;" /> |
| ----------------------------------------------- | ------------------------------------------------------------ |







- 3.3.2 diffusion-reaction system

### Figure 9:



| Figure A | ![figure9 A](./paper%20figure/figure9%20A.png)               |
| -------- | ------------------------------------------------------------ |
| 复现     | PINN-10:![pinn10](./result/figure/diffusion-reaction-inverse/u-pinn-10.png)gPINN-10:![gpinn10](./result/figure/diffusion-reaction-inverse/u-gpinn-10.png)PINN-100:![pinn100](./result/figure/diffusion-reaction-inverse/u-pinn-100.png) |





| Figure B | ![figure9 B](./paper%20figure/figure9%20B.png)               |
| -------- | ------------------------------------------------------------ |
| 复现     | PINN-10:![pinn10](./result/figure/diffusion-reaction-inverse/k-pinn-10.png)gPINN-10:![gpinn10](./result/figure/diffusion-reaction-inverse/k-gpinn-10.png)PINN-100:![pinn100](./result/figure/diffusion-reaction-inverse/k-pinn-100.png)gPINN-100:![pinn100](./result/figure/diffusion-reaction-inverse/k-gpinn-100.png) |





| Figure C | ![figure9 C](./paper%20figure/figure9%20C.png)               |
| -------- | ------------------------------------------------------------ |
| 复现     | PINN-10:![pinn10](./result/figure/diffusion-reaction-inverse/u`-pinn-10.png)gPINN:![gpinn10](./result/figure/diffusion-reaction-inverse/u`-gpinn-10.png) |









- ## 代码说明：

**针对论文的 3.1 3.2 3.3 部分的所以算例均可使用[solver.py](./solver.py)来解决，此部分代码简介如下：**

- 3.1.[Function approximation](./function.py)

- 3.2.1 [1D poisson equation](./poisson%201d.py)
- 3.2.2 [Diffusion-reaction equation](./diffusion%20reaction.py)
- 3.3.1 Brinkman-Forchheimer model
  - [Case 1](./BF1.py)
  - [Case 2](./BF2.py)

- 3.3.2 [Diffusion-reaction system](./diffusion%20reaction%20inverse.py)



**以下模块，由于需要结合RAR，统一使用solver来实现，较复杂，所以，解决方案(train, predict模块)均写在一个python文件里**

- gPINN enhanced by RAR





#### 运行代码：

##### python file

默认：pinn模型来优化

（详细参数见不同问题的源代码）

- 模型中的参数说明：

```bash
usage: PINNs/gPINNs for forward diffusion reaction model [--lr LR] [--net_type NET_TYPE] [--num_epochs NUM_EPOCHS] [--resume_epoch RESUME_EPOCH]
                                                         [--num_epochs_decay NUM_EPOCHS_DECAY] [--num_train_points NUM_TRAIN_POINTS]
                                                         [--g_weight G_WEIGHT] [--output_transform OUTPUT_TRANSFORM]
                                                         [--num_supervised_train_points NUM_SUPERVISED_TRAIN_POINTS] [--test_epochs TEST_EPOCHS]
                                                         [--num_test_points NUM_TEST_POINTS] [--model_save_dir MODEL_SAVE_DIR]
                                                         [--result_dir RESULT_DIR] [--log_dir LOG_DIR] [--log_step LOG_STEP]
                                                         [--model_save_step MODEL_SAVE_STEP] [--lr_update_step LR_UPDATE_STEP]

```









- ## 复现过程中存在的问题：

  

1. 内容
2. 内容

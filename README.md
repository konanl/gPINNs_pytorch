# gPINNs pytorch 复现

- ## 论文：

  ### 原文：[gPINNs](https://www.sciencedirect.com/science/article/pii/S0045782522001438?via%3Dihub)

###   源码：[Deepxde](https://github.com/lu-group/gpinn)



- ## 复现情况：



#### 1. 正向问题(Forward question)

- **Function approximation via a gradient-enhanced neural network** (Figure.1)

  $$ u(x) = −(1.4 − 3x) sin(18x), x \in [0, 1] $$

  - **Loss Function:**

    $$\mathcal{L}  = \frac{1}{n}\sum_{i = 1}^{n}\left|u(x_{i})-\hat{u}(x_{i})\right |^{2} + w_{g}\frac{1}{n}\sum_{i = 1}^{n}\left|\bigtriangledown u(x_{i})-\bigtriangledown \hat{u}(x_{i})\right |^{2}$$

### **预测对比：**

| Figure.1 C & D | ![Figure.1 C](./paper figure/figure1 C.png)                  | ![Figure.1 D](./paper figure/figure1 D.png)                  |
| :------------: | ------------------------------------------------------------ | ------------------------------------------------------------ |
|    15-u/u`     | ![PINNs 15 u](./result/figure/function/u-pinn-15.png)![gPINNs 15 u](./result/figure/function/u-gpinn-15.png) | ![gPINNs 15 u_g](./result/figure/function/u_g-pinn-15.png)![gPINNs 15 u_g](./result/figure/function/u_g-gpinn-15.png) |
|    20-u/u`     | ![PINNs 15 u](./result/figure/function/u-pinn-20.png)![gPINNs 15 u](./result/figure/function/u-gpinn-20.png) | ![PINNs 15 u](./result/figure/function/u_g-pinn-20.png)![gPINNs 15 u](./result/figure/function/u_g-gpinn-20.png) |
|                |                                                              |                                                              |



### Figure.1 A & B 对比：



| Figure.1 A & B   | ![figure1 A](./paper figure/figure1 A.png)     | ![figure1 B](./paper figure/figure1 B.png)      |
| ---------------- | ---------------------------------------------- | ----------------------------------------------- |
| Figure.1  NN/gNN | ![f1 NN](./result/figure/function/L2 0f u.png) | ![f1 NN](./result/figure/function/L2 0f u`.png) |
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

| Figure.2 D & E | ![figure 2 D](./paper figure/figure2 D.png)                  | ![figure 2 D](./paper figure/figure2 E.png)                  |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 15-u/u'        | ![figure2 15 D](./result/figure/poisson-1D/u-pinn-15.png)![figure2 15 D](./result/figure/poisson-1D/u-gpinn-15.png) | ![figure2 15 E](./result/figure/poisson-1D/u_g-pinn-15.png)![figure2 15 E](./result/figure/poisson-1D/u_g-gpinn-15.png) |
| 20-u/u`        | ![figure2 20 D](./result/figure/poisson-1D/u-pinn-20.png)![figure2 20 D](./result/figure/poisson-1D/u-gpinn-20.png) | ![figure2 20 E](./result/figure/poisson-1D/u_g-pinn-20.png)![figure2 20 E](./result/figure/poisson-1D/u_g-gpinn-20.png) |







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







----



#### 2. 反向问题(Inverse problem)

- **3.3.1 Brinkman-Forchheimer model**

$$
-\frac{\nu_{e} }{\epsilon } \nabla^{2}u + \frac{\nu u}{K} = g, \qquad x \in [0, H],
$$





##### Loss Function:


$$
\mathcal{L = L_{f}+wL_{g}+L_{data}}
$$




### 预测对比：

| Figure 6. D                        | ![figure.6 D](./paper figure/figure6 D.png)                  |
| ---------------------------------- | ------------------------------------------------------------ |
| 10 train points - 5 observe points | ![nn](./result/figure/BF/case 1/u-pinn-10.png)![nn](./result/figure/BF/case 1/u-gpinn-10.png) |
| 20 train points - 5 observe points | ![nn](./result/figure/BF/case 1/u-pinn-20.png)![nn](./result/figure/BF/case 1/u-gpinn-20.png) |
| 30 train points - 5 observe points | ![nn](./result/figure/BF/case 1/u-pinn-30.png)![nn](./result/figure/BF/case 1/u-gpinn-30.png) |


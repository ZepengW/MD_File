# Unity Style Transfer for Person Re-Identification

> 摘自CVPR 2020
>
> 解决Style Variation的问题，即受限于环境和不同设备影响，在摄像头网络中，不同监控设备采集的图像存在差异。
>
> 本文提出利用UnityGAN网络来学习不同cameras之间的style change，处理得到shape-stable style-unity images，以此消除图片间非关键性的差异。

## Comparison

对于该问题，有三类方法

* 基于传统方法的KISSME，DNS
* 基于deep representation learning的IDE，PCB
* 利用GANs网络的CamStyle（本文基础）

然而CamStyle具有下面局限性

1. 使用CycleGAN转换会存在image artifacts，如Figure 2中的红色坏点
2. 生成的增强图像带来了噪音，需要用Label Smooth Regularization(LSR) 来调整网络
3. 生成的增强图像只能被用来做扩展训练集，不是很有效
4. 模型被训练数目为$C_{C}^2$，随着摄像头数量增多，训练模型数目大大增强，就像图2中的表述：Cam 6要对每一个cam生成一个风格转移模型

<img src=".\img\image-20200818214109570.png" alt="image-20200818214109570" style="zoom: 67%;" />

本文提出的UnityGAN网络，可以克服上述easy deformation，以及依赖每个camera的style data来学习生成适合所有camera style 的图像，提高了效率又减少了模型计算量。

本文提出UnityStyle方法的**优点**：

1. 作为数据增强方法，生成的增强样本可以被当作原始样本处理，无需LSR处理
2. 还可以适应同一个摄像头的style change
3. 无需额外的信息
4. 只需要训练C个UnityGAN模型

## Related Work

IBN-NET

​	Batch Normalization (BN) 改善图片内容特征的敏感性sensitivity

​	**Instance Normalization (IN)** 改善风格变化的鲁棒性

​	故IN achieves better results than BN in the field of style migration. **Shallow features are related to style, and deep features are related to high-level features (content features such as face and gesture)**

故本文提出两条准则：

* UN只能被加在shallow network，不能放在deep network，因为IN消除图片间差异，在shallow network消除风格差异，要是放在deep network会消除关键特征差异
* 为了保证内容相关的信息平滑地传递到deep layer，要限制在shallow BN layer

## Method 

### Unity-GAN

​	吸收了DiscoGAN和CycleGAN的优点并进行改进

​	DiscoGAN：narrow bottleneck layer可以防止输出的图像retaining visual details in the input image

​	CycleGAN：residual block增加了DiscoGAN的capacity，但是single-scale layer上residual block的引入不能保留multiple scale的信息。

​	Unity-GAN融合两种网络

![image-20200818222517661](.\img\image-20200818222517661.png)

如上图，Unity-GAN在引入residual blocks的同时并skip connections in multi-scale layer，可以保留multi-scale information，使得transformation更加精细。通过multi-scale information的风格转移，UnityGAN可以生成结构稳定的图像，如图4。同时UnityGAN引入IBN-ResBlock（如上图），来保证模型生成统一style的图片。

![image-20200818222957914](.\img\image-20200818222957914.png)

UnityGAN的损失函数引入identity mapping loss：
$$
L_{ID}=E_{x\sim l_{x}}(\|F(x)-x\|_{1})+E_{y\sim l_{y}(\|G(y)-y\|_{1})}
$$
​	其中$G:X\rightarrow Y$ ，$F:Y\rightarrow X$    $D_{X}$，$D_{Y}$代表discriminators for G and F

故UnityGAN 的loss 有4种loss function构成：标准GAN loss，feature matching loss，identiy mapping loss和cyclic reconstruction loss
$$
L_{UnityGAN}(x,y)=\lambda_{GAN}SLN(L_{GAN})\\
+\lambda_{FM}SLN(L_{FM})\\
+\lambda_{ID}SLN(L_{ID})\\
+\lambda_{CYC}SLN(\lambda_{SS}L_{SS}+\lambda_{L1}L_{L1})
$$
SLN is scheduled loss normalization

处理流程如下图，将train set中的所有cam的图像来训练UnityGAN，然后用训练好的GAN网络进行generate stable structure pictures，但是UnityGAN产生的图片在style上是不稳定的，故我们下一节引入UnityStyle loss

![image-20200819082818544](.\img\image-20200819082818544.png)

### UnityStyle

如上面Figture 3，为了保证UnityGAN生成UnityStyle的图片，加入Style Attention Module。低维图片特征通过该模块得到style-related attention features。

**注：Figure 3的蓝色线指向不是表示虚线框内是IBN-Res，而实表示经过IBN-RES的输出经过该Style Attention Module直接输出到对应Scale**
$$
A(x)=Sigmoid(A_{style}(G_{1}(x)))\\
其中，A_{style}是Style~Attention~Module,G_{1}是第一个IBN-Res Block的输出
$$
UnityStyle loss为
$$
L_{UnityStyle}=\sum_{c=1}^L(A(y_{i}^{(c)})L_{UnityGAN}(x_{i},y_{i}^{(c)}))\\
c~is~camera~number,C~is~the~number~of~cameras\\
A(y_{i}):the~ith~camera's~style~attention
$$
从而使UntityGAN改进如下

![image-20200819084557075](.\img\image-20200819084557075.png)

### Pipeline

![image-20200819091927065](.\img\image-20200819091927065.png)

训练好UnityGAN后，我们对ReID任务的神经网络部署

在Training阶段，将原始训练数据以及生成的UnityStyle训练数据一起作为Enhanced Data输入训练神经网络

Loss函数为
$$
L_{REID}=\frac{1}{N}\sum_{i=1}^N(L_{R}^i+L_{U}^i)\\
L_{R}^i是ith原始训练图像样本的Cross-entropy loss\\
L_{U}^i是ith的UnityStyle图像样本的Cross-entropy loss\\
L_{Cross}(x)=-\sum_{l=1}^L log(p(l))q(l)
$$
化简
$$
L_{REID}=-\frac{1}{N}\sum_{i=1}^Nlog(p_{R}^ip_{U}^i)
$$
​	第一节提到的方法，real image 和 fake image分别用不同的loss function训练，本文克服了这个问题，无需使用LSR

测试阶段，对测试数据集的query和gallery数据进行UnityStyle Transfer，利用生成的new query 和gallery进行测试

## Experiment

![image-20200819093409994](.\img\image-20200819093409994.png)
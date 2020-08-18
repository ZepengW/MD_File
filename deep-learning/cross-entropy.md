# Cross-Entropy
  交叉熵主要刻画的是实际输出（概率）与期望输出（概率）的距离，即概率分布的接近程度。比如在分类问题中，期望输出的向量是[0,0,1,0,0]，表明有100%的概率属于第三个分类，而实际输出的可能是[0.1, 0.1, 0.7, 0.01, 0.09] 这样的概率分布。这时候计算交叉熵可以通过数值的形式表征两者分布之间的差异，从而进行训练拟合。
## 计算公式

$$
H(p,q)=-\sum_{x}( p(x)logq(x)+(1-p(x))log(1-q(x)) )
$$

  这里x是样本值，p和q分别是两个概率分布的样本概率，给出下面的例子：
  期望概率p的分布是[1,0]，实际概率q的分布是[0.8, 0.2]，则H(p,q)的计算结果如下：

$$
H(p,q)=(1*log0.8+0*log0.2)+(0*log0.2+1*log0.8)
$$
## Pytorch中的CrossEntropyLoss函数
pytorch中采用的不是上述公式，而是用另一个种计算形式得到的
$$
H(p,q)=-\sum_{x}(p(x)logq(x))
$$
再引入 log_softmax 和 nll_loss
其中：

### log_softmax
log_softmax函数是log和softmax函数的组合
softmax函数公式如下：

$$
\frac{exp(x_{i})}{\sum_{j}exp(x_{j})}
$$
softmax对于输入的实数向量x，输出一个概率分布，每个元素都是非负且综合为1。实际意义是：根据向量中每个元素的大小，来输出一个概率分布。
例如：对于输入的向量[1,5,3]，输出为[0.015, 0.866, 0.117]，即样本属于第二个分类的概率最大，这样就可以把输出的特征距离归一化表示为概率（个人理解）
log_softmax 函数则是：log(softmax(x))

### nll_loss: negative log likelihood loss
函数表达式为：
$$
f(x, class)=-x[class]
$$
故**pytorch**中的**CrossEntropyLoss**公式如下
$$
loss(x, class)=-log(\frac{exp(x[class]}{\sum_{j}exp(x[j])})
$$
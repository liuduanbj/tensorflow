# TensorFlow.js 使用

## 安装

```bash
## 浏览器安装
pnpm add @tensorflow/tfjs
pnpm add @tensorflow/tfjs-vis

## node.js 安装
pnpm add @tensorflow/tfjs-node
```

## 神经网络工作机制

<https://www.youtube.com/watch?v=aircAruvnKk>

> 神经网络处理信息的核心机制：一层的激活值是通过怎么的运算，算出下一层的激活值的

```txt
输入数据 → 神经网络层 → 激活函数 → 输出预测
                                ↓
                               损失函数 计算误差
                                ↓
                               评估指标 记录性能
                                ↓
                               优化器 调整参数
                                ↓****
                               更新参数
                                ↓
                    重复这个过程直到模型性能达到要求
```

28*28= 784 像素(784个神经元)
的图片识别出 0-9 的数字

### 神经元

装着数字的容器，
神经元中装着的数字代表对应像素的灰度值，0 代表纯黑像素，1 代表纯白像素

### 激活值（Activation）

神经元中装着的数字叫做激活值

* 输入层：有 784 个神经元 (第一层)
* 隐藏层：有 16*2 个神经元 (中间层)
* 输出层：有 10 个神经元（最后一层 0-9）

#### 隐藏层

> 转化为抽象元素，一层层的抽丝剥茧

#### 权重 weight

神经元和上一层神经元之间连线的权限 w，煮饭-各种调料的比例

#### 偏置 bias

神经元和上一层神经元之间连线的偏值 b，煮饭-基础的火候大小

线性代数-矩阵


### 多层感知器 MLP

> 多层感知器（Multilayer Perceptron，MLP）是一种前馈神经网络
> "前馈"的意思是数据只向前流动，像流水一样：输入层 → 隐藏层 → 输出层

基本结构：

1. 输入层：接收原始数据
2. 隐藏层：可以有一层或多层
3. 输出层：产生最终结果

主要特点：

1. 全连接：每层的每个节点都与下一层的所有节点相连
2. 非线性：使用非线性（能画曲线，能拟合更复杂的形状）激活函数（如 ReLU, Sigmoid）
3. 可训练：通过反向传播算法学习权重
   1. 先预测一个结果（投篮（前馈））
   2. 计算预测结果与实际结果的误差（看球是偏左还是偏右了（计算误差））
   3. 根据误差调整权重（根据偏差调整姿势（反向传播调整权重））
   4. 重复以上步骤直到误差最小（不断重复直到投准）

### 激活函数 (Activation Function)

> 作用：激活函数让神经网络能够学习复杂的模式，就像人类大脑能够处理复杂的信息一样。没有激活函数，神经网络就只能学习简单的线性关系。

为什么需要非线性？

* 如果神经网络中所有层都是线性的，那么整个网络仍然是一个线性模型。线性模型无法处理复杂的非线性问题，因为它只能进行线性组合。
* 非线性激活函数允许神经网络学习更复杂的模式和关系，从而提高模型的表达能力和泛化能力。

举个例子：

```txt
温度和人的感受：
10度 → 冷
20度 → 舒适
30度 → 热
40度 → 非常热

这不是线性关系！
温度每升高10度，感受的变化并不是均匀的


识别数字"3"：
- 不同人写的"3"形状各异
- 可能有倾斜、粗细不同
- 需要非线性特性来适应这些变化
```

#### 常见 激活函数

ReLU：常用于中间层

* 输入小于0时输出0
* 输入大于0时保持不变
  
Softmax：常用于分类问题的输出层

* 将输出转换为概率分布
* 所有输出值的和为1

```js
// ReLU 的实现
function relu(x: number): number {
    return Math.max(0, x);  // 如果 x > 0，返回 x；否则返回 0
}
```

```txt
     ↑
  y  |     /
     |    /
     |   /
     |  /
     | /
-----+------ →
     |     x
     |
```

```txt
Sigmoid(过时了)

     ↑
  1  |    ----
     |   /
     |  /
     | /
     |/
-----+------ →
     |     x
```

```js
// Sigmoid 的实现
function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));  // 将任何数映射到 0~1 之间
}

// 例子
console.log(Math.exp(1));   // e^1  ≈ 2.718
console.log(Math.exp(0));   // e^0  = 1
console.log(Math.exp(-1));  // e^-1 ≈ 0.367

// 当 x 很大时，e^(-x) 接近 0，所以结果接近 1
// 当 x 很小时，e^(-x) 很大，所以结果接近 0
// 当 x = 0 时，结果正好是 0.5
```

### 损失函数 (Loss Function)

> 作用：
>
> 1. 衡量模型预测结果与实际结果之间的差异
> 2. 为优化器提供优化方向

举个例子：

```txt
想象你在练习投篮：
- 目标：球要进篮筐
- 损失函数就是衡量"球偏离篮筐有多远"
  - 完全命中 → 损失值 = 0
  - 偏左10厘米 → 损失值 = 10
  - 偏右20厘米 → 损失值 = 20
  - 完全投偏 → 损失值很大

- 分类问题（比如识别数字）：
  预测：这是数字"3"
  实际：这是数字"8"
  损失函数计算预测错误的程度

- 回归问题（比如预测房价）：
  预测：100万
  实际：120万
  损失函数计算预测值与实际值的差距
```

#### 常见损失函数

分类问题：

* categoricalCrossentropy（多分类）
* binaryCrossentropy（二分类）
  
回归问题：

* meanSquaredError（均方误差）

### 优化器 (Optimizer)

> 作用：调整模型参数，以最小化损失函数
> 损失函数和优化器是配合使用的，它们各自有不同的职责：
损失函数：告诉你"差多远"
优化器：告诉你"怎么调整"

举个例子：

```txt
继续用投篮的例子：
- 你发现球总是偏左
- 优化器就像你的"调整策略"：
  1. 先预测一个结果（投篮（前馈））
  2. 计算预测结果与实际结果的误差（看球是偏左还是偏右了（计算误差））
  3. 根据误差调整姿势（根据偏差调整姿势（反向传播调整权重））
  4. 重复以上步骤直到投准（不断重复直到误差最小）

常见优化器：
- Adam（智能教练）：
  - 自动调整改进的幅度
  - 开始时改进幅度大
  - 接近目标时改进幅度小
  
- SGD（基础教练）：
  - 固定的改进幅度
  - 需要手动设置改进的步伐大小
```

1. 模型预测 → 输出结果
2. 损失函数 → 计算误差（多远）
3. 优化器 → 调整参数（怎么改）
4. 重复以上步骤

#### 常见优化器

Adam：自适应学习率优化器

* 常用且效果好
* 自动调整学习速度
  
SGD：随机梯度下降

* 最基础的优化器
* 需要手动设置学习率

### 评估指标 (Metrics)

> 作用：评估指标是训练过程的"记录员"，提供可视化和分析依据
>
> 1. 训练过程中的记录
> 2. 可视化监控
> 3. 多指标监控

```txt
// 场景：手写数字识别
metrics: ['accuracy']

// 例子：
总样本：100张图片
正确识别：90张
错误识别：10张
准确率 = 90/100 = 90%
```

#### 常见评估指标

* accuracy：准确率
  * 分类问题最常用
  * 正确预测数/总预测数
* precision：精确率
* recall：召回率

### 权重参数（weight） & 偏置参数（bias）

* 可训练参数：模型在训练过程中可以调整 权重&偏置
* 不可训练参数：模型在训练过程中不能调整 权重&偏置

`trainable: true`  // 默认值，参数可以更新

```js
// 不可训练层（冻结层）
model.add(tf.layers.conv2d({
  filters: 8,
  kernelSize: 5,
  trainable: false  // 参数不会更新
}));
```

```txt
输入层(256)    输出层(10)
   ○             ○ (0)
   ○             ○ (1)
   ○             ○ (2)
   ○      →      ○ (3)
   ○             ○ (4)
   ○             ○ (5)
   ○             ○ (6)
  ...            ○ (7)
   ○             ○ (8)
                 ○ (9)
```

每条连线代表一个权重参数
每个输出节点有一个偏置参数

对于每个输出节点：

```txt
输出值 = (输入1 * 权重1 + 
         输入2 * 权重2 + 
         ... + 
         输入256 * 权重256) 
         + 偏置值
```

然后通过softmax激活函数得到每个数字的概率

```txt
输出值 = softmax(输出值)
```

这些权重和偏置参数通过训练过程不断调整，最终学会将256维的特征向量映射为10个数字类别的概率分布。

### 神经网络中的层类型

1. 密集层 (Dense Layer)
最基础的神经网络层
生活比喻：像是一个完全连通的办公室
每个员工(神经元)都和下一层所有员工有联系
信息可以充分流通和交换

2. 卷积层 (Conv2D)
主要用于图像处理
生活比喻：像是用放大镜观察图片
不是一次看整张图，而是局部局部地看
就像我们识别猫时，会注意到耳朵、尾巴等特征
能够提取图像的重要特征（边缘、纹理等）

3. 池化层 (MaxPooling2D)
用于压缩数据，保留重要信息
生活比喻：像是看缩略图
把一张4K图片缩小，但仍能看出主要内容
减少数据量，保留关键特征

4. 展平层 (Flatten)
转换数据维度
生活比喻：像是整理衣服
把叠好的衣服（多维）展开成一排（一维）
为后续处理做准备

举例：
图像识别流程：
输入图片 → 卷积层(提取特征) → 池化层(压缩数据) → 展平层(转换维度) → 密集层(分类) → 输出结果

### 数据集：训练集 + 验证集

* 训练集：用于训练模型
* 测试集：用于评估模型

```js
// 完整数据集分为：
// 训练集（Training Set）：用于训练模型
// 验证集（Validation Set）：用于评估模型

// 在代码中的体现：
const TRAIN_DATA_SIZE = 5500;  // 训练集
const TEST_DATA_SIZE = 1000;   // 验证集

// 使用方式：
model.fit(trainXs, trainYs, {
  validationData: [testXs, testYs],  // 验证集
  epochs: 10
});
```

#### 为什么需要验证集？

* 检测过拟合
* 评估泛化能力
* 指导模型调优

理想情况：

* 训练集和验证集性能接近
* loss ≈ val_loss
* acc ≈ val_acc

过拟合警告：
如果出现：

* val_loss >> loss
* val_acc << acc

说明模型过度拟合训练数据

* 死记硬背考试题 → 过度拟合
* 理解知识点 → 良好拟合

### 批次 (Batch)

每个批次 (Batch) 处理一部分数据

为什么要分批？

* 内存限制：无法一次处理所有数据
* 训练效率：小批量训练可以更频繁地更新模型

### 训练周期 (Epoch)

* 每个周期 (Epoch) 包含多个批次 (Batch)
* 每个批次 (Batch) 处理一部分数据
* 训练轮数 (Epochs) 表示训练的总次数

```js
// 在代码中的体现：
model.fit(trainXs, trainYs, {
  epochs: 10
});
```

### 评估模型

* 验证集
  * 监控训练过程中的模型性能
  * 时机：在每个训练周期（epoch）结束时
* 评估模型
  * 全面分析模型的性能表现
  * 时机：在训练结束后

```js
/**
 * 做出预测
 * @param model
 * @param data
 * @param testDataSize
 * @returns
 */
function doPrediction(
  model: Sequential, // 训练好的模型
  data: MnistData, // MNIST数据集
  testDataSize = 500 // 测试数据量
) {
  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;

  const [testxs, labels] = tf.tidy(() => {
    const d = data.nextTestBatch(testDataSize);
    return [
      d.xs.reshape([testDataSize, IMAGE_WIDTH, IMAGE_HEIGHT, 1]),
      d.labels.argMax(-1)
    ];
  });

  // 使用训练好的模型对测试数据进行预测，输出是一个概率分布数组，形状为 [样本数, 10]，每个样本会得到10个数字（0-9）的概率值
  const preds = (model.predict(testxs) as tf.Tensor).argMax(-1); // argMax 返回最大值的索引，比如概率分布 [0.1, 0.05, 0.8, 0.05] 会返回 2（因为0.8最大）
  // 清理内存
  testxs.dispose();
  return [preds, labels];
}
```

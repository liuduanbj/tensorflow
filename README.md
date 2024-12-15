# learn-tensorflow-demo

> <https://tensorflow.google.cn/tutorials?hl=zh-cn>

## TensorFlow.js 简介

* 使用 TensorFlow.js 开发**机器学习模型**
* TensorFlow.js 为 JavaScript 中**神经网络编程**提供了灵活的构建块
* 直接在浏览器或 Node.js 中使用机器学习模型，无需后端服务器支持，保护用户隐私（数据不需要传输到服务器）
* 支持 JavaScript/TypeScript 开发，与现代前端框架（React、Vue等）完美集成

## 实际应用场景

1. 图像识别
   * 人脸检测（ModiFace 的在线试妆功能）
     * 使用 TensorFlow.js 在浏览器中识别用户脸部特征
     * 实时跟踪面部位置
     * 在正确的位置叠加化妆品效果
   * 物体识别（Airbnb 的图像分类应用）
     * 就像我们人类可以看出一张照片是客厅还是厨房一样，Airbnb 使用 TensorFlow 来自动识别房东上传的房间照片
     * 自动识别房间类型（卧室、厨房、浴室等）
     * 检查照片质量（是否清晰、光线是否充足）
     * 识别房间内的设施（是否有空调、电视等）
   * 手势识别
2. 自然语言处理（InSpace 的在线聊天内容过滤）
   * 文本分类
     * 在消息发送前就能识别不当内容，就像有一个助手实时帮你过滤不友好的评论
   * 情感分析
   * 语言翻译
3. 数据预测（中国移动的网络优化）
   * 销售预测
   * 用户行为分析
   * 趋势预测
     * 预测网络故障
     * 自动检测异常
     * 优化网络性能

## 基本概念解释

### 机器学习模型

> 机器学习模型就像是一个能够"学习"的程序，类似于人类通过经验学习一样。

举个生活中的例子：
想象你在教一个小朋友识别水果：

1. 学习阶段：
    * 给小朋友看很多苹果和橘子的图片
    * 告诉他："这是苹果，是红色的、圆形的"
    * 告诉他："这是橘子，是橙色的、圆形的"
2. 使用阶段：
    * 给小朋友看一个新的水果
    * 他能根据学到的特征（颜色、形状）判断是苹果还是橘子

机器学习模型就是这样工作的！

### 机器学习模型的三个阶段

1. 训练阶段
2. 评估阶段
3. 推理阶段

```js
// 创建一个简单的模型
const model = tf.sequential({
  layers: [
    tf.layers.dense({units: 1, inputShape: [1]})
  ]
});

// 1. 训练模型
await model.fit(训练数据, 标准答案, {
  epochs: 100  // 训练100次
});


// 2. 测试模型效果
const result = model.evaluate(测试数据, 测试答案);

// 3. 使用模型预测
const prediction = model.predict(新数据);
```

### 场景模型类型

1. 分类模型
   * 用途：区分不同类别的事物
   * 例子：
      * 垃圾邮件识别
      * 图片中是猫还是狗
      * 用户评论是正面还是负面
2. 回归模型
   * 用途：预测连续值
   * 例子：
      * 预测房价
      * 预测股票价格
3. 聚类模型
   * 用途：将相似的数据点分组
   * 例子：
      * 将用户分组为不同的兴趣群体
      * 将新闻文章分组为不同的主题

### 神经网络

> 机器学习模型 = 大类（父类）
> 神经网络 = 一种特定类型的机器学习模型（子类）
> 其他类型的机器学习模型：决策树、支持向量机、随机森林、逻辑回归
> 交通工具 = 大类
> 汽车 = 一种特定类型的交通工具

餐厅（机器学习模型）
  ├── 快餐店（简单机器学习模型）
  │   └── 适合简单、快速的需求
  └── 高级餐厅（神经网络）
      └── 适合复杂、精细的需求

使用简单机器学习模型的情况：

* 数据量较小
* 问题相对简单
* 需要模型可解释性
* 计算资源有限

使用神经网络的情况：

* 处理复杂问题（如图像识别）
* 有大量训练数据
* 需要高精度
* 有足够的计算资源

举例：

```js
// 简单预测（使用基础机器学习模型）：预测用户是否会点击广告
const clickPredictor = tf.sequential({
  layers: [
    tf.layers.dense({units: 1, activation: 'sigmoid', inputShape: [5]})
  ]
});

// 使用
const userFeatures = tf.tensor2d([[age, income, timeOnSite, pageViews, deviceType]]);
const clickProbability = clickPredictor.predict(userFeatures);
```

```js
// 神经网络适用场景：图像识别
const imageClassifier = tf.sequential({
  layers: [
    // 卷积层
    tf.layers.conv2d({
      inputShape: [28, 28, 1],
      kernelSize: 3,
      filters: 32,
      activation: 'relu'
    }),
    // 池化层
    tf.layers.maxPooling2d({poolSize: [2, 2]}),
    // 展平层
    tf.layers.flatten(),
    // 全连接层
    tf.layers.dense({units: 10, activation: 'softmax'})
  ]
});
```

### Tensors (张量)

> 张量是一个数学概念，可以理解为数据的容器：

```txt
0维张量：标量（单个数字）
    例子：1, 2, 3

1维张量：向量（一维数组）
    例子：[1, 2, 3]

2维张量：矩阵（二维数组）
    例子：[
      [1, 2, 3],
      [4, 5, 6]
    ]

3维及以上张量：
    例子：[
      [[1, 2], [3, 4]],
      [[5, 6], [7, 8]]
    ]
```

```js
// 创建不同维度的张量
const scalar = tf.scalar(3);                   // 0维
const vector = tf.tensor1d([1, 2, 3]);         // 1维
const matrix = tf.tensor2d([[1, 2], [3, 4]]);  // 2维
const tensor3d = tf.tensor3d([[[1], [2]], [[3], [4]]]); // 3维
```

### 向量 (Vector)

> 向量是具有方向和大小的量，可以表示为一组有序的数字。向量是一维张量的特例。

```js
const vector = tf.tensor1d([1, 2, 3]);

// 3维向量示例
const vector3D = [1, 2, 3];  // x=1, y=2, z=3
```

### Layers (层)

> 层是神经网络的基本构建块

层是对输入数据进行特定运算的单元
多个层组合在一起形成神经网络
每一层都有特定的功能（如卷积、池化等）

```js
// 1. 密集层（全连接层）
const denseLayer = tf.layers.dense({
  units: 32,              // 输出维度
  activation: 'relu',     // 激活函数
  inputShape: [64]        // 输入形状
});

// 2. 卷积层（用于图像处理）
const convLayer = tf.layers.conv2d({
  filters: 32,            // 过滤器数量
  kernelSize: 3,         // 卷积核大小
  activation: 'relu'
});

// 3. 池化层
const poolingLayer = tf.layers.maxPooling2d({
  poolSize: [2, 2]       // 池化窗口大小
});
```

输入数据（张量） -> 层的处理 -> 输出数据（张量）

### logits | log-odds（对数几率）

> 想象你是一个老师，在判断一个学生的作业是否及格：

不是简单地说"及格"或"不及格"
而是给出一个分数，比如 85 分
这个分数反映了你的确信程度

> 你闭着眼睛摸到一个水果
模型会给出每种水果的可能性：
是苹果的可能性：80%
是橘子的可能性：15%
是香蕉的可能性：5%

logits 就像这个分数，表示模型对每个可能结果的确信程度。

举例：假设我们在识别手写数字（0-9），模型会这样工作：

```js
// 假设我们输入了一个手写的数字"3"
const predictions = model.predict(input);

// 模型返回的结果可能是这样的：
[
  0.01,  // 是数字"0"的可能性
  0.02,  // 是数字"1"的可能性
  0.05,  // 是数字"2"的可能性
  0.80,  // 是数字"3"的可能性 (最高分！)
  0.03,  // 是数字"4"的可能性
  0.02,  // 是数字"5"的可能性
  0.02,  // 是数字"6"的可能性
  0.02,  // 是数字"7"的可能性
  0.02,  // 是数字"8"的可能性
  0.01   // 是数字"9"的可能性
]
```

### Keras（凯瑞斯 深度学习）

> Keras是基于Theano的一个深度学习框架，它的设计参考了Torch，用Python语言编写，是一个高度模块化的神经网络库，支持GPU和CPU。
> tf.keras 是用于构建和训练深度学习模型的 TensorFlow 高阶 API。利用此 API，可实现快速原型设计、先进的研究和生产

### 构建学习模型

Sequential 对于堆叠层很有用，其中每一层都有一个输入张量和一个输出张量。层是具有已知数学结构的函数，可以重复使用并具有可训练的变量。大多数 TensorFlow 模型都由层组成。

```js
// 创建一个简单的分类模型
const model = tf.sequential({
  layers: [
    tf.layers.dense({
      units: 10,  // 10个可能的类别
      activation: 'softmax',  // 将结果转换为概率
      inputShape: [784]  // 输入特征数量
    })
  ]
});

// 预测一个样本
const prediction = model.predict(input);
prediction.print();

// 输出可能像这样：
// 数组中的每个数字代表该类的可能性
// [0.1, 0.05, 0.02, 0.6, 0.03, 0.05, 0.05, 0.05, 0.03, 0.02]
```

## 代码演示

### 简单的线性回归

## 性能考虑

## 开发注意事项

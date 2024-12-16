/**
 * TensorFlow.js - 使用 CNN 识别手写数字
 * https://codelabs.developers.google.com/codelabs/tfjs-training-classfication/index.html?hl=zh-cn#0
 *
 * 目标：学会构建一个 TensorFlow.js 模型，以使用卷积神经网络识别手写数字。
 *
 * 通过让分类器“观察”数千个手写数字图片及其标签来训练分类器。然后，我们会使用模型从未见过的测试数据来评估该分类器的准确性。
 * 我们将通过显示输入的多个示例和正确的输出来训练模型。这称为**监督式学习**。
 *
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { MnistData } from '../data/data';
import type { Sequential } from '@tensorflow/tfjs';

async function showExamples(data: MnistData) {
  // Create a container in the visor
  const surface = tfvis
    .visor()
    .surface({ name: 'Input Data Examples', tab: 'Input Data' });

  // Get the examples
  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0]; // shape[0] 获取这个张量第一个维度的大小，也就是样本数量 20

  // Create a canvas element to render each example
  for (let i = 0; i < numExamples; i++) {
    /**
     *
     * 在 tf.tidy() 的回调函数中，如果创建了中间张量，
     * 这些中间张量在回调函数执行完毕后会被自动清理，
     * 从而释放内存，避免内存泄漏。
     *
     * 具体来说：
     * examples.xs.slice() 操作会创建一个新的中间张量
     * .reshape() 操作也会创建一个新的中间张量
     *
     * 需要注意的是，tf.tidy() 不会清理：
     * 1. 函数返回的张量（这里是最终 reshape 后的 imageTensor）
     * 2. 函数外部创建的张量（如 examples.xs）
     */
    const imageTensor = tf.tidy(() => {
      // Reshape the image to 28x28 px
      return examples.xs
        .slice(
          // [i, 0] 是起始位置，表示从第 i 个样本的开始处切片
          [i, 0],
          // [1, examples.xs.shape[1]] 是切片的大小，表示只取1个样本，长度为 shape[1]
          [1, examples.xs.shape[1]]
        )
        .reshape(
          // 用于单个图像
          // 是将切片后的数据重新整形成 28×28 像素的图像格式
          // 1 是通道数（灰度图像只有一个通道）
          [28, 28, 1]
        );
      /* 
        reshape 之后的样子 
        [ // 28行
            [1], [2], [3], ..., [28],        // 第1行的28个数
            [29], [30], [31], ..., [56],     // 第2行的28个数
            ...
            [...], [...], [...], ..., [784]   // 第28行的28个数
        ] 
            */
    });
    if (i === 0) {
      imageTensor.print();
    }

    const canvas = document.createElement('canvas');
    canvas.width = 28;
    canvas.height = 28;
    canvas.style.cssText = 'margin: 4px;';
    // 用于将张量（tensor）数据渲染到 Canvas 元素上。
    await tf.browser.toPixels(imageTensor, canvas);
    surface.drawArea.appendChild(canvas);

    imageTensor.dispose();
  }
}

/**
 * 构建模型
 * @returns
 */
function getModel() {
  const model = tf.sequential();

  const IMAGE_WIDTH = 28;
  const IMAGE_HEIGHT = 28;
  const IMAGE_CHANNELS = 1;

  /* 
输入图像 (28x28)
    ↓
第一卷积层 (8个特征图)
    ↓
第一池化层 (降维)
    ↓
第二卷积层 (16个特征图)
    ↓
第二池化层 (降维)
    ↓
扁平化层
    ↓
  输出层 
*/

  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(
    // 这一层的主要作用是从原始图像中提取低级特征，如边缘、角点等。经过这一层后，图像数据会被转换成8个特征图，每个特征图代表图像的不同特征。
    /**
     * Output Shape: [batch, 24, 24, 8]
     *              [批次, 高度, 宽度, 特征图数量]
     * 水平方向：28 - 5 + 1 = 24 次移动
     * 垂直方向：28 - 5 + 1 = 24 次移动
     *
     *
     * 参数数量: 208
     * 计算: (5*5*1)*8 + 8 = 208
     * 5x5: 卷积核大小
     * 1: 输入通道数（灰度图像）
     * 8: 过滤器数量
     * +8: 偏置项（8个过滤器，每个过滤器都需要一个偏置项，所以需要8个偏置项）
     */
    tf.layers.conv2d({
      inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS], // 将流入模型第一层的数据的形状 [row, column, depth]
      kernelSize: 5, // 定义卷积核（过滤器）的大小为5×5，这个卷积核会在图像上滑动，每次处理5×5的区域
      filters: 8, // 指定使用8个不同的卷积核，每个卷积核会学习检测不同的特征（如边缘、纹理等），输出将会有8个特征图（feature maps）
      strides: 1, // 指定卷积核在图像上移动的步幅为1，即每次移动一个像素
      activation: 'relu', // 卷积完成后应用于数据的激活函数。使用ReLU（修正线性单元）作为激活函数，将负值转换为0，保持正值不变，帮助模型学习非线性特征
      kernelInitializer: 'varianceScaling' // 定义卷积核权重的初始化方法，'varianceScaling'会根据输入输出的规模自动调整初始权重的范围
    })
  );

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  // 降维
  /**
   * 最大池化层(maxPooling2d)
   *
   * 1. 降维压缩数据
   * 减少数据的空间维度（高度和宽度）
   * 减少后续层需要处理的参数数量
   *
   * 2. 提取特征
   * 通过选择每个池化窗口中的最大值，最大池化层可以提取图像中的显著特征，这些特征对于分类任务非常有用。
   *
   * 3. 增强模型鲁棒性
   * 提高模型的泛化能力
   * 减少过拟合风险
   *
   * 举例：
   * 假设识别手写数字"7"：
   * 原始图像可能有细微的抖动或倾斜
   * 经过最大池化后，保留了"7"的主要特征（横线和斜线）
   * 丢弃了一些细节，使模型更容易识别不同人写的"7"
   */
  /**
   * Output Shape: [batch, 12, 12, 8]
   * - 24x24 → 12x12: 尺寸减半（2x2池化）
   * - 8: 保持特征图数量不变
   *
   * 权重参数数量: 0
   * - 池化层没有可训练参数
   */
  model.add(
    tf.layers.maxPooling2d({
      // 指定一个 2×2 的窗口
      poolSize: [2, 2],
      // 表示窗口每次移动2个单位
      strides: [2, 2]
    })
  );

  /* 最大池化层(maxPooling2d) 计算逻辑示例：
假设输入数据为：
  1  2  3  4
  5  6  7  8
  9  10 11 12
  13 14 15 16

使用 2×2 的最大池化：
  6   8
  14  16 


第一个窗口位置（从左上角开始）：
[1  2] 3  4
[5  6] 7  8
9  10 11 12
13 14 15 16

第二个窗口位置（向右移动2格）：
1  2 [3  4]
5  6 [7  8]
9  10 11 12
13 14 15 16

第四个窗口位置（向右移动2格）：
1  2  3  4
5  6  7  8
9  10 [11 12]
13 14 [15 16]
  */

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  // 第二个卷积层可以基于第一层的输出提取更高级的特征（如形状、部件等）
  // 每一层都在前一层特征的基础上构建更复杂的特征表示
  /**
   * Output Shape: [batch, 8, 8, 16]
   * - 12x12 → 8x8: 再次卷积
   * - 16: 新的特征图数量
   *
   * 权重参数数量: 3,216
   * - 计算: (5*5*8)*16 + 16 = 3,216
   * - 5x5: 卷积核大小
   * - 8: 这个 8 是来自上一层的特征图数量（通道数）
   * - 16: 过滤器数量
   * - +16: 偏置项
   */
  model.add(
    tf.layers.conv2d({
      kernelSize: 5,
      filters: 16, // 增加特征图数量可以让模型捕捉更多不同类型的特征
      strides: 1,
      activation: 'relu',
      kernelInitializer: 'varianceScaling'
    })
  );
  /**
   * Output Shape: [batch, 4, 4, 16]
   * - 8x8 → 4x4: 尺寸再次减半
   * - 16: 保持特征图数量
   */
  model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  /**
   * 展平数据表示法
   * 图片是高维数据，而卷积运算往往会增大传入其中的数据的大小。在将数据传递到最终分类层之前，我们需要将数据展平为一个长数组。
   * 在 flatten 之前：数据是三维的 [height × width × channels]
   * 在 flatten 之后：数据变成一维的 [features]
   *
   * 最后的密集层（dense layer）只接受一维输入
   * 如果不使用 flatten，密集层将无法处理多维输入，会导致模型报错。这是连接卷积层和密集层的必要桥梁。
   */
  /**
   * Output Shape: [batch, 256]
   * - 4*4*16 = 256: 将三维数据展平为一维
   *
   * 参数数量: 0
   * - 展平操作不需要参数
   */
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  /**
   * Output Shape: [batch, 10]
   * - 10: 输出类别数（0-9数字）
   *
   * 参数数量: 2,570
   * - 计算: 256*10 + 10 = 2,570
   * - 256: 输入维度
   * - 10: 输出维度
   * - +10: 偏置项
   */
  model.add(
    // 密集层（我们会用作最终层）只需要采用 tensor1d
    tf.layers.dense({
      units: NUM_OUTPUT_CLASSES, // 0-9 共10个输出
      kernelInitializer: 'varianceScaling',
      activation: 'softmax' // 我们将使用具有 softmax 激活的密集层计算 10 个可能的类的概率分布。得分最高的类将是预测的数字。
      // Softmax 很可能是您要在分类任务的最后一个层中使用的激活函数
    })
  );

  /**
   * 总参数数量: 208 + 0 + 3,216 + 0 + 0 + 2,570 = 5,994
   * - 所有参数都是可训练的（Trainable: true）
   * - 模型结构: 卷积→池化→卷积→池化→展平→全连接
   */

  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  // 编译模型，指定优化器、损失函数和评估指标
  // 编译后模型才能:
  //    1. 使用优化器调整参数
  //    2. 使用损失函数计算误差
  //    3. 使用评估指标评估性能
  // 必须在训练(model.fit())之前完成
  model.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy', // 当模型的输出为概率分布时，就会使用此函数。categoricalCrossentropy 会衡量模型的最后一层生成的概率分布与真实标签提供的概率分布之间的误差。
    metrics: ['accuracy'] // 对于分类问题，这是正确预测在所有预测中所占的百分比
  });
  return model;
}

/**
 * 训练模型
 * @param model 模型
 * @param data 数据
 * @returns
 */
async function train(model: Sequential, data: MnistData) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training',
    tab: 'Model',
    styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 512; // 每批次处理512张图片
  const TRAIN_DATA_SIZE = 5500; // 总共5500张训练图片
  const TEST_DATA_SIZE = 1000; // 总共1000张验证图片

  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [
      d.xs.reshape([
        // 用于批量图像
        TRAIN_DATA_SIZE,
        28,
        28,
        1
      ]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(TEST_DATA_SIZE);
    return [d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]), d.labels];
  });

  // 用 model.fit 来启动训练循环
  return model.fit(
    // trainXs 是"问题"（这张图片是什么？）
    trainXs,
    // rainYs 是"答案"（这张图片代表的数字）
    trainYs,
    {
      validationData: [testXs, testYs], // 验证集
      batchSize: BATCH_SIZE, // 每批次处理512张图片
      // 因此每个训练周期(epoch)会有：5500 ÷ 512 ≈ 11 个批次，10个周期就是 110 个批次
      // 每个批次(batch)处理 512 张图片
      epochs: 10, // 训练轮数
      shuffle: true, // 是否打乱训练数据顺序，有助于模型更好地学习，避免记住数据顺序
      callbacks: fitCallbacks // 回调函数
    }
  );
}

/**
 * 评估模型
 */
const classNames = [
  'Zero',
  'One',
  'Two',
  'Three',
  'Four',
  'Five',
  'Six',
  'Seven',
  'Eight',
  'Nine'
];

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

/**
 * 显示每个类的准确率
 * @param model
 * @param data
 *
 * 1. 用模型预测500张测试图，计算每个数字（0-9）被正确识别的比例
 * 2. 用柱状图显示每个数字的识别准确率
 */
async function showAccuracy(model: Sequential, data: MnistData) {
  // 1. 获取预测结果和实际标签
  const [preds, labels] = doPrediction(model, data);
  // 2. 计算每个类的准确率
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  // 3. 显示每个类的准确率
  const container = { name: 'Accuracy', tab: 'Evaluation' };
  tfvis.show.perClassAccuracy(container, classAccuracy, classNames);
  // 4. 清理内存
  labels.dispose();
}

/**
 * 显示混淆矩阵
 * @param model
 * @param data
 */
async function showConfusion(model: Sequential, data: MnistData) {
  const [preds, labels] = doPrediction(model, data);
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = { name: 'Confusion Matrix', tab: 'Evaluation' };
  tfvis.render.confusionMatrix(container, {
    values: confusionMatrix,
    tickLabels: classNames
  });

  labels.dispose();
}

export async function run() {
  // 加载数据
  const data = new MnistData();
  await data.load();
  // 显示示例图片
  await showExamples(data);
  // 构建模型
  const model = getModel();
  // 显示模型结构
  tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);
  // 训练模型
  await train(model, data);
  // 显示准确率
  await showAccuracy(model, data);
  // 显示混淆矩阵
  await showConfusion(model, data);
}

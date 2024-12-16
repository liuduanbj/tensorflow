/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 784; // 图片大小 (28x28=784像素)
const NUM_CLASSES = 10; // 类别数量（0-9共10个数字）
const NUM_DATASET_ELEMENTS = 65000; // 总数据量

const NUM_TRAIN_ELEMENTS = 55000; // 训练集元素数量
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS; // 测试集元素数量

// MNIST数据集的sprite sheet(精灵图)
// MNIST数据集包含了65000张28x28像素的手写数字图片
// 将图片展平(flatten)成一行：28×28=784像素，所以一张图片就是一行，宽度就是 784，垂直排列高度就是 65000
const MNIST_IMAGES_SPRITE_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';

// MNIST数据集的标签：这个文件包含了对应图片的标签信息（即每张手写数字图片实际代表的数字是什么）
const MNIST_LABELS_PATH =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

/**
 * A class that fetches the sprited MNIST dataset and returns shuffled batches.
 *
 * NOTE: This will get much easier. For now, we do data fetching and
 * manipulation manually.
 *
 * MNIST 手写数字数据集，包含了0-9的手写数字图片
 */
export class MnistData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;
  }

  async load() {
    // Make a request for the MNIST sprited image.
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d', { willReadFrequently: true }); // 可以提高读取操作的性能
    const imgRequest = new Promise((resolve, reject) => {
      img.crossOrigin = '';
      img.onload = () => {
        // 确保图片以1:1的比例显示,不会被拉伸或压缩
        img.width = img.naturalWidth; // 784
        img.height = img.naturalHeight; // 65000

        // 创建一个缓冲区，用于存储所有图片的像素数据，最终存储的是单通道的灰度值
        // ArrayBuffer 是最基础的二进制数据容器，只是一段连续的内存空间，不能直接读写，需要通过视图（TypedArray）来操作
        const datasetBytesBuffer = new ArrayBuffer(
          NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4 // 每个浮点数占用4字节
        );

        const chunkSize = 5000;
        canvas.width = img.width;
        canvas.height = chunkSize;

        // 使用 5000 个一组（chunkSize = 5000）进行分批处理是出于性能和内存考虑：
        // 1. 性能：分批处理可以减少内存占用，避免一次性加载大量数据导致内存不足
        // 2. 内存：分批处理可以减少内存占用，避免一次性加载大量数据导致内存不足
        for (let i = 0; i < NUM_DATASET_ELEMENTS / chunkSize; i++) {
          const datasetBytesView = new Float32Array( // Float32Array 是一种 TypedArray（类型化数组） 可以直接读写数据
            datasetBytesBuffer, // 要操作的 ArrayBuffer
            i * IMAGE_SIZE * chunkSize * 4, // 从哪个位置开始
            IMAGE_SIZE * chunkSize // 要操作多少个元素
          );

          // 将图片绘制到canvas上
          ctx.drawImage(
            img,
            0,
            i * chunkSize,
            img.width,
            chunkSize,
            0,
            0,
            img.width,
            chunkSize
          );

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          for (let j = 0; j < imageData.data.length / 4; j++) {
            // All channels hold an equal value since the image is grayscale, so
            // just read the red channel.
            // 由于 MNIST 数据集的图片是灰度图，意味着 R=G=B，所以只需要读取一个通道的值就够了（这里选择读取红色通道
            datasetBytesView[j] = imageData.data[j * 4] / 255; // 是将 0-255 的值归一化到 0-1 范围
          }
        }
        this.datasetImages = new Float32Array(datasetBytesBuffer);

        resolve();
      };
      img.src = MNIST_IMAGES_SPRITE_PATH;
    });

    const labelsRequest = fetch(MNIST_LABELS_PATH);
    const [imgResponse, labelsResponse] = await Promise.all([
      imgRequest,
      labelsRequest
    ]);

    this.datasetLabels = new Uint8Array(await labelsResponse.arrayBuffer()); // 是一种用于处理8位无符号整数的类型化数组

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    /**
     * 创建一个随机打乱的索引数组 Uint32Array
     *
     * 为什么需要打乱索引？
     * 在机器学习中，随机抽样非常重要
     * 打乱数据可以防止模型学习到数据的顺序特征
     * 有助于提高模型的泛化能力
     * 可以避免模型产生偏差
     */
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    this.trainImages = this.datasetImages.slice(
      0,
      IMAGE_SIZE * NUM_TRAIN_ELEMENTS
    );
    this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    this.trainLabels = this.datasetLabels.slice(
      0,
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
    this.testLabels = this.datasetLabels.slice(
      NUM_CLASSES * NUM_TRAIN_ELEMENTS
    );
  }

  /**
   * 从训练集返回随机批次的图片及其标签
   * @param {*} batchSize
   * @returns
   */
  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize,
      [this.trainImages, this.trainLabels],
      () => {
        // 使用模运算符 % 确保索引在数组长度范围内循环
        this.shuffledTrainIndex =
          (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      }
    );
  }

  /**
   * 从测试集中返回一批图片及其标签
   * @param {*} batchSize
   * @returns
   */
  nextTestBatch(batchSize) {
    return this.nextBatch(batchSize, [this.testImages, this.testLabels], () => {
      this.shuffledTestIndex =
        (this.shuffledTestIndex + 1) % this.testIndices.length;
      return this.testIndices[this.shuffledTestIndex];
    });
  }

  nextBatch(batchSize, data, index) {
    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);

    for (let i = 0; i < batchSize; i++) {
      const idx = index(); // 获取随机索引

      // 提取图像数据
      // 从原始图像数组中切片出一张图片的数据(784个像素)
      const image = data[0].slice(
        idx * IMAGE_SIZE,
        idx * IMAGE_SIZE + IMAGE_SIZE
      );
      // 将这张图片的数据放入批次数组中
      batchImagesArray.set(image, i * IMAGE_SIZE);

      // 提取标签数据
      // 从原始标签数组中切片出一个标签(10个类别的one-hot编码)
      const label = data[1].slice(
        idx * NUM_CLASSES,
        idx * NUM_CLASSES + NUM_CLASSES
      );
      batchLabelsArray.set(label, i * NUM_CLASSES);
    }
    // 将数组转换为TensorFlow.js的张量（2 维数组）
    // xs形状为 [batchSize, 784] - 每行代表一张图片
    // 创建2维张量的：tf.tensor2d(values, shape)
    // values: 输入数据，可以是数组或类型化数组（如Float32Array）
    // shape: 指定张量的形状，是一个包含两个数字的数组 [行数, 列数]
    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    xs.print();
    // labels.print();
    return { xs, labels };
  }
}

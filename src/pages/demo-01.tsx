/**
 * 根据 2D 数据进行预测
 * 
 * 您将训练模型，使之能够根据描述一组汽车的数值数据做出预测。
 * 目标：本练习将演示训练许多不同类型的模型的常见步骤，但将使用小型数据集和简单（浅显）模型。主要目标是帮助您熟悉有关使用 TensorFlow.js 训练模型的基本术语、概念和语法，让您为进一步探索和学习打下良好的基础。
 * 
 * https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html?hl=zh-cn#0
 * 
 * 提供汽车的“马力”，模型将学习预测“每加仑的英里数”(MPG)
 * 
 * 1. 加载数据，并准备将其用于训练。
 * 2. 定义模型的架构。
 * 3. 训练模型并监控其训练时的性能。
 * 4. 通过进行一些预测来评估经过训练的模型。
 */

import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';

console.log('tf', tf);
console.log('tfvis', tfvis);


/**
 * 加载数据
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData() {
    const carsDataResponse = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carsData = await carsDataResponse.json();
    const cleaned = carsData.map((car: Record<string, any>) => ({
        mpg: car.Miles_per_Gallon,
        horsepower: car.Horsepower,
    }))
        .filter((car: Record<string, any>) => (car.mpg != null && car.horsepower != null));

    return cleaned;
}

/**
 * 定义模型架构
 * @returns 
 */
function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // 输入层
    model.add(tf.layers.dense({
        inputShape: [1], 
        units: 50, // 增加到50个神经元
        activation: 'relu',   // 使用 ReLU 激活函数
        useBias: true
    }));

    // 添加多个隐藏层
    model.add(tf.layers.dense({
        units: 50,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 20,
        activation: 'relu'
    }));

    // 输出层
    model.add(tf.layers.dense({
        units: 1,           // 输出层保持1个神经元
        useBias: true
    }));

    return model;
}



/**
 * 准备数据以用于训练
 * 将输入数据转换为张量，以便进行机器学习。我们还将执行重要最佳实践，即对数据进行混洗和归一化。
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
export function convertToTensor(data: Record<string, any>[]) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
        // Step 1. Shuffle the data 重排数据
        // 最佳做法 1：您应始终重排数据，然后再将其传递给 TensorFlow.js 中的训练算法
        tf.util.shuffle(data);

        // Step 2. Convert data to Tensor 转换为2D张量：张量的形状将为 [num_examples, num_features_per_example]。
        const inputs = data.map(d => d.horsepower)
        const labels = data.map(d => d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        console.log('inputTensor', inputTensor.dataSync().length, data.length);
        console.log('labelTensor', labelTensor.dataSync());

        //Step 3. 对数据进行归一化 Normalize the data to the range 0 - 1 using min-max scaling
        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

        console.log('normalizedInputs', normalizedInputs.dataSync());
        console.log('normalizedLabels', normalizedLabels.dataSync());

        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return the min/max bounds so we can use them later.
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
}

// 训练模型
export async function trainModel(model: any, inputs: any, labels: any) {
    // Prepare the model for training. 编译模型
    model.compile({
        optimizer: tf.train.adam(), // 优化器
        loss: tf.losses.meanSquaredError, // meanSquaredError 将模型所做的预测与真实值进行比较
        metrics: ['mse'], // 度量标准：均方误差
    });

    const batchSize = 32; // 模型在每次训练迭代时会看到的数据子集的大小
    const epochs = 50; // 模型将看到整个数据集的次数，我们将对数据集执行 50 次迭代

    // 启动训练循环
    return await model.fit(inputs, labels, {
        batchSize,
        epochs,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] }
        )
    });
}

// 测试模型
function testModel(model: any, inputData: any, normalizationData: any) {
    const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

    // Generate predictions for a uniform range of numbers between 0 and 1;
    // We un-normalize the data by doing the inverse of the min-max scaling
    // that we did earlier.
    const [xs, preds] = tf.tidy(() => {

        // 我们生成了 100 个新“样本”，以提供给模型。
        const xs = tf.linspace(0, 1, 100);
        const preds = model.predict(xs.reshape([100, 1]));

        // 它们必须具有与训练时相似的形状
        const unNormXs = xs
            .mul(inputMax.sub(inputMin))
            .add(inputMin);

        const unNormPreds = preds
            .mul(labelMax.sub(labelMin))
            .add(labelMin);

        // Un-normalize the data
        // 要将数据恢复到原始范围（而非 0-1），我们会使用归一化过程中计算的值，但只是进行逆运算
        return [unNormXs.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPoints = Array.from(xs).map((val, i) => {
        return { x: val, y: preds[i] }
    });

    const originalPoints = inputData.map((d: any) => ({
        x: d.horsepower, y: d.mpg,
    }));

    tfvis.render.scatterplot(
        { name: 'Model Predictions vs Original Data' },
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );
}


export async function run() {
    // Load and plot the original input data that we are going to train on.
    const data = await getData();
    const values = data.map((d: Record<string, any>) => ({
        x: d.horsepower, // 马力
        y: d.mpg, // 每加仑的英里数
    }));

    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        }
    );

    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');

    testModel(model, data, tensorData);
}

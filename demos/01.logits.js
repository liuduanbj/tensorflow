// 1. 加载图片
const img = document.getElementById('myImage');
const tensor = tf.browser.fromPixels(img);

// 2. 预处理图片
const processed = tensor.expandDims(0).toFloat().div(255);

// 3. 进行预测
const predictions = await model.predict(processed).data();

// 4. 处理预测结果
const topPrediction = Array.from(predictions)
  .map((p, i) => ({
    probability: p,
    className: classNames[i]  // 类别名称
  }))
  .sort((a, b) => b.probability - a.probability)[0];

console.log(`这很可能是: ${topPrediction.className}`);
const form = document.getElementById('priceForm');
const resultBox = document.getElementById('result');
const statusBox = document.getElementById('status');

let session = null;

async function loadModel() {
  try {
    statusBox.textContent = 'Loading ONNX model...';
    session = await ort.InferenceSession.create('./house_price_model.onnx');
    statusBox.textContent = 'Model ready';
    statusBox.classList.add('success');
  } catch (error) {
    statusBox.textContent = 'Model failed to load. Make sure house_price_model.onnx is uploaded to the same folder.';
    statusBox.classList.add('error');
    console.error(error);
  }
}

async function predictPrice(inputValues) {
  if (!session) {
    throw new Error('Model not loaded');
  }

  const tensor = new ort.Tensor('float32', new Float32Array(inputValues), [1, inputValues.length]);
  const outputs = await session.run({ input: tensor });

  const outputName = Object.keys(outputs)[0];
  if (!outputName) {
    throw new Error('ONNX model returned no output tensor');
  }

  const price = outputs[outputName].data[0];
  return price;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!session) {
    statusBox.textContent = 'Model is still loading...';
    return;
  }

  const inputValues = [
    Number(document.getElementById('sqft').value),
    Number(document.getElementById('bedrooms').value),
    Number(document.getElementById('bathrooms').value),
    Number(document.getElementById('age').value),
    Number(document.getElementById('garage_spaces').value),
    Number(document.getElementById('lot_size').value),
    Number(document.getElementById('neighborhood_score').value),
  ];

  try {
    const estimatedPrice = await predictPrice(inputValues);
    const formatted = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 0,
    }).format(estimatedPrice);

    resultBox.textContent = `Estimated price: ${formatted}`;
  } catch (error) {
    resultBox.textContent = 'Prediction failed. Try again.';
    console.error(error);
  }
});

loadModel();

import './style.css'
import * as ort from 'onnxruntime-web'

const BASE = import.meta.env.BASE_URL;

document.querySelector('#app').innerHTML = `
  <div class="container">
    <div class="header-logos">
      <img src="${BASE}assets/uet-logo.png" alt="UET logo" class="header-logo" />
      <img src="${BASE}assets/logo-vnu.jpg" alt="VNU logo" class="header-logo" />
    </div>

    <h1>Tomato Leaf Disease Classifier</h1>
    <p class="desc">Upload a tomato leaf image to preview it.</p>

    <div class="warning-box">
      <strong>Notice:</strong>
      This AI model is for reference only. Predictions may be incorrect or uncertain and should not be treated as 100% accurate. Please verify results with expert knowledge or additional inspection when needed.
    </div>

    <div class="toolbar">
      <input type="file" id="imageInput" accept="image/*" />
      <button id="predictBtn" disabled>Predict</button>
      <button id="guideBtn" class="guide-btn">Guide</button>
      <button id="clearBtn" disabled class="secondary-btn">Clear</button>
    </div>

    <div id="guideBox" class="guide-box hidden">
      <h3>How to Use</h3>
      <ol>
        <li>Upload a tomato leaf image from your device.</li>
        <li>Adjust the threshold if needed.</li>
        <li>Click <strong>Predict</strong> to run the model.</li>
        <li>Review the predicted labels and confidence scores.</li>
        <li>Use the results as reference only, since the model may be uncertain or incorrect.</li>
      </ol>
    </div>

    <div class="threshold-box">
      <label for="thresholdRange">Threshold: <span id="thresholdValue">0.50</span></label>
      <input
        type="range"
        id="thresholdRange"
        min="0.05"
        max="0.95"
        step="0.05"
        value="0.50"
      />
    </div>

    <div class="preview-box">
      <img id="previewImage" alt="Preview image" />
    </div>

    <div id="status">Initializing...</div>
    <div id="result"></div>
  </div>
`;

const imageInput = document.getElementById('imageInput');
const previewImage = document.getElementById('previewImage');
const status = document.getElementById('status');
const result = document.getElementById('result');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');
const guideBtn = document.getElementById('guideBtn');
const guideBox = document.getElementById('guideBox');
const thresholdRange = document.getElementById('thresholdRange');
const thresholdValue = document.getElementById('thresholdValue');

const IMAGE_SIZE = 384;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let classNames = [];
let session = null;
let selectedFile = null;

thresholdRange.addEventListener('input', () => {
  thresholdValue.textContent = Number(thresholdRange.value).toFixed(2);
});

guideBtn.addEventListener('click', () => {
  guideBox.classList.toggle('hidden');
});

imageInput.addEventListener('change', (event) => {
  const file = event.target.files[0];

  if (!file) {
    clearSelection();
    return;
  }

  selectedFile = file;
  const imageUrl = URL.createObjectURL(file);
  previewImage.src = imageUrl;
  previewImage.style.display = 'block';

  predictBtn.disabled = false;
  clearBtn.disabled = false;
  result.innerHTML = '';
  status.textContent = `Selected: ${file.name}`;
});

clearBtn.addEventListener('click', () => {
  clearSelection();
  status.textContent = 'Selection cleared.';
});

predictBtn.addEventListener('click', async () => {
  if (!selectedFile || !session) return;

  try {
    predictBtn.disabled = true;
    status.textContent = 'Running inference...';

    const tensor = await imageFileToTensor(selectedFile, IMAGE_SIZE);
    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];

    const feeds = {
      [inputName]: tensor
    };

    const outputs = await session.run(feeds);
    const logits = outputs[outputName].data;
    const threshold = Number(thresholdRange.value);

    const predictions = getMultiLabelPredictions(logits, classNames, threshold);
    renderPredictions(predictions, threshold);

    status.textContent = `Done. Found ${predictions.length} predicted label(s).`;
  } catch (error) {
    console.error(error);
    status.textContent = `Error: ${error.message}`;
  } finally {
    predictBtn.disabled = false;
  }
});

function clearSelection() {
  imageInput.value = '';
  selectedFile = null;
  previewImage.src = '';
  previewImage.style.display = 'none';
  predictBtn.disabled = true;
  clearBtn.disabled = true;
  result.innerHTML = '';
}

async function initApp() {
  try {
    status.textContent = 'Loading classes...';
    classNames = await fetch(`${BASE}assets/class_names.json`).then((res) => res.json());

    status.textContent = 'Loading ONNX model...';
    session = await ort.InferenceSession.create(`${BASE}models/mobilenet_v3.onnx`, {
      executionProviders: ['wasm'],
    });

    console.log('Classes:', classNames);
    console.log('Number of classes:', classNames.length);
    console.log('Inputs:', session.inputNames);
    console.log('Outputs:', session.outputNames);

    status.textContent = `Ready. Loaded ${classNames.length} classes and model successfully.`;
  } catch (error) {
    console.error(error);
    status.textContent = `Error: ${error.message}`;
  }
}

async function imageFileToTensor(file, imageSize) {
  const bitmap = await createImageBitmap(file);

  const canvas = document.createElement('canvas');
  canvas.width = imageSize;
  canvas.height = imageSize;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(bitmap, 0, 0, imageSize, imageSize);

  const imageData = ctx.getImageData(0, 0, imageSize, imageSize).data;

  const floatData = new Float32Array(1 * 3 * imageSize * imageSize);

  for (let y = 0; y < imageSize; y++) {
    for (let x = 0; x < imageSize; x++) {
      const pixelIndex = (y * imageSize + x) * 4;

      const r = imageData[pixelIndex] / 255.0;
      const g = imageData[pixelIndex + 1] / 255.0;
      const b = imageData[pixelIndex + 2] / 255.0;

      const offset = y * imageSize + x;

      floatData[0 * imageSize * imageSize + offset] = (r - MEAN[0]) / STD[0];
      floatData[1 * imageSize * imageSize + offset] = (g - MEAN[1]) / STD[1];
      floatData[2 * imageSize * imageSize + offset] = (b - MEAN[2]) / STD[2];
    }
  }

  return new ort.Tensor('float32', floatData, [1, 3, imageSize, imageSize]);
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function formatLabel(label) {
  return label
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

function getMultiLabelPredictions(logits, classNames, threshold) {
  return Array.from(logits)
    .map((logit, idx) => ({
      label: classNames[idx],
      prob: sigmoid(logit),
    }))
    .filter((item) => item.prob >= threshold)
    .sort((a, b) => b.prob - a.prob);
}

function renderPredictions(predictions, threshold) {
  if (predictions.length === 0) {
    result.innerHTML = `
      <h3>Predictions</h3>
      <p class="no-result">No label exceeded the threshold of <strong>${threshold.toFixed(2)}</strong>.</p>
    `;
    return;
  }

  result.innerHTML = `
    <h3>Predictions</h3>
    <ul class="prediction-list">
      ${predictions
        .map(
          (item) => `
            <li class="prediction-item">
              <div class="prediction-top">
                <span class="prediction-label">${formatLabel(item.label)}</span>
                <span class="prediction-score">${(item.prob * 100).toFixed(2)}%</span>
              </div>
              <div class="progress-bar">
                <div class="progress-fill" style="width: ${(item.prob * 100).toFixed(2)}%"></div>
              </div>
            </li>
          `
        )
        .join('')}
    </ul>
  `;
}

initApp();
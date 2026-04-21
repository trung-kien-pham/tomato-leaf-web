// import './style.css'
// import javascriptLogo from './assets/javascript.svg'
// import viteLogo from './assets/vite.svg'
// import heroImg from './assets/hero.png'
// import { setupCounter } from './counter.js'

// document.querySelector('#app').innerHTML = `
// <section id="center">
//   <div class="hero">
//     <img src="${heroImg}" class="base" width="170" height="179">
//     <img src="${javascriptLogo}" class="framework" alt="JavaScript logo"/>
//     <img src="${viteLogo}" class="vite" alt="Vite logo" />
//   </div>
//   <div>
//     <h1>Get started</h1>
//     <p>Edit <code>src/main.js</code> and save to test <code>HMR</code></p>
//   </div>
//   <button id="counter" type="button" class="counter"></button>
// </section>

// <div class="ticks"></div>

// <section id="next-steps">
//   <div id="docs">
//     <svg class="icon" role="presentation" aria-hidden="true"><use href="/icons.svg#documentation-icon"></use></svg>
//     <h2>Documentation</h2>
//     <p>Your questions, answered</p>
//     <ul>
//       <li>
//         <a href="https://vite.dev/" target="_blank">
//           <img class="logo" src="${viteLogo}" alt="" />
//           Explore Vite
//         </a>
//       </li>
//       <li>
//         <a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank">
//           <img class="button-icon" src="${javascriptLogo}" alt="">
//           Learn more
//         </a>
//       </li>
//     </ul>
//   </div>
//   <div id="social">
//     <svg class="icon" role="presentation" aria-hidden="true"><use href="/icons.svg#social-icon"></use></svg>
//     <h2>Connect with us</h2>
//     <p>Join the Vite community</p>
//     <ul>
//       <li><a href="https://github.com/vitejs/vite" target="_blank"><svg class="button-icon" role="presentation" aria-hidden="true"><use href="/icons.svg#github-icon"></use></svg>GitHub</a></li>
//       <li><a href="https://chat.vite.dev/" target="_blank"><svg class="button-icon" role="presentation" aria-hidden="true"><use href="/icons.svg#discord-icon"></use></svg>Discord</a></li>
//       <li><a href="https://x.com/vite_js" target="_blank"><svg class="button-icon" role="presentation" aria-hidden="true"><use href="/icons.svg#x-icon"></use></svg>X.com</a></li>
//       <li><a href="https://bsky.app/profile/vite.dev" target="_blank"><svg class="button-icon" role="presentation" aria-hidden="true"><use href="/icons.svg#bluesky-icon"></use></svg>Bluesky</a></li>
//     </ul>
//   </div>
// </section>

// <div class="ticks"></div>
// <section id="spacer"></section>
// `

// setupCounter(document.querySelector('#counter'))

import './style.css'
import * as ort from 'onnxruntime-web'

const BASE = import.meta.env.BASE_URL;

document.querySelector('#app').innerHTML = `
  <div class="container">
    <div class="header-logos">
      <img src="${BASE}assets/logo-vnu.jpg" alt="School logo" class="header-logo" />
      <img src="${BASE}assets/uet-logo.png" alt="School logo" class="header-logo" />
    </div>

    <h1>Tomato Leaf Disease Classification</h1>
    <p class="desc">Upload a tomato leaf image to preview it.</p>

    <div class="warning-box">
      <strong>Notice:</strong>
      This AI model is for reference only. Predictions may be incorrect or uncertain and should not be treated as 100% accurate. Please verify results with expert knowledge or additional inspection when needed.
    </div>

    <div class="toolbar">
      <input type="file" id="imageInput" accept="image/*" />
      <button id="predictBtn" disabled>Predict</button>
      <button id="clearBtn" disabled class="secondary-btn">Clear</button>
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

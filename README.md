# Tomato Leaf Disease Classifier Web App

A web-based demo for **multi-label tomato leaf disease classification** using **ONNX Runtime Web** and **Vite**.  
This project allows users to upload a tomato leaf image, run inference directly in the browser, and view predicted labels with confidence scores.

## Author
**Trung Kien Pham**

## Live Demo
[Open the web app](https://trung-kien-pham.github.io/tomato-leaf-web/)

## Features
- Upload and preview tomato leaf images
- Multi-label disease classification in the browser
- Adjustable prediction threshold
- Confidence score visualization with progress bars
- Simple and user-friendly interface
- Built-in usage guide
- Warning notice for model uncertainty

## Tech Stack
- **Vite**
- **JavaScript**
- **ONNX Runtime Web**
- **HTML/CSS**

## Project Structure
```text
tomato-leaf-web/
├─ public/
│  ├─ assets/
│  │  ├─ class_names.json
│  │  ├─ logo-vnu.jpg
│  │  └─ uet-logo.png
│  └─ models/
│     └─ mobilenet_v3.onnx
├─ src/
│  ├─ main.js
│  └─ style.css
├─ index.html
├─ package.json
└─ vite.config.js
```

## How It Works
1. Upload a tomato leaf image from your device.
2. Adjust the prediction threshold if needed.
3. Click **Predict** to run the model.
4. Review the predicted labels and confidence scores.
5. Use the output as reference only.

## Installation and Local Run

### 1. Clone the repository
```bash
git clone https://github.com/trung-kien-pham/tomato-leaf-web.git
cd tomato-leaf-web
```

### 2. Install dependencies
```bash
npm install
```

### 3. Run the development server
```bash
npm run dev
```

### 4. Build for production
```bash
npm run build
```

## Deployment
This project is deployed with **GitHub Pages** using **GitHub Actions**.

## Model Information
- Model format: **ONNX**
- Inference engine: **ONNX Runtime Web**
- Task: **Multi-label classification**
- Input size: **384 × 384**
- Normalization:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

## Notes
- The model runs entirely in the browser.
- No backend server is required for prediction.
- Results may vary depending on image quality, lighting conditions, and leaf appearance.

## Disclaimer
This AI model is intended for **reference only**.  
Predictions may be incorrect or uncertain and should **not** be considered 100% reliable.  
Users should verify results with domain knowledge, expert consultation, or additional inspection before making decisions.

## Future Improvements
- Better mobile responsiveness
- Support for multiple model options
- Improved result explanations
- More polished UI/UX

## License
This project is licensed under the MIT License.
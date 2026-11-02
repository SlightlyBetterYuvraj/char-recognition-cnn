# ✍️ Handwritten Character Recognition using CNN

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-EMNIST%20Letters-success.svg)](https://www.kaggle.com/datasets/crawford/emnist)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-94.08%25-brightgreen.svg)]()

An interactive, deep learning-powered handwriting recognition web application. The system classifies handwritten English alphabet characters (**A – Z**) in real-time from a sleek drawing canvas using a Convolutional Neural Network (CNN) trained on the **EMNIST Letters** dataset.

---

## ✨ Features

- 🎨 **Interactive Drawing Canvas**: Smooth, responsive freehand drawing canvas with adjustable stroke thickness and instant clear functionality.
- ⚡ **Real-Time Classification**: Live inference powered by a trained Keras CNN model as you sketch.
- 📐 **Intelligent Preprocessing Pipeline**:
  - Automatically crops stroke bounding boxes.
  - Scales glyphs while preserving aspect ratio inside a 20×20 bounding box.
  - Centers the glyph inside the standard 28×28 EMNIST frame for maximum prediction accuracy.
- 📊 **Confidence & Ranking Breakdown**: Displays the top predicted letter, confidence percentage, confidence tier (High / Moderate / Ambiguous), and a ranked probability breakdown of the top candidate letters.
- 🔍 **Input Tensor Inspector**: Visualizes the exact preprocessed 28×28 grayscale array that the CNN evaluates.

---

## 🧠 Model Architecture & Performance

The model was developed and trained in [`main.ipynb`](main.ipynb) using TensorFlow/Keras on the **EMNIST Letters** dataset (145,600 handwritten character samples across 26 balanced classes, A–Z).

### Network Architecture:
```text
Input: (28, 28, 1) Grayscale
  │
  ├── Conv2D (32 filters, 3x3 kernel, ReLU activation)
  ├── MaxPooling2D (2x2 pool size)
  │
  ├── Conv2D (32 filters, 3x3 kernel, ReLU activation)
  ├── MaxPooling2D (2x2 pool size)
  │
  ├── Flatten (800 features)
  ├── Dense (256 units, ReLU activation)
  ├── Dropout (rate = 0.6)
  │
  └── Dense (26 units, Softmax activation) -> Output: Classes A-Z
```

### Training Highlights:
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 20
- **Validation Accuracy**: **~94.08%**
- **Model File**: Stored in standard Keras format as [`emnist_cnn_model2.keras`](emnist_cnn_model2.keras).

---

## 📁 Repository Structure

```text
char-recognition-cnn/
├── app.py                      # Interactive Streamlit web application
├── emnist_cnn_model2.keras     # Trained CNN model weights & architecture
├── main.ipynb                  # Training, preprocessing & evaluation notebook
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (ignores raw dataset & caches)
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/SlightlyBetterYuvraj/char-recognition-cnn.git
cd char-recognition-cnn
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the drawing canvas and live character recognizer!

---

## 📓 Model Training Notebook

To retrain the model or reproduce experiments:
1. Open [`main.ipynb`](main.ipynb) in Jupyter Lab, VS Code, or Google Colab.
2. Provide your Kaggle credentials via environment variables or prompt when prompted by the notebook to download the EMNIST dataset:
   ```bash
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   ```
3. Execute the cells to download data, apply orientation transformations (EMNIST transpose fix), train the CNN, and save the model.

---

## 🤖 AI Usage Disclosure

In alignment with transparent development practices:
- **Artificial Intelligence (Google DeepMind Antigravity)** was utilized to assist in designing the **Streamlit user interface (`app.py`)**, including the glassmorphism styling, responsive dual-column grid layout, bounding-box centering preprocessing pipeline, and interactive canvas integration.
- The underlying machine learning model, training workflow in [`main.ipynb`](main.ipynb), and hyperparameter tuning were developed by the project author.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

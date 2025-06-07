# DEEP_LEARNING_PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: UPASANA PRAJAPATI

*INTERN ID*: CT08DF387

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH


*DESCRIPTION OF THE TASK*:

Based on the images and script (`image_classification.py`) you provided, here's a complete GitHub README file description for your deep learning image classification project using CIFAR-10:

---

# 🧠 Deep Learning CIFAR-10 Image Classification

This project is a **Convolutional Neural Network (CNN)** implementation using PyTorch for classifying images from the **CIFAR-10 dataset**, which consists of 60,000 32x32 color images in 10 different classes such as plane, car, bird, cat, etc.

---

## 📂 Project Structure

* **`image_classification.py`**: Main script that defines the CNN architecture, trains the model, evaluates it, and visualizes performance.
* **Training & Validation Graphs**:

  * Shows how loss decreases and accuracy improves across epochs for both training and validation datasets.
* **Sample Image Visualizations**:

  * Displays a few images with their actual and predicted labels for model prediction evaluation.
* **Console Output Logs**:

  * Displays training progress, per-class accuracy, and overall model accuracy.

---

## 🧠 Model Architecture

```
Conv2d(3 → 6, kernel_size=5) → ReLU → MaxPool2d
Conv2d(6 → 16, kernel_size=5) → ReLU → MaxPool2d
Flatten → 
Linear(400 → 120) → ReLU → 
Linear(120 → 84) → ReLU → 
Linear(84 → 10) → Output
```

---

## 📊 Training Results

* **Final Validation Accuracy**: \~62%
* **Epochs**: 10
* **Optimizer**: SGD (lr=0.001, momentum=0.9)
* **Loss Function**: CrossEntropyLoss

### 🔎 Accuracy Per Class:

| Class | Accuracy |
| ----- | -------- |
| Plane | 59.5%    |
| Car   | 75.8%    |
| Bird  | 53.4%    |
| Cat   | 39.3%    |
| Deer  | 67.4%    |
| Dog   | 45.5%    |
| Frog  | 69.4%    |
| Horse | 63.6%    |
| Ship  | 75.1%    |
| Truck | 73.9%    |

---

## 📈 Performance Graphs

### Loss Graph

Displays decreasing training and validation loss over 10 epochs.

### Accuracy Graph

Shows the steady increase in training and validation accuracy.

---

## 🖼️ Sample Predictions

Model predictions on random CIFAR-10 test images are visualized. Green labels indicate correct predictions, red labels indicate incorrect ones.

---

## 🚀 How to Run

```bash
# Clone the repository and navigate to the folder
git clone <your-repo-url>
cd DEEP_LEARNING_PROJECT

# Set up the virtual environment
python -m venv venv_dl_project
.\venv_dl_project\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the classifier
python image_classification.py
```

---

## 🛠️ Requirements

* Python 3.10+
* PyTorch
* torchvision
* matplotlib
* numpy

---

## 📌 Notes

* The project uses CPU for training by default. If CUDA is available, it will automatically switch to GPU.
---


# 🍎 Automatic Fruit and Vegetable Classification using YOLOv8 and Computer Vision

This repository implements an intelligent fruit and vegetable classification system designed for **self-service weighing stations** in supermarkets.  
It uses a **YOLOv8s** image classifier (classification mode) for compact, near–real-time prediction, trained on the public *Fruits & Vegetable Detection for YOLOv4* dataset.

---

## 🛠 Customizations and Development Effort

This repository extends the standard YOLOv8 classification workflow with several project-specific customizations.

## 1️⃣ Adapting a detection dataset for classification

The original Fruits & Vegetable Detection for YOLOv4 dataset is intended for object detection.
We implement a preparation script that:

📂 Reads images from the Kaggle folders

🏷️ Extracts the class label from the filename using regular expressions

🧾 Builds a DataFrame with image paths and categorical labels suitable for classification

## 2️⃣ Automatic “Bag” attribute (bagged / unbagged)

Instead of manually annotating packaging, we infer bag status from filename patterns:

wb ➜ with bag

wob ➜ without bag

From this, we create a binary Bag column, which is then mapped to final labels such as
banana_with_bag and banana_without_bag.

## 3️⃣ Two label taxonomies: 8 classes and 14 classes

8-class configuration: ignores bag information and focuses only on the product type.

14-class configuration: six products have both with_bag and without_bag variants
(chilli, lemon, banana, apple, tomato, grapes), while raspberry and blackberries remain only without bag.

The mapping logic for both settings is implemented directly in the dataset preparation code.

## 4️⃣ Automatic folder structure for YOLOv8 classification

Starting from the DataFrame, the code:

🔀 Performs a stratified split (70% train, 30% validation)

📁 Copies images into the YOLO classification layout:

dataset/train/<class>/...

dataset/val/<class>/...

This removes the need for manual directory organization.

## 5️⃣ Custom training and evaluation pipeline

We fine-tune the pretrained yolov8s-cls.pt model with:

imgsz = 224

batch = 16

A chosen number of epochs

The pipeline:

📈 Uses the Ultralytics training loop and logger to track loss and accuracy

💾 Saves and reloads the best checkpoint (best.pt) for evaluation

🔍 Runs inference on all validation images to compute:

Accuracy

Precision

Recall

F1-score

Confusion matrix and per-class PRF plots

## 6️⃣ Visual robustness stress test

A dedicated function:

🖼️ Loads a validation image

🌞 Generates bright, 🌚 dark, and ⬛ occluded (black square) versions

🤖 Runs YOLOv8s on each variant and displays them in a grid with predicted label and confidence

This provides qualitative evidence of robustness to illumination changes and partial occlusions.

## 7️⃣ Latency and model-size measurement

⏱ Benchmarks inference time (in milliseconds) over multiple runs for a single image

💽 Computes the size of the best.pt file (in MB)

These measurements support analysis of feasibility for edge and embedded deployment in supermarket weighing stations.


## 🧰 Requirements

Install dependencies:

```bash
pip install ultralytics opencv-python scikit-learn matplotlib pandas numpy


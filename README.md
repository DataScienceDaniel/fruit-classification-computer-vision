# 🍎 Automatic Fruit and Vegetable Classification using YOLOv8 and Computer Vision

This repository implements an intelligent fruit and vegetable classification system designed for **self-service weighing stations** in supermarkets.  
It uses **YOLOv8s** for compact, real-time image classification, trained on the public *Fruits & Vegetable Detection for YOLOv4* dataset.

---

## 📦 Features
- ✅ Multi-class classification for 7 categories (chilli, lemon, banana, apple, tomato, grapes, raspberry, blackberries)  
- 🧺 “Bagged” vs. “Unbagged” item annotation using filename pattern detection  
- ⚡ Fine-tuned YOLOv8s classifier with support for GPU acceleration (CUDA)  
- 📊 Evaluation pipeline: confusion matrix, precision–recall–F1 visualization, and most confused label pairs  
- 🧠 Modular Python scripts for dataset preparation, training, and inference  

---

## 🧰 Requirements
Install dependencies:
```bash
pip install ultralytics opencv-python scikit-learn matplotlib pandas numpy ```bash

## 🍎 Colab
https://colab.research.google.com/drive/1JV254KhEbcmoKXaLtrxvacwKy2Ll5Wc1?usp=sharing

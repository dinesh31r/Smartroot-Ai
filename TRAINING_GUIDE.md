# Model Training Guide - PlantVillage

This guide explains how to download the **PlantVillage** dataset and train the **MobileNetV2** model for use in SmartRoot-AI.

## 1. Download the Dataset

The best source for the PlantVillage dataset is **Kaggle**.

- **URL**: [Kaggle - PlantVillage Dataset](https://www.kaggle.com/datasets/emmareed/plantvillage-dataset)
- **Instructions**:
  1. Download the ZIP file.
  2. Create a folder named `data/plantvillage` in your project root.
  3. Extract the ZIP so that the category folders (e.g., `Tomato_healthy`, `Apple_Black_rot`) are directly inside `data/plantvillage/`.

## 2. Set Up Environment

Ensure you have the necessary libraries installed:

```bash
pip install tensorflow numpy pillow
```

## 3. Train the Model

I have provided a training script: `scripts/train_plantvillage.py`.

Run it using:

```bash
python3 scripts/train_plantvillage.py
```

### What the script does:
- Loads the images from `data/plantvillage`.
- Applies **Data Augmentation** (rotation, zoom, flips) to prevent overfitting.
- Uses **Transfer Learning** with MobileNetV2 pretrained on ImageNet for high accuracy with little data.
- Saves the final model as `model/mobilenetv2_plantvillage.h5`.

## 4. Activation in SmartRoot-AI

Once the `.h5` file is in the `model/` folder, SmartRoot-AI will detect it **automatically** and use it for predictions instead of the fallback model.

---

> [!TIP]
> **GPU Training**: If you have an NVIDIA GPU, ensure `tensorflow[and-cuda]` is installed for 10x faster training.

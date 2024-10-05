# Fashion Product Image Classification using Vision Transformer (ViT)

This project implements an **Image Classification** pipeline using a Vision Transformer (ViT) to classify fashion products from the **Fashion Products Small** dataset. The ViT model leverages transformer architecture, commonly used in NLP, for image classification.

## Dataset
The **Fashion Products Small** dataset from Hugging Face is used in this project. The dataset contains images of various fashion items categorized into 6 "masterCategory" labels:
- **Apparel**, **Footwear**, **Accessories**, **Personal Care**, **Free Items**, and **Sporting Goods**.

In this project:
- The images are resized to **224x224** pixels to meet the Vision Transformer input requirements.
- The pixel values are **normalized** based on the mean and standard deviation used during the model's pretraining.

## Model Architecture
### 1. Vision Transformer (ViT)
The **Vision Transformer (ViT)** is used as the image classification model. It transforms images into patches and processes them as sequences using transformer encoder layers:
- **Input**: The images are divided into **16x16 patches**, resulting in a sequence of 196 patches for an image of size 224x224.
- **Transformer Encoder Layers**: The model consists of **12 transformer encoder layers**, each with **12 attention heads** and a hidden size of **768**.
- **Output**: A classification prediction that corresponds to one of the 6 fashion categories.

The ViT model is **pretrained** on the large-scale **ImageNet-21k** dataset, and fine-tuned on the fashion dataset to classify images into the defined categories.

## Loss Function
The model uses **Cross-Entropy Loss** (handled internally by the Hugging Face `Trainer` class) for the classification task:
- It measures the difference between the predicted probabilities and the actual category labels.
- During training, the model optimizes this loss to improve its predictions for fashion categories.

## Training Procedure
- The training process runs for **3 epochs**.
- In each training loop:
  1. **Data Loading**: The images are loaded in batches, resized to 224x224, and normalized. Data augmentation is applied to the training set through random horizontal flips.
  2. **Model Training**: The Vision Transformer is fine-tuned on the fashion dataset, using the AdamW optimizer with a learning rate of `2e-5`.
  3. **Evaluation**: The model's accuracy on the test set is calculated at the end of each epoch.

- **Loss values** and **accuracy** are tracked throughout the training, and the model is evaluated on the test dataset to measure its classification performance.

- **Best Model Saving**: The model with the highest accuracy is saved during training and stored as `vit_model.pth` in the `./results` directory.

## Results
- The model achieves increasingly better classification results as training progresses, and its ability to correctly classify fashion items improves.
- **Predicted Labels**: The model outputs predictions for the master categories of fashion products. Accuracy is the primary metric used to evaluate the model's performance.
- **Model Checkpoints**: The final model weights are saved as `vit_model.pth` in the `./results` directory for future use or fine-tuning.

The saved model can be loaded to make new predictions on unseen images or to continue training from the last checkpoint.

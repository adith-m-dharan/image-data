# GAN for CIFAR-10 Image Generation

This project implements a **Generative Adversarial Network (GAN)** using PyTorch to generate images that resemble those in the CIFAR-10 dataset. The GAN consists of two main components:
- A **Generator** that creates synthetic images from random noise.
- A **Discriminator** that attempts to classify images as real (from the dataset) or fake (created by the Generator).

## Dataset
The **CIFAR-10** dataset is used in this project. It contains 60,000 32x32 pixel color images across 10 different categories:
- Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, and Truck.

In this project:
- The images are resized to **64x64** pixels.
- They are also **normalized** to a range of [-1, 1] to help with the stability of GAN training, a common practice in deep learning.

## Model Architectures
### 1. Generator
The **Generator** is responsible for creating fake images from a random input noise vector. It uses a series of **transposed convolutional layers** to upsample the noise into an image:
- **Input**: Random noise of size 100x1x1.
- **Layers**: Transposed convolution layers with **batch normalization** and **ReLU activations**.
- **Output**: A 3x64x64 image with pixel values in the range of [-1, 1] (using **Tanh** activation at the output layer).

The Generator is trained to learn how to produce images that look like real CIFAR-10 images, tricking the Discriminator.

### 2. Discriminator
The **Discriminator** acts as a binary classifier, attempting to distinguish real CIFAR-10 images from the fake images produced by the Generator:
- **Input**: A 3x64x64 image.
- **Layers**: Standard convolution layers with **LeakyReLU activations** and **batch normalization**.
- **Output**: A single probability value (using **Sigmoid** activation) representing whether the input image is real or fake.

The Discriminator is trained to correctly classify real images as real and fake images as fake.

## Loss Function
Both models use **Binary Cross-Entropy Loss (BCELoss)**:
- For the **Discriminator**: It tries to **maximize** its ability to correctly classify real images as real and fake images as fake. The loss is calculated by comparing the Discriminator's output with the true labels.
- For the **Generator**: It tries to **minimize** the Discriminator's ability to detect fake images. Essentially, the Generator’s goal is to make the Discriminator classify its generated images as real.

The training is a competition between the two models: as the Generator improves at creating realistic images, the Discriminator must improve at distinguishing real from fake.

## Training Procedure
- The training process runs for **25 epochs**.
- In each training loop:
  1. **Discriminator Update**: The Discriminator is trained to classify real images as real and fake images as fake. Label smoothing and small noise are applied to labels to improve stability.
  2. **Generator Update**: The Generator is updated to try and fool the Discriminator by generating images that are classified as real.
  
- **Loss values** for both the Generator and the Discriminator are tracked and printed every 100 steps.

- **Generated images** and **real images** are saved as `.png` files in the `./results` directory during training, allowing you to visually track the progress of the Generator.

- At the end of training, both the **Generator** and **Discriminator** models are saved as `netG.pth` and `netD.pth` in the `./results` directory.

## Results
- Over time, the **Generator** improves its ability to produce images that look increasingly similar to the real CIFAR-10 dataset.
- You can visually inspect the saved images in the `./results` directory to see how the quality of generated images evolves during training.

The final weights of the Generator and Discriminator models can be used for future tasks, such as generating more images or continuing training from where it was left off.

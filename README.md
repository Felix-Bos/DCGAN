# DCGAN with PyTorch — Image and Digit Generation

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch to generate synthetic images.  
The model is trained on two datasets: handwritten digits (MNIST) and human faces (CelebA), demonstrating the ability of GANs to learn visual structure and generate realistic samples.

---

## Overview

A DCGAN consists of:

- **Generator:** learns to produce images from random noise (latent space).
- **Discriminator:** learns to differentiate between real dataset samples and generated ones.

Through adversarial training, both networks improve simultaneously, leading to increasingly realistic generated images.

---

## Features

- DCGAN architecture fully implemented in PyTorch
- Training on MNIST and CelebA datasets
- Visualization of training evolution and generated samples
- Latent space exploration and image sampling

---


# Movie Poster Genre Classification

This project is designed to classify movie genres based on movie posters using advanced deep learning models. The primary models used are Vision Transformer (ViT) and ResNet50.

## Overview

The goal of this project is to predict the genre of a movie by analyzing its poster. By leveraging state-of-the-art image classification models, we aim to achieve high accuracy in genre classification.

## Models Used

1. **Vision Transformer (ViT)**:
   - ViT models treat an image as a sequence of patches and process them using transformer architecture.
   - This model can capture complex relationships within the image, making it suitable for tasks like genre classification.

2. **ResNet**:
   - ResNet is a convolutional neural network (CNN) that is widely used for image classification tasks.
   - By using residual learning, ResNet can effectively train very deep networks.

## Dataset

The dataset consists of movie posters along with their corresponding genres. The genres are encoded as multi-labels since a movie can belong to more than one genre.
download：https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

## Prerequisites

To run this project, you need the following dependencies:

- Python 3.11
- PyTorch with gpu
- Transformers
- Scikit-learn
- Pillow
- Matplotlib
- TQDM
.......


## data processing

run data_check.py

## poster downloading

run download.py

## train 

run vit_train.py
or resnet_train.py

## predict

In RESNET/VIT 
run the predict scripts.

have a great day!!





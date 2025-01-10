### Image-Captioning

## Architecture
<img width="836" alt="image" src="https://github.com/user-attachments/assets/6a770821-0e79-4e39-bff0-08ba91acb8c8" />


This project demonstrates an image captioning system that generates textual descriptions for images. The model utilizes a combination of Convolutional Neural Networks (CNN) for feature extraction and Long Short-Term Memory (LSTM) for sequence generation to predict captions for a given image.

## Overview
The goal of this project is to create a system capable of analyzing an image and generating a human-readable description based on its content. The model employs the following architecture:

CNN (Convolutional Neural Network):
DenseNet - a pre-trained model on Imagenet dataset (having Json format)
Extracts high-level features from the image. A pre-trained CNN model such as InceptionV3 or ResNet is used as a feature extractor.
LSTM (Long Short-Term Memory):
Once the CNN extracts features from the image, these features are passed to an LSTM network, which generates a sequence of words to form a caption.
The LSTM is trained on sequences of words to model the temporal dependencies between them, allowing the generation of grammatically correct captions.
Features

<img width="408" alt="image" src="https://github.com/user-attachments/assets/fd17992a-31db-4847-8e06-fe6f916cb099" />


Image Upload: Upload an image via the web interface and generate a caption for the uploaded image.
Model Architecture: A hybrid CNN-LSTM architecture that combines image processing and language modeling for caption generation.
Pre-trained Models: Pre-trained models like InceptionV3 for feature extraction are fine-tuned to work with your image data.
Real-time Caption Generation: The application provides real-time captions for uploaded images using the trained model.
## Output
<img width="842" alt="image" src="https://github.com/user-attachments/assets/692d1f49-3ba3-4258-a04a-f9c574867c32" />

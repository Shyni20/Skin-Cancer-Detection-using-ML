# Skin Cancer Detection using Machine Learning

Early detection of skin cancer is critical, yet manual visual inspection is error-prone and requires expert knowledge. This project implements a Machine Learning–based system to classify skin lesions as **benign** or **malignant** using dermoscopic images.

The model is trained on publicly available Kaggle datasets (HAM10000) containing real-world skin lesion images. A Convolutional Neural Network (CNN) is used to automatically learn visual patterns such as texture, color variation, and lesion shape.

## Objectives
- Automate early skin cancer detection  
- Reduce dependency on manual diagnosis  
- Provide a fast and scalable ML-based solution  

## Dataset
- Source: Kaggle – *Skin Cancer MNIST (HAM10000)*  
- Classes: Benign, Malignant  
- Data Type: RGB dermoscopic images  
- Preprocessing:
  - Image resizing to 64×64
  - Normalization (0–1 range)
  - Label encoding

## My Role
- Collected image data from Kaggle and other open sources  
- Organized datasets into class-wise folders  
- Cleaned and validated image data  
- Prepared data pipeline for training and testing  
- Ensured dataset quality for reliable model learning  

## Model Architecture
- Convolutional Neural Network (CNN)
- Layers:
  - Conv2D + ReLU
  - MaxPooling
  - Flatten
  - Dense + Dropout
  - Softmax Output (2 classes)

## Workflow
1. Dataset collection & organization  
2. Image preprocessing  
3. Train-test split  
4. CNN model building  
5. Training & validation  
6. Evaluation & model saving  

## Results
The trained model achieves reliable accuracy in distinguishing benign and malignant lesions, demonstrating the effectiveness of CNNs for medical image classification.

## Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib  
- Scikit-learn  

## How to Run
```bash
pip install tensorflow opencv-python matplotlib pandas scikit-learn
python train.py

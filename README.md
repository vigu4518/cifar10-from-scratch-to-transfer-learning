# cifar10-from-scratch-to-transfer-learning
Deep learning on CIFAR-10: from ANN and custom CNNs to data augmentation, learning rate tuning, ResNet transfer learning, and ensemble models.


This project demonstrates multiple deep learning approaches for image classification using the CIFAR-10 dataset. It starts with a basic Artificial Neural Network (ANN), builds up to custom CNNs, and incorporates advanced techniques like data augmentation, learning rate tuning, transfer learning using ResNet20, and model ensembling.


## ✅ Features

- ✅ ANN baseline model
- ✅ Custom CNNs with 2 and 3 convolutional layers
- ✅ Dropout regularization to reduce overfitting
- ✅ Data augmentation using `ImageDataGenerator`
- ✅ Learning rate tuning using `ReduceLROnPlateau`
- ✅ Transfer learning using ResNet20 (adapted for CIFAR-10)
- ✅ Ensemble learning by averaging model predictions
- ✅ Classification report & confusion matrix for evaluation

---

## 📦 Dataset

- **Name:** CIFAR-10  
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)  
- **Size:** 60,000 color images (50,000 training + 10,000 testing)  
- **Shape:** 32x32 pixels, 3 channels (RGB)  

```python
from tensorflow.keras.datasets import cifar10

Models Trained
Model	Description	Accuracy
ANN	Flatten + Dense layers	~49%
CNN (2 Conv Layers)	With ReLU,	~70%
CNN (3 Conv Layers)	Deeper model 	~71%
CNN(3 Conv Layers + Augmentation	Real-time data augmentation	~74%
CNN + LR Scheduler	With ReduceLROnPlateau+Augmentation	~80%
ResNet20 (Transfer)	Residual network adapted for CIFAR-10	~82%
Ensemble of models	Averaged predictions	~81%



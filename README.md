# Oxford-IIIT-Pet-Classification-VGG16-vs-ViT-


This project compares the performance of **VGG16** and **Vision Transformer (ViT-B/16)** on the **Oxford-IIIT Pet Dataset**.  
It includes:
- **Custom PyTorch Dataset** for image loading
- **Data augmentation pipeline** (light, medium, heavy) using `torchvision.transforms`
- **Preprocessing function** with train/valid split loaders
- **VGG16 model** (pretrained, with frozen conv layers)
- **Vision Transformer (ViT-B/16)** model (pretrained, with frozen encoder)
- **Evaluation utilities** with accuracy, classification report, confusion matrix, and training curves

#Dataset
We use the **Oxford-IIIT Pet Dataset**, which contains **37 categories** of pets with roughly 200 images per class.  
Each image has a class label (breed).  

- **Train split file:** `trainval.txt`  
- **Test split file:** `test.txt`  
- **Images folder:** `/content/images/images`

#Usage
git clone https://github.com/YOUR-USERNAME/pet-classification.git
cd pet-classification

#Model Architectures
**VGG16**
Pretrained on ImageNet
Frozen convolutional layers
Replaced classifier head with custom nn.Linear
Optimizer: Adam
Loss: CrossEntropyLoss

**ViT-B/16**
Pretrained on ImageNet
Frozen encoder layers
Replaced classification head
Optimizer: Adam
Loss: CrossEntropyLoss

#Results
Evaluation metrics:
Accuracy on test set
Precision, Recall, F1-score per class
Confusion Matrix (heatmap)

Training curves:
Loss (train vs valid)
Accuracy (train vs valid)

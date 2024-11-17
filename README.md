 # ML Model CI/CD Pipeline

 [![ML Pipeline](https://github.com/garima-mahato/ERAV3_Session5_NNAndMLOps/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/garima-mahato/ERAV3_Session5_NNAndMLOps/actions/workflows/ml-pipeline.yml)

This repository demonstrates a CI/CD pipeline for a simple Deep Neural Network trained on the MNIST dataset. The project includes automated testing, model validation, and a deployment process using GitHub Actions.

## Project Structure

```
.
├── README.md
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml
├── src/
│ ├── init.py
│ ├── model.py
│ ├── train.py
│ └── test_model.py
```


## Model Architecture
- CNN with below architecture:
```
├── Input Block
│ ├── 3x3 Conv2D (16 channels)
│ ├── ReLU
│ ├── BatchNorm2d
│ └── Dropout
├── Convolutional Block 1
│ ├── 3x3 Conv2D (24 channels)
│ ├── ReLU
│ ├── BatchNorm2d
│ └── Dropout
├── Transition Block 1
│ ├── 1x1 Conv2D (8 channels)
│ └── MaxPool2d
├── Convolutional Block 2
│ ├── Convolution Layer
│ │   ├── 3x3 Conv2D (16 channels)   
│ │   ├── ReLU
│ │   ├── BatchNorm2d
│ │   └── Dropout
│ ├── Convolution Layer
│ │   ├── 3x3 Conv2D (32 channels)   
│ │   ├── ReLU
│ │   ├── BatchNorm2d
│ │   └── Dropout
│ ├── Convolution Layer
│ │   ├── 3x3 Conv2D (48 channels)   
│ │   ├── ReLU
│ │   ├── BatchNorm2d
│ │   └── Dropout
├── Output Block
│ ├── Global Average Pooling (6 channels)
│ └── 1x1 Conv2D (10 outputs)
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 24, 24, 24]           3,456
              ReLU-6           [-1, 24, 24, 24]               0
       BatchNorm2d-7           [-1, 24, 24, 24]              48
           Dropout-8           [-1, 24, 24, 24]               0
            Conv2d-9            [-1, 8, 24, 24]             192
        MaxPool2d-10            [-1, 8, 12, 12]               0
           Conv2d-11           [-1, 16, 10, 10]           1,152
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 32, 8, 8]           4,608
             ReLU-16             [-1, 32, 8, 8]               0
      BatchNorm2d-17             [-1, 32, 8, 8]              64
          Dropout-18             [-1, 32, 8, 8]               0
           Conv2d-19             [-1, 48, 6, 6]          13,824
             ReLU-20             [-1, 48, 6, 6]               0
      BatchNorm2d-21             [-1, 48, 6, 6]              96
          Dropout-22             [-1, 48, 6, 6]               0
        AvgPool2d-23             [-1, 48, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             480
================================================================
Total params: 24,128
Trainable params: 24,128
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.96
Params size (MB): 0.09
Estimated Total Size (MB): 1.06
----------------------------------------------------------------
```

- Input: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Total parameters:  24,128
- Hyperparameters:
    - Epochs: 1
    - Learning Rate: 0.15
    - Batch Size: 64
    - Dropout: 0.2
    - Optimizer: SGD
    - Loss Function: NLLLoss
    - Device: CPU
    - Dataset: MNIST
    - Learning Rate: 0.15
    - SGD Momentum: 0.9
    - SEED: 1
    - Image Size: 28x28x1
    - is_aug: False (by default) Enables or disables augmentations
    - Image Augmentation: Rotation, Scaling, Color Jitter
- Training Logs:
```
EPOCH: 1
Loss=0.010176203213632107 Batch_id=937 Accuracy=95.78: 100%|██████████| 938/938 [02:11<00:00,  7.15it/s]

Test set: Average loss: 0.0416, Accuracy: 9876/10000 (98.76%)
```

### Image Augmentation
![Image Augmentation](https://raw.githubusercontent.com/garima-mahato/ERAV3_Session5_NNAndMLOps/refs/heads/main/assets/img_aug.png) 

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- pytest

## Local Setup

###1. Clone the repository
```bash
git clone <repository-url>
cd <folder-path>
```

### 2. Create and activate virtual environment
```
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
``` 

###4. Run tests
```
cd src
python -m pytest test_model.py -v
```

### 5. Train the model
To train the model locally:
```
cd src
python train.py
```


The trained model will be saved with a timestamp suffix (e.g., `model_20240101_120000.pth`).

## CI/CD Pipeline

The project includes a GitHub Actions workflow that automatically:
1. Sets up a Python environment
2. Installs dependencies
3. Runs all tests
4. Validates the model

### Tests Include:
- Model parameter count verification (< 25,000 parameters)
- Input shape validation (28x28)
- Output shape validation (10 classes)
- Training accuracy check (> 95%)

## File Descriptions

- `src/model.py`: Contains the CNN architecture
- `src/train.py`: Training script with data loading and model saving
- `src/test_model.py`: Test suite for model validation
- `.github/workflows/ml-pipeline.yml`: GitHub Actions workflow configuration
- `requirements.txt`: Project dependencies
- `.gitignore`: Specifies which files Git should ignore

## Notes

- Training is performed on CPU
- MNIST dataset will be automatically downloaded during first run
- Model files and downloaded data are excluded from Git tracking
- Training runs for 1 epoch to demonstrate the pipeline

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
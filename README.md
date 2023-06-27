# Summer Of Science-2023 - Fundamentals of Machince Learning

This repository contains Python code for implementing various deep learning algorithms as part of my report on the fundamentals of deep learning. The code covers topics such as linear regression, logistic regression, feed-forward neural networks, and convolutional neural networks (CNNs). The implementations include examples using the MNIST dataset for digit classification and a CNN for the Cats vs Dogs image classification problem.

## Contents

The repository includes the following files and directories:
- `Machine Learning`: Folder contains the following
   - `linear_regression`: Python notebook implementing linear regression.
   - `binary_classification`: Python notebook implementing logistic regression.
   - `multi_class_classification`: Python notebook implementing logistic regression.

- `Deep Learning`: Folder contains the following
   - `mnist_digits_FFN`: Python notebook implementing a feed-forward neural network for MNIST dataset.
   - `mnist_digits_CNN`: Python notebook implementing a CNN for the MNIST dataset.
   - `cats_vs_dogs`: Python notebook implementing a CNN for Cats vs Dogs classification.


## Usage

1. Clone this repository to your local machine using the following command:
   ```
   git clone https://github.com/ChaitanyaKatti/Summer-Of-Science-2023.git
   ```
2. Download and extract the following .rar file into `/data` folder from link below(~770 MB's). It contains the image datasets used in cats_vs_dogs.
   Change the `data_path` in the `cats_vs_dogs/trainer.ipynb` notebook file to this extracted folder.
   ```
   https://drive.google.com/file/d/1FmQrel0H9tSWnBFEHbm1nUCEBf2z0SpG/view?usp=sharing
   ```
4. Open your favorite code editor and run the notebook files.

## Requirements

To run the code in this repository, you need the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Pytorch

Create a virtual environment and install the modules listed in requirments.txt
```bash
pip install -r requirments.txt
```

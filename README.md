# TabTransformer
This repository hosts a comprehensive PyTorch implementation of the TabTransformer model, tailored for handling tabular data efficiently. The model architecture comprises various components designed to process both categorical and numerical features seamlessly.
The Encoder class is responsible for implementing the Transformer encoder stack, facilitating the encoding of input features. By specifying parameters such as the number of input features, encoder heads, feedforward dimension, encoder layers, and dropout rate, this class efficiently encodes input data using a Transformer encoder.
The Head class defines the classification head for the model, enabling classification based on input features. With parameters like input size, hidden layer size, and the number of output classes, this class plays a crucial role in the classification process.
The TabTransformer class serves as the core component of the model, integrating Encoder, Head, and Embedding layers to create a cohesive TabTransformer architecture. With configurable parameters for model architecture setup, this class effectively processes both categorical and numerical features for various classification tasks.
# Dataset
The TabTransformerDataset class acts as a custom PyTorch Dataset handler specifically designed for tabular data. It manages data loading and retrieval tasks efficiently, accommodating categorical and numerical features along with target labels for seamless training of the TabTransformer model.
# Tokenizer
Lastly, the TabTransformerTokenizer class focuses on tokenizing categorical data essential for the TabTransformer model. With methods like extracting unique tokens from datasets, creating tokenizers, fitting and transforming datasets, this class streamlines the tokenization process for categorical columns in tabular datasets.
# Usage
```python
import torch
import pandas as pd

from src.tokenizer import TabTransformerTokenizer
from src.dataset import TabTransformerDataset

# This sample csv file is stored in the repo, it is just a sample split from titanic dataset.
dataset = pd.read_csv("./test_df_titanic.csv")

# Create categorical only dataframe.
categorical_columns = ["Pclass", "Sex", "Embarked", "SibSp"]
categorical_dataset = dataset[categorical_columns]
# Tokenize categorical dataset.
tokenizer = TabTransformerTokenizer()
categorical_dataset = tokenizer.fit_transform(categorical_columns, dataset)

# Create numerical only dataframe.
numerical_columns = ["Age", "Fare", "Parch"]
numerical_dataset = dataset[numerical_columns]

# Convert labels to pandas Series.
labels = dataset["Survived"].squeeze()

# Create a TabTransformerDataset object. (Does not need custom collate-fn)
dataset = TabTransformerDataset(labels, categorical_dataset, numerical_dataset)
```

# Contributing
We welcome contributions to enhance the TabTransformer repository. If you wish to contribute, please follow these guidelines:
Fork the repository and create a new branch for your feature or bug fix.
Ensure your code follows the existing style and conventions.
Write clear and concise commit messages.
Test your changes thoroughly before submitting a pull request.
Provide detailed descriptions of your changes in the pull request.  

# License
This project is licensed under the MIT License. By contributing to this project, you agree to abide by the terms of this license.  

# Contact
For any inquiries, feedback, or collaboration opportunities related to the TabTransformer model, feel free to reach out to us at ardalan.dsht@gmail.com .  

# Documentation
Comprehensive documentation for the TabTransformer repository can be found in the docstrings. It includes detailed information on how to use the model, dataset handling, tokenization process, and more. If you have any questions or need further clarification, refer to the documentation or reach out to us via email.
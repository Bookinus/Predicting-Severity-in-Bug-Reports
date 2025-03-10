# Thesis: Automated Severity Classification in Bug Reports - Evaluating a Hybrid BERT and K-Means Approach

## Overview
This study investigates the use of a machine learning model that combines BERT (Bidirectional Encoder Representations from Transformers) and K-means clustering to predict the severity of bug reports. The primary objective is to explore how well this model can classify bug report severity levels based on various features of the reports.

## Research Approach
This study employs an experimental design to manipulate independent variables such as batch size, learning rate, and the number of epochs for BERT, as well as the number of clusters, initialization method, and distance metric for K-means. The dependent variables include evaluation metrics like Accuracy (ACC), Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI), which will be used to measure the model's performance.

## Dataset
The bug reports for this study are sourced from a dataset from Kaggle ([Software Bug Reports - Kaggle Dataset](https://www.kaggle.com/datasets/samanthakumara/software-bug-reports)) through secondary data collection. This dataset consists of Bugzilla bug reports and will be used to fine-tune a pre-trained BERT model.

## Model Implementation
TILLFÄLLIG:
The machine learning model will be implemented using Python and the PyTorch deep learning framework......

## Evaluation Metrics
The performance of the model will be assessed using the following quantitative metrics:

- Accuracy (ACC)
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)

## Libraries & Frameworks Used
TILLFÄLLIG:
- Python: The programming language used for the implementation.
- PyTorch: Deep learning framework used for fine-tuning the BERT model.
- Hugging Face Transformers: Pre-trained BERT model and tools for NLP tasks.
- Scikit-learn: Used for data normalization, K-means clustering, and evaluating model performance.

## Folder Structure
TILLFÄLLIG: 
/DataSet: Contains the dataset of bug reports used for model training and testing.
/notebooks: Jupyter Notebooks for exploratory analysis and visualization.
/src: Python scripts for data preprocessing, model training, and evaluation.
/models: Pre-trained and fine-tuned models for bug report severity prediction.
/results: Logs, evaluation metrics, and output from model experiments.

## How to Run the Code
Install Dependencies: You can install the required libraries using pip:


## License
TILLFÄLLLIG
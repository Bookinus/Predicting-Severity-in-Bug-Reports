# Thesis: Automated Bug Severity Classification from Bug Reports - Evaluating a Hybrid BERT and K-means Approach

## Overview
This implementation is a result from the bachelor thesis "Automated Bug Severity
Classification from Bug Reports - Evaluating a Hybrid BERT and K-means Approach"

As software becomes more integral to modern life, the challenge of managing
bugs in large-scale projects grows. Bug tracking systems help address this issue,
and automating parts of the maintenance process using machine learning has
emerged as a promising solution to reduce manual effort and enhance efficiency.

Advancements in machine learning, driven by affordable hardware and vast data
availability, have enabled the automation of software maintenance tasks such as
bug categorization, prioritization, triage, and severity prediction. Additionally,
pre-trained NLP models have contributed to research in this field. Further
research is needed to evaluate the effectiveness of state-of-the-art pre-trained
natural language processing models in combination with unsupervised cluster-
ing algorithms for automating the classification of Bugzilla bug report severity
levels.

Therefore, the research question is:
- How well can a machine learning model combining BERT and K-means predict
severity in bug reports?

## Research Approach
This study employs an experimental design to manipulate independent variables such as batch size
and the number of epochs for BERT. The dependent variables include evaluation metrics 
like Accuracy (ACC), Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI), 
which will be used to measure the performance of the model. 

## Dataset
The bug reports for this study are sourced from a dataset from Kaggle ([Software Bug Reports - Kaggle Dataset](https://www.kaggle.com/datasets/samanthakumara/software-bug-reports)) through secondary data collection. This dataset consists of Bugzilla bug reports and will be used to fine-tune a pre-trained BERT model.

## Model Implementation

This study fine-tunes BERT to generate text embeddings, 
which are then clustered using K-means with K-means++ initialization 
to group bug reports based on severity. The fine-tuned BERT model produces embeddings, 
which are used for clustering to predict severity levels. The performance of the clustering 
is evaluated using metrics such as NMI and ARI.


## Evaluation Metrics
The accuracy of the model is evaluated after fine-tuning, and the quality of the
clusters is assessed by comparing them to the actual labels using the Adjusted
Rand Index (ARI) and Normalized Mutual Information (NMI).

## Results
The model that achieved the best results was trained for 3 epochs with a batch
size of 8. After fine-tuning, the accuracy reached 0.6535, while the clustering
results showed an NMI of 0.0477 and an ARI of 0.0214. All other models yielded
similar scores, demonstrating that the results were consistently poor across the
experiments.

## Libraries & Frameworks Used
- Python 3.12.3: The programming language used for the implementation.
- PyTorch 2.6.0: Deep learning framework used for fine-tuning the BERT model.
- CUDA 12.6 (if you're using a GPU)
- Jupyter Notebook version 7.3.3: Used to implement, visualize, and document your model
- scikit-learn 1.6.1: Used for data normalization, K-means clustering, and evaluating model performance.
- Hugging Face Transformers 4.50.2: Pre-trained BERT model and tools for NLP tasks.
- seaborn 0.13.2: Used for visualization.
- numpy 2.2.4: Used for data structures
- pandas 2.2.3: Used for the data structure DataFrame

## Folder Structure

Project Folder/
├── DataSet/
│   ├── OriginalDataSet/
│   │   └── Eclipse.csv
│   │   └── Mozilla.csv
│   │   └── Netbeans.csv
│   │   └── Openoffice.csv
│   └── ThesisDataSet/
│   │   └── Eclipse.csv
│   │   └── Mozilla.csv
│   │   └── Openoffice.csv
│   └── CleanDataSet.csv
│   └── README.md
├── src/
│   ├── runs/profiler
│   └── CleanDataSet.ipynb
│   └── Model.ipynb
└── README.md

## How to Run the Code
You can install the required libraries using pip:
- Python 3.12.3
- PyTorch 2.6.0
- CUDA 12.6 (if you're using a GPU)
- Jupyter Notebook version 7.3.3
- numpy 2.2.4
- pandas 2.2.3
- scikit-learn 1.6.1
- transformers 4.50.2
- seaborn 0.13.2

## Authors

- Melinda Avery
- Sanna Nummelin

Bachelor’s Thesis, 
Stockholm University, 
Department of Computer and Systems Sciences, 
Degree project 15 credits, 
Spring term 2025, 
Supervisor: Mateus de Oliveira Oliveira, 
Swedish title: Automatiserad klassificering av allvarlighetsgrad för buggar i buggrapporter: Utvärdering av en hybridmetod med BERT och K-Means

## License
VAD SKA VI SKRIVA HÄR?
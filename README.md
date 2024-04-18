# ML-Interview-Challenge
A machine learning approach to predict the subjects of scientific papers
# Scientific Paper Subject Prediction
# Overview
This project aims to develop a machine learning approach to predict the subjects of scientific papers using the Cora dataset. The dataset consists of 2708 scientific publications classified into one of seven classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, and Theory. Each paper is described by a 0/1-valued word vector indicating the absence/presence of corresponding words from the dictionary.

# Dataset
The Cora dataset includes the following files:

cora.content: Descriptions of papers in the format <paper_id> <word_attributes>+ <class_label>.
cora.cites: Citation graph of the corpus in the format <ID of cited paper> <ID of citing paper>.
The task involves the following steps:

Data Loading and Preprocessing: Load the dataset and preprocess it for model training.
Model Development: Develop a machine learning approach to predict the subjects of papers.
Evaluation: Evaluate the model's performance using 10-fold cross-validation and calculate the accuracy.
Storing Predictions: Store the predictions in a tab-separated values (TSV) file.
Documentation: Document the approach, implementation details, and results in this README file.
# Execution
To execute the approach:

Clone the repository.
Run the Jupyter notebook scientific_paper_subject_prediction.ipynb to train the model, make predictions, and evaluate performance.
View the generated prediction file predictions.tsv to examine the predicted subjects for each paper.
# Files Included
scientific_paper_subject_prediction.ipynb: Jupyter notebook containing the code for data loading, preprocessing, model development, evaluation, and prediction.
predictions.tsv: Tab-separated values file containing the predicted subjects for each paper.
cora.content and cora.cites: Original dataset files.
# Dependencies
numpy
pandas
scikit-learn
Install dependencies using the following command:

''' pip install numpy pandas scikit-learn '''
# Author
Jayalaxmi Botsa

# ML-Interview-Challenge
A machine learning approach to predict the subjects of scientific papers
# Scientific Paper Subject Prediction
# Overview
This project aims to develop a machine learning approach to predict the subjects of scientific papers using the Cora dataset. The dataset consists of 2708 scientific publications classified into one of seven classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, and Theory. Each paper is described by a 0/1-valued word vector indicating the absence/presence of corresponding words from the dictionary.

# Dataset
The Cora dataset includes the following files:

**cora.content:** Descriptions of papers in the format <paper_id> <word_attributes>+ <class_label>.
**cora.cites:** Citation graph of the corpus in the format <ID of cited paper> <ID of citing paper>.
The task involves the following steps:

# Code Description
The provided Python script Scientific_Papers_Classification.ipynb contains the following functions:

1. **load_data()**: Loads the Cora dataset from the provided files and returns features representing word attributes (X) and class labels (y).
2. **load_citation_graph()**: Loads the citation graph from the .cites file and returns a list of citation information.
3. **split_dataset(X, y)**: Splits the dataset into train and test sets using Stratified K-Fold cross-validation.
4. **train_and_predict(X_train, y_train, X_test)**: Trains a Multinomial Naive Bayes classifier using the training set (X_train, y_train) and makes predictions on the test set (X_test).
5. **save_predictions(predictions, test_indices, y_true)**: Saves the predictions to a tab-separated values (TSV) file.
6. **evaluate_accuracy(y_true, predictions)**: Calculates the accuracy of the predictions.
7. **Main Execution**: Loads the data, performs cross-validation, trains the model, evaluates its performance, saves predictions, and calculates the mean accuracy across all folds.
   
# Files Included
scientific_paper_subject_prediction.ipynb: Jupyter notebook containing the code for data loading, preprocessing, model development, evaluation, and prediction.
predictions.tsv: Tab-separated values file containing the predicted subjects for each paper.
cora.content and cora.cites: Original dataset files.

# Dependencies
1. numpy
2. pandas
3. scikit-learn
Install dependencies using the following command:

'''**pip install numpy pandas scikit-learn** '''

# Author
Jayalaxmi Botsa


# Task 4: Classification with Logistic Regression

This project focuses on building a binary classification model using Logistic Regression.

In this task, a built-in dataset called the Breast Cancer Wisconsin dataset is used from the Scikit-learn library. This dataset contains features related to cell nuclei and is used to classify whether a tumor is malignant or benign.

The dataset is first loaded and explored to understand its structure, shape, and class distribution. The features and target variable are separated for model training.

The data is then split into training and testing sets using the train-test split method. Feature scaling is applied using StandardScaler to normalize the data and improve model performance.

A Logistic Regression model is trained on the scaled training data. After training, predictions are made on the test dataset.

The model is evaluated using different metrics such as accuracy, precision, recall, and ROC-AUC score. A confusion matrix is used to understand the classification performance in detail.

A ROC curve is plotted to visualize the performance of the classifier. The sigmoid function is also plotted to understand how logistic regression maps input values to probabilities.

Threshold tuning is performed to observe how changing the decision threshold affects precision and recall.

From the results, it is observed that logistic regression performs well for binary classification problems and provides good evaluation metrics.

This task helps in understanding binary classification, model evaluation, ROC curve analysis, and the concept of the sigmoid function in machine learning.

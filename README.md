Loan Prediction with Machine Learning
This project implements a machine learning model to predict loan approval using a Decision Tree Classifier. The dataset contains information about loan applicants, and the goal is to classify whether a loan will be approved or not based on multiple features.

Project Structure
Data: The dataset used is loan_data_train.csv, which contains applicant details such as gender, marital status, loan amount, credit history, and more.
Model: A Decision Tree Classifier is used for loan status prediction.
Evaluation: The model is evaluated using accuracy score and classification report.
Dataset
The dataset includes the following columns:

Loan_ID: Unique Loan identifier (not used in modeling)
Gender, Married, Dependents, Self_Employed, LoanAmount, Loan_Amount_Term, Credit_History: Applicant information
Loan_Status: Target variable (Y) indicating loan approval (1) or rejection (0)

Steps:
Data Loading: Load the dataset using pandas.
train_df = pd.read_csv("loan_data_train.csv")

Data Cleaning: Handle missing values by dropping rows with missing values in critical columns.
train_df_clean = train_df.dropna(subset=['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'])

Feature Engineering: Prepare the feature matrix (X) by dropping non-predictive columns and convert categorical features into dummy variables.
X = pd.get_dummies(train_df_clean.drop(['Loan_Status', 'Loan_ID'], axis=1))
Y = train_df_clean['Loan_Status']

Data Splitting: Split the data into training and testing sets.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

Model Training: Initialize and train a Decision Tree Classifier.
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, Y_train)

Prediction and Evaluation: Predict loan approval on the test set and evaluate model accuracy.
from sklearn.metrics import accuracy_score, classification_report
prediction = model.predict(X_test)
accuracy = accuracy_score(Y_test, prediction)

print(f"Model Accuracy: {accuracy:.2f}")
The model achieved an accuracy score of 0.68 (68%).

Requirements
Python 3.x
pandas
numpy
scikit-learn
matplotlib
seaborn

How to Run
Clone the repository.
Install the required libraries

Let me know if you'd like to add or modify anything!

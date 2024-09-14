Predicting Insurance Premium using Linear Regression

Objective:
To predict insurance premium based on user data such as age, income, claims history, and insurance type using linear regression.

Implementation:

Libraries Used:

pandas, numpy: For handling the data.
scikit-learn: For model building, training, and evaluation.
Steps:

Data Preparation:
A sample dataset is used, consisting of features like Age, Annual Income, Claims History, Insurance Type, and Premium.

Train-Test Split:
The dataset is split into training and testing sets.

Model Training:
A Linear Regression model is trained on the training set.

Model Evaluation:
The model’s performance is evaluated using mean absolute error (MAE) and mean squared error (MSE).

Premium Prediction for New User:
A forecast is made for a new user’s premium based on their input.

Design Choices:

Linear regression was chosen as a baseline model since the relationship between the features and the target (premium) is likely linear.
Challenges Faced:

The small dataset size made it difficult to assess model performance on real-world data. A larger dataset would provide more reliable predictions.

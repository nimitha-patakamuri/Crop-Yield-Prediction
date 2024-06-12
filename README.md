# Crop Yield Prediction using Machine Learning Algorithms

This repository contains two assignments focused on predicting crop yield using various machine learning algorithms, including Support Vector Regression (SVR), Radial Basis Function Neural Network (RBFNN), Back Propagation Neural Network (BPNN), and Random Forest.

### Crop Yield Prediction using Random Forest

### Objective
The aim of this assignment is to predict crop yield using the Random Forest machine learning algorithm.

### Random Forest Algorithm
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both classification and regression problems in ML. Random Forest works in two phases: first, it creates the random forest by combining N decision trees, and second, it makes predictions for each tree created in the first phase.

### Dataset
The dataset used for this assignment is provided in an Excel file (`21BCE9274.xlsx`). The dataset contains features related to crop yield, and the target variable is the actual crop yield (`Yeild (Q/acre)`).

### Code
The code for this assignment is written in Python and utilizes the following libraries:
- `pandas` for data manipulation
- `sklearn` for machine learning models and evaluation metrics
- `matplotlib` for data visualization

The code performs the following steps:
1. Load the dataset from the Excel file.
2. Preprocess the data by normalizing the features and splitting the dataset into training and testing sets.
3. Train the Random Forest Regression model on the training data.
4. Evaluate the model's performance on the testing data using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared score.
5. Visualize the actual vs. predicted crop yield using scatter plots.

### Results
The Random Forest algorithm achieved the following results on the given dataset:
- R-squared score: 92.28%
- Mean Squared Error (MSE): 0.075%
- Root Mean Squared Error (RMSE): 22.6%

### Crop Yield Prediction using SVR, RBFNN, and BPNN

### Objective
The aim of this assignment is to design and implement machine learning algorithms such as Support Vector Regression (SVR), Radial Basis Function Neural Network (RBFNN), and Back Propagation Neural Network (BPNN) for predicting crop yield. The performance of these algorithms is evaluated using metrics like R-squared (R²), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

### Algorithms
1. **Support Vector Regression (SVR)**: SVR is a popular machine learning algorithm used for regression analysis. It is based on the Support Vector Machine (SVM) algorithm, which is primarily used for classification problems. SVR aims to find a hyperplane that best fits the given data points while maximizing the margin of error.

2. **Radial Basis Function Neural Network (RBFNN)**: The Radial Basis Function (RBF) neural network algorithm is a type of feedforward neural network composed of three layers: input, hidden, and output. It is commonly used for function approximation and classification tasks.

3. **Back Propagation Neural Network (BPNN)**: Backpropagation is a popular algorithm used for training artificial neural networks in machine learning. It is a supervised learning algorithm that uses a gradient descent approach to optimize the parameters of a neural network model.

### Code
The code for this assignment is written in Python and utilizes the following libraries:
- `pandas` for data manipulation
- `sklearn` for machine learning models and evaluation metrics
- `matplotlib` for data visualization
- `openpyxl` for writing predictions to an Excel file

The code performs the following steps:
1. Load the dataset from an Excel file.
2. Split the data into input features (X) and target variable (y).
3. Normalize the input features using StandardScaler.
4. Split the data into training and testing sets.
5. Train the SVR, RBFNN, and BPNN models on the training data.
6. Evaluate the models' performance on the testing data using metrics like R-squared (R²), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
7. Visualize the actual vs. predicted crop yield for each model using scatter plots.
8. Save the predictions to a new sheet in the Excel file.

### Results
The code reports the following results for the three algorithms:

**SVR**:
- R-squared: [value]
- Mean Squared Error (MSE): 2.0065878741521795e-06
- Root Mean Squared Error (RMSE): [value]

**RBFNN**:
- R-squared: [value]
- Mean Squared Error (MSE): 4.3456013950577274e-06
- Root Mean Squared Error (RMSE): [value]

**BPNN**:
- R-squared: [value]
- Mean Squared Error (MSE): 0.0550280310073301
- Root Mean Squared Error (RMSE): [value]

Based on the reported MSE values, the SVR and RBFNN algorithms perform better than the BPNN algorithm for this particular dataset. However, it is important to note that the performance of these algorithms can vary depending on the dataset and the hyperparameters used.

## Usage
To run the code for either assignment, follow these steps:

1. Clone the repository or download the source code files.
2. Install the required Python libraries (e.g., pandas, scikit-learn, matplotlib, openpyxl).
3. Open the respective Python file (`assignment1.py` or `assignment2.py`) in a Python IDE or text editor.
4. Ensure that the dataset file (`21BCE9274.xlsx`) is in the correct location specified in the code.
5. Run the Python script.

The code will load the dataset, preprocess the data, train the specified machine learning models, evaluate their performance, and generate visualizations and predictions as per the assignment requirements.

## Contributing
Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).

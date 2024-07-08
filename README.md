# Crop Yield Prediction using Machine Learning Algorithms

## Table of Contents

- [Crop Yield Prediction using Random Forest](#assignment-1-crop-yield-prediction-using-random-forest)
  - [Objective](#objective)
  - [Random Forest Algorithm](#random-forest-algorithm)
  - [Dataset](#dataset)
  - [Code](#code)
  - [Results](#results)
- [Crop Yield Prediction using SVR, RBFNN, and BPNN](#assignment-2-crop-yield-prediction-using-svr-rbfnn-and-bpnn)
  - [Objective](#objective-1)
  - [Algorithms](#algorithms)
    - [Support Vector Regression (SVR)](#support-vector-regression-svr)
    - [Radial Basis Function Neural Network (RBFNN)](#radial-basis-function-neural-network-rbfnn)
    - [Back Propagation Neural Network (BPNN)](#back-propagation-neural-network-bpnn)
  - [Code](#code-1)
  - [Results](#results-1)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Crop Yield Prediction using Random Forest

### Objective

The aim of this assignment is to predict crop yield using the Random Forest machine learning algorithm.

### Random Forest Algorithm

Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both classification and regression problems in ML. Random Forest works in two phases: first, it creates the random forest by combining N decision trees, and second, it makes predictions for each tree created in the first phase.

### Dataset

The dataset used for this assignment is provided in an Excel file. The dataset contains features related to crop yield, and the target variable is the actual crop yield (`Yeild (Q/acre)`).

### Code

The code for this assignment is written in Python and utilizes the following libraries:

- `pandas` for data manipulation
- `sklearn` for machine learning models and evaluation metrics
- `matplotlib` for data visualization

### Results

The Random Forest algorithm achieved the following results on the given dataset:

- R-squared score: 92.28%
- Mean Squared Error (MSE): 0.075%
- Root Mean Squared Error (RMSE): 22.6%

## Crop Yield Prediction using SVR, RBFNN, and BPNN

### Objective

The aim of this assignment is to design and implement machine learning algorithms such as Support Vector Regression (SVR), Radial Basis Function Neural Network (RBFNN), and Back Propagation Neural Network (BPNN) for predicting crop yield. The performance of these algorithms is evaluated using metrics like R-squared (R²), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

### Algorithms

#### Support Vector Regression (SVR)

SVR is a popular machine learning algorithm used for regression analysis. It is based on the Support Vector Machine (SVM) algorithm, which is primarily used for classification problems. SVR aims to find a hyperplane that best fits the given data points while maximizing the margin of error.

#### Radial Basis Function Neural Network (RBFNN)

The Radial Basis Function (RBF) neural network algorithm is a type of feedforward neural network composed of three layers: input, hidden, and output. It is commonly used for function approximation and classification tasks.

#### Back Propagation Neural Network (BPNN)

Backpropagation is a popular algorithm used for training artificial neural networks in machine learning. It is a supervised learning algorithm that uses a gradient descent approach to optimize the parameters of a neural network model.

### Code

The code for this assignment is written in Python and utilizes the following libraries:

- `pandas` for data manipulation
- `sklearn` for machine learning models and evaluation metrics
- `matplotlib` for data visualization
- `openpyxl` for writing predictions to an Excel file

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

### Usage

To run the code for either assignment, follow these steps:

1. Clone the repository or download the source code files.
2. Install the required Python libraries (e.g., pandas, scikit-learn, matplotlib, openpyxl).
3. Open the respective Python file  in a Python IDE or text editor.
4. Ensure that the dataset file is in the correct location specified in the code.
5. Run the Python script.

The code will load the dataset, preprocess the data, train the specified machine learning models, evaluate their performance, and generate visualizations and predictions as per the assignment requirements.

### Contributing

Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

### License

This project is licensed under the [MIT License](LICENSE).

# Diamond Price Prediction using Machine Learning

This project aims to predict the price of diamonds using machine learning models based on various features such as carat, cut, color, clarity, and dimensions (depth, table, x, y, z). The dataset used for this project is from Kaggle's "Diamonds" dataset, and the goal is to build models that can accurately predict diamond prices.

The project employs various machine learning models, including linear regression, decision trees, random forests, k-nearest neighbors, and XGBoost, among others. Through the process of training and hyperparameter tuning, the best model is selected based on performance metrics such as Root Mean Squared Error (RMSE) and R² score.

## Table of Contents

* [Project Overview](#project-overview)
* [Installation](#installation)
* [Features](#features)
* [Model Evaluation](#model-evaluation)
* [How to Run](#how-to-run)
* [Project Structure](#project-structure)
* [Results](#results)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Project Overview

This project consists of the following key tasks:

1. **Data Preprocessing**: The data is cleaned by handling missing values, encoding categorical variables, and scaling numerical features.
2. **Model Training**: A variety of regression models are trained on the dataset.
3. **Model Evaluation**: Multiple metrics (RMSE, R²) are used to evaluate model performance.
4. **Hyperparameter Tuning**: Model hyperparameters are tuned using GridSearchCV to improve performance.
5. **Visualization**: Graphical representation of actual vs predicted diamond prices.

## Installation

To get started with this project, you'll need to install the required dependencies.

### 1. Clone the repository

Clone this repository to your local machine:

```bash
git clone https://github.com/yashnegi11/diamond-price-prediction.git
```

### 2. Set up a Virtual Environment

It's highly recommended to use a virtual environment for Python projects. Here's how to create one:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Run the following command to install all the necessary libraries:

```bash
pip install -r requirements.txt
```

### 4. Optional: Set up Jupyter Notebooks

If you'd like to run the notebooks, install Jupyter:

```bash
pip install jupyter
```

## Features

* **Data Preprocessing**: This involves cleaning the dataset by handling missing values, encoding categorical features (e.g., cut, color, clarity), and scaling numerical features using `StandardScaler`.
* **Model Training**: Trains various machine learning models like:

  * Linear Regression
  * Lasso Regression
  * Ridge Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * K-Nearest Neighbors Regressor
  * XGBoost Regressor
* **Model Evaluation**: The models are evaluated using k-fold cross-validation to calculate Root Mean Squared Error (RMSE) and R² scores.
* **Hyperparameter Tuning**: The project uses `GridSearchCV`  to tune the hyperparameters of models like Random Forest and XGBoost.
* **Visualization**: Plots to compare actual vs predicted prices, and also a grid search visualization to see how the model performance improves with tuning.

## Model Evaluation

The models are evaluated on the basis of:

* **Root Mean Squared Error (RMSE)**: A measure of how well the model predicts the target variable. Lower values are better.
* **R² Score**: Indicates the proportion of the variance in the target variable that is explained by the model. Values closer to 1 indicate better performance.

After testing all models, **XGBoost** was found to have the best performance with the lowest RMSE and highest R² score.

## Results

* **XGBoost Regressor** outperforms all other models with the lowest RMSE and highest R² score.
* Models such as **Random Forest** and **Decision Tree** also perform well but don't reach the level of accuracy achieved by XGBoost.

Example performance metrics:

* **XGBoost RMSE**: 536.91
* **Random Forest RMSE**: 546.61
* **Decision Tree RMSE**: 746.53
* **Linear Regression RMSE**: 1338.35

These results are evaluated on the test set after training the models with various hyperparameters.

## How to Run

1. **Training the Models:**
   You can run the pipeline to train all models and evaluate them:

   ```bash
   python src/pipeline.py
   ```

2. **Running Jupyter Notebooks:**
   For an interactive exploration of the project, use Jupyter notebooks:

   ```bash
   jupyter notebook notebooks/Model_Training.ipynb
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

* Thanks to [Kaggle](https://www.kaggle.com/datasets) for the Diamond dataset.
* Special thanks to the authors and contributors of the libraries used, including Scikit-learn, XGBoost, and Pandas.
* Inspiration from various machine learning resources, blogs, and tutorials.

---

This README provides a comprehensive overview of your project, how to set it up, and how to evaluate the models. You can easily tweak it to suit your preferences or any additional details you'd like to include.

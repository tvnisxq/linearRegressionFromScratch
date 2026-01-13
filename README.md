# Linear Regression From Scratch

A Python implementation of linear regression built from first principles, demonstrating how linear regression algorithms work without relying on scikit-learn or other ML libraries.

## Overview

This project implements linear regression using **gradient descent** to find the optimal line of best fit for a given dataset. The implementation includes:

- Manual calculation of loss function (Mean Squared Error)
- Gradient descent optimization algorithm
- Data visualization with scatter plots and regression lines
- Study hours vs. test scores prediction

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- numpy

## Dataset

The dataset is a synthetically created dataset(`data/data.csv`) contains:
- `study_hours`: Number of hours spent studying
- `scores`: Corresponding test scores

## How It Works

### 1. Loss Function
The implementation uses Mean Squared Error (MSE) to measure prediction error:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $\hat{y} = mx + b$

### 2. Gradient Descent
The algorithm iteratively updates the slope ($m$) and intercept ($b$) to minimize the loss:

$$m_{new} = m_{old} - L \cdot \frac{\partial \text{MSE}}{\partial m}$$
$$b_{new} = b_{old} - L \cdot \frac{\partial \text{MSE}}{\partial b}$$

Where:
- $L$ = Learning rate (controls step size)
- $\frac{\partial \text{MSE}}{\partial m}$ and $\frac{\partial \text{MSE}}{\partial b}$ = Gradients

### 3. Configuration
- **Learning Rate**: 0.0001
- **Epochs**: 1000 iterations

## Usage

1. Ensure you're in the project directory
2. Activate the virtual environment (if needed)
3. Run the script:
   ```bash
   python main.py
   ```

The script will:
- Load the dataset from `data/data.csv`
- Create a scatter plot of the raw data
- Train the linear regression model using gradient descent
- Output the final slope ($m$) and intercept ($b$)
- Generate a visualization showing the fitted regression line
- Save both plots to the `assets/` folder

## Output

The program prints:
- Epoch checkpoints every 50 iterations
- Final slope ($m$) and intercept ($b$) values
- Two PNG files in the `assets/` folder showing data visualization and the fitted line

## Notes

- This is an educational implementation demonstrating core ML concepts
- The gradient descent algorithm can be tuned by adjusting learning rate and epochs
- The current learning rate and epoch count are suitable for the provided dataset

# Credits 
@NeuralNine

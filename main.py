import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/data.csv')
plt.scatter(data.study_hours, data.scores)
plt.show()

# Function for loss function
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].study_hours
        Y = points.iloc[i].scores
        total_error += (Y - (m*x + b)) ** 2
    total_error / float(len(points))

# Function for gradient descent
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].study_hours
        y = points.iloc[i].scores

        m_gradient += -(2/n) * x * (y - (m_now *x - b_now))
        b_gradient += -(2/n) * (y - (m_now *x - b_now))
        
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


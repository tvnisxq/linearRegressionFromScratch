import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/data.csv')
plt.scatter(data.study_hours, data.scores)
plt.show()

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].study_hours
        Y = points.iloc[i].scores
        total_error += (Y - (m*x + b)) ** 2
    total_error / float(len(points))
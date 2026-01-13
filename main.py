import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set seaborn style for better-looking plots
sns.set_style("darkgrid")
sns.set_palette("husl")

# Create an assets directoy to store visualizations
os.makedirs('assets', exist_ok=True)

data = pd.read_csv('data/data.csv')

# First plot with seaborn styling
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='study_hours', y = 'scores', s=100, alpha=0.6)
plt.title('Study Hours vs Scores', fontsize=16, fontweight='bold')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Scores', fontsize=12)
plt.savefig('assets/scatterplot.png', dpi=300, bbox_inches='tight')
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

m = 0
b = 0
L = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, L)

print(m ,b)

# Second plot with seaborn styling and regression line
plt.figure(figsize=(10, 6))
plt.scatter(data=data, x='study_hours', y='scores', s=100, alpha=0.6, color='steelblue')
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)], color="red", linewidth=2.5, label="Regression Line")
plt.title('Linear Regeression: Study Hours Vs Scores', fontsize=16, fontweight='bold')
plt.xlabel("Study Hours", fontsize=12)
plt.ylabel("Scores", fontsize=12)
plt.legend(fontsize=10)
plt.savefig('assets/regressionline.png', dpi=300, bbox_inches='tight')
plt.show()
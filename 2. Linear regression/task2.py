import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the Excel file
path = r"C:\Users\kezie\OneDrive\Skrivebord\fys-2021\2. Linear regression\global.temp.xlsx"
df = pd.read_excel(path)

# Prepare the data
X = df['Years'].values
y = df['Temperature'].values

# Calculate the number of data points
N = len(X)

# Calculate the sums needed for the slope (m) and intercept (b)
sum_x = np.sum(X)
sum_y = np.sum(y)
sum_xy = np.sum(X * y)
sum_x_squared = np.sum(X ** 2)

# Calculating slope (m)
m = (N * sum_xy - sum_x * sum_y) / (N * sum_x_squared - sum_x ** 2)

# Calculating intercept (b)
b = (sum_y - m * sum_x) / N

# Calculate the predicted y values
y_pred = m * X + b

# Print the slope and intercept
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# Plot the original data points
plt.figure(figsize=(10, 5))
plt.plot(X, y, marker='o', markersize=2.5, linestyle='', label='Actual Data')

# Plot the regression line
plt.plot(X, y_pred, color='red', label='Regression Line')

# Add title and labels
plt.title('Temperature Over Years')
plt.xlabel('Year')
plt.ylabel('Temp')

# Add grid and legend
plt.grid(True)
plt.legend()

# Display the plot
plt.show()

# r^2 = 1 would mean the regression line fits perfectly, while r^2 = 0 the line does not
# predict future values, and r^2 = negative means the actual values perform worse than the regression line

# Calculating total Sum of Squares (SST) the actual sum
y_mean = np.mean(y)
SST = np.sum((y - y_mean) ** 2)
 
# Calculating the residual sum of squares (SSE) the regression sum
SSE = np.sum((y - y_pred) ** 2)

# Calculate R^2 value
R2 = 1 - (SSE / SST)

# Print the R^2 value
print(f"R^2 value: {R2:.4f}")

# R squared says how good or bad the model is based on square error

# Beta one indicated the slope of the regression line; the increasing of the regression in y for each x value

# Oppga 1, sett ned og se at æ skjønne d

# plot the residual : residual vise avvik fra regresjonslinje til faktisk data 
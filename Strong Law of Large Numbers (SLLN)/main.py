import matplotlib.pyplot as plt
import numpy as np

# Exercice 1
n = 10000
p1 = 1 / 2  # Parameter for the first Bernoulli distribution
p2 = 1 / 3  # Parameter for the second Bernoulli distribution

x1 = np.random.binomial(size=n, n=1, p=0.5)
x2 = np.random.binomial(1, p2, size=n)

series1 = [np.mean(x1[:k]) for k in range(1, n + 1)]
series2 = [np.mean(x2[:k]) for k in range(1, n + 1)]

plt.figure(figsize=(10, 5))
plt.plot(range(1, n + 1), series1, label=f'p={p1}')
plt.plot(range(1, n + 1), series2, label=f'p={p2}')
plt.xlabel('Number of elements (k)')
plt.ylabel('Series Value')
plt.title('Bernoulli Distribution Series')
plt.legend()
plt.grid(True)
plt.show()


# Exercice 2
M = np.random.rand(100, 100)

column_index = np.random.randint(0, 100)  # Choose a random column index
xi = M[:, column_index]
sigma_i = np.std(xi)  # Standard deviation of the selected column
xei = (xi - np.mean(xi)) / sigma_i

# Plot histogram of normalized column xi
plt.figure(figsize=(8, 6))
plt.hist(xei, bins=20, color='skyblue', edgecolor='black')
plt.title('Histogram of Normalized Column xi')
plt.xlabel('Normalized Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Step 3: Build the series of averages of all columns xa and plot its histogram
xa = np.mean(M, axis=0)

# Plot histogram of series of averages of all columns
plt.figure(figsize=(8, 6))
plt.hist(xa, bins=20, color='lightgreen', edgecolor='black')
plt.title('Histogram of Series of Averages of All Columns')
plt.xlabel('Average Values')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

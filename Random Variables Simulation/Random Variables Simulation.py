import random

import matplotlib.pyplot as plt
import numpy as np


# Bernoulli Distribution
def my_bernoulli(n, p):
    """Generates a Bernoulli random sample.

    Args:
        n: The number of samples to generate.
        p: The probability of success.

    Returns: A list of samples, where each sample is an integer representing the outcome of the Bernoulli trial (1
    for success, 0 for failure).
    """

    # Check if the probability is between 0 and 1.
    if p < 0 or p > 1:
        raise ValueError("The probability must be between 0 and 1.")

    # Generate the samples.
    samples = [1 if random.random() < p else 0 for _ in range(n)]
    return samples


# Binomial Distribution
def my_binomial(N, n, p):
    """Generates a binomial random sample.

       Args:
           N: The number of samples to generate.
           n: The number of trials in each sample.
           p: The probability of success in each trial.

       Returns:
           A list of samples, where each sample is the number of successes in n trials.
       """

    # Check if the probability is between 0 and 1.
    if not 0 <= p <= 1:
        raise ValueError("The probability must be between 0 and 1.")

    # Generate the samples.
    samples = [sum(my_bernoulli(n, p)) for _ in range(N)]
    return samples


def discrete_generator(N, p):
    """Generates a discrete random sample from a given probability distribution.

    Args:
        N: The number of samples to generate.
        p: A list of probabilities, where the sum of all probabilities is equal to 1.

    Returns: A list of samples, where each sample is an integer representing the index of the corresponding
    probability in the `p` list.
    """

    # Check if the sum of all probabilities is equal to 1.
    if abs(sum(p) - 1) > 1e-10:
        raise ValueError("The sum of all probabilities must be equal to 1.")

    # Create a cumulative probability distribution.
    cdf = []
    for i in range(len(p)):
        cdf.append(sum(p[:i + 1]))

    # Generate the samples.
    samples = []
    for i in range(N):
        r = random.random()
        index = 0
        while r > cdf[index]:
            index += 1
        samples.append(index)
    return samples


# Exponential Distribution
def my_exponential(N, lam):
    """Generates an exponential random sample.

    Args:
        N: The number of samples to generate.
        lam: The rate parameter of the exponential distribution.

    Returns:
        A list of samples, where each sample is the time until the next event occurs.
    """

    # Check if the rate parameter is positive.
    if lam <= 0:
        raise ValueError("The rate parameter must be positive.")

    # Generate the samples.
    samples = [-np.log(1 - random.random()) / lam for _ in range(N)]
    return samples


def gamma(N, n, lam):
    samples = [sum(my_exponential(n, lam)) for _ in range(N)]
    return samples


# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot Bernoulli Distribution
n_bernoulli = 1000
p_bernoulli = 0.3
bernoulli_samples = my_bernoulli(n_bernoulli, p_bernoulli)
axs[0, 0].hist(bernoulli_samples, bins=2, density=True, alpha=0.6, color='g', label='Custom Bernoulli')

# Compare with numpy's implementation
np_bernoulli_samples = np.random.choice([0, 1], size=n_bernoulli, p=[1 - p_bernoulli, p_bernoulli])
axs[0, 0].hist(np_bernoulli_samples, bins=2, density=True, alpha=0.6, color='orange', label='Numpy Bernoulli')

axs[0, 0].set_title('Bernoulli Distribution')

# Plot Binomial Distribution
N_binomial = 1000
n_binomial = 10
p_binomial = 0.3
binomial_samples = my_binomial(N_binomial, n_binomial, p_binomial)
axs[0, 1].hist(binomial_samples, bins=n_binomial + 1, density=True, alpha=0.6, color='r', label='Custom Binomial')

# Compare with numpy's implementation
np_binomial_samples = np.random.binomial(n_binomial, p_binomial, size=N_binomial)
axs[0, 1].hist(np_binomial_samples, bins=n_binomial + 1, density=True, alpha=0.6, color='orange',
               label='Numpy Binomial')

axs[0, 1].set_title('Binomial Distribution')

# Plot Discrete Distribution
N_discrete = 1000
p_discrete = [0.1, 0.4, 0.2, 0.3]
discrete_samples = discrete_generator(N_discrete, p_discrete)
axs[1, 0].hist(discrete_samples, bins=len(p_discrete), density=True, alpha=0.6, color='b', label='Custom Discrete')

# Compare with numpy's implementation
np_discrete_samples = np.random.choice(range(len(p_discrete)), N_discrete, p=p_discrete)
axs[1, 0].hist(np_discrete_samples, bins=len(p_discrete), density=True, alpha=0.6, color='orange',
               label='Numpy Discrete')

axs[1, 0].set_title('Discrete Distribution')

# Plot Exponential Distribution
N_exponential = 1000
lam_exponential = 0.5
exponential_samples = my_exponential(N_exponential, lam_exponential)
axs[1, 1].hist(exponential_samples, bins=30, density=True, alpha=0.6, color='m', label='Custom Exponential')

# Compare with numpy's implementation
np_exponential_samples = np.random.exponential(scale=1 / lam_exponential, size=N_exponential)
axs[1, 1].hist(np_exponential_samples, bins=30, density=True, alpha=0.6, color='orange', label='Numpy Exponential')

axs[1, 1].set_title('Exponential Distribution')

# Show the plots
for ax in axs.flat:
    ax.legend()

plt.tight_layout()
plt.show()
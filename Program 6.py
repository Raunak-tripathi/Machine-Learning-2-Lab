import random

# Step 1: Function to generate uniform random number in [0,1]
def uniform_random():
    return random.uniform(0, 1)

# Step 2: Define f(x) = x^2
def f(x):
    return x**2

# Step 3: Generate 10 random samples
N = 10
samples = [uniform_random() for _ in range(N)]

# Step 4: Evaluate f(x) for each sample
f_values = [f(x) for x in samples]

print("Random Samples (x):")
print(samples)
print("\nFunction values f(x) = x^2:")
print(f_values)

# Step 5: Monte Carlo estimate of integral
I_est = sum(f_values) / N  # Since interval is [0,1]
print("\nMonte Carlo Estimate of ∫0^1 x^2 dx =", I_est)

# Step 6: True value for comparison
I_true = 1/3
print("True Value =", I_true)


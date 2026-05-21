import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Set random seed
np.random.seed(42)

# Number of records
n = 5000

# Generate dataset
customer_id = np.arange(1001, 1001 + n)
print(customer_id)

age = np.random.randint(21, 60, n)
print(age)

income = np.random.randint(20000, 150000, n)
print(income)

loan_amount = np.random.randint(5000, 500000, n)
print(loan_amount)

credit_score = np.random.randint(300, 900, n)
print(credit_score)

loan_term = np.random.choice([12, 24, 36, 48, 60, 72, 84], n)
print(loan_term)

# 1 means yes and 0 means no #

default_status = np.random.choice([1,0],n,p=[0.2, 0.8])
print(default_status)


# STEP 1 : CENTRAL TENDENCY & DISPERSION

# Mean #
mean_inc=np.mean(income)
print("mean of income is=",mean_inc)

# mediam #
mediam_inc=np.median(income)
print("median of income is=",mediam_inc)

# mode #
mode_inc=stats.mode(income)
print("mode of income is=",mode_inc)

# range #
range_loan=np.max(loan_amount)-np.min(loan_amount)
print("range is=",range_loan)

# variance #
var_loan=np.var(loan_amount)
print("variance of loan amount is=",var_loan)

# s.d #
sd_loan=np.std(loan_amount)
print("sd of loan amount is= ",sd_loan)


# STEP 2 : PROBABILITY & EVENTS

# Probability of Loan Default
default_probability = np.mean(default_status)
print("\nProbability of Default =", default_probability)

# Contingency Table
table = pd.crosstab(["Default_Status"], ["Score_Category"])

print("\nContingency Table:\n")
print(table)

# Conditional Probability
low_score = (credit_score < 600)

conditional_prob = np.mean(low_score[default_status])

print("\nP(Default | Credit Score < 600) =",conditional_prob)

# STEP 3 : DISTRIBUTIONS & VISUALIZATION

# Histogram with Gaussian Curve

plt.hist(credit_score, bins=5, density=True)

# Gaussian Curve
mean = np.mean(credit_score)
std = np.std(credit_score)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)

p = stats.norm.pdf(x, mean, std)

plt.plot(x, p)

plt.title("Histogram of Credit Score")
plt.xlabel("Credit Score")
plt.ylabel("Density")

plt.show()

# Skewness
skewness = stats.skew(loan_amount)
print("\nSkewness =", skewness)

# Kurtosis
kurtosis = stats.kurtosis(loan_amount)
print("Kurtosis =", kurtosis)

# Q-Q Plot
stats.probplot(income, dist="norm", plot=plt)

plt.title("Q-Q Plot of Income")
plt.show()

# STEP 4 : LINEAR ALGEBRA APPLICATION

vectors = list(zip(income, loan_amount))

# First vector
v1 = np.array(vectors[0])

# Second vector
v2 = np.array(vectors[1])

# dot product #

dot_pro=np.dot(v1,v2)
print("dot pro of two vectors is=",dot_pro)

# Norm of Vector

norm_v1 = np.linalg.norm(v1)
print("Norm of First Vector =", norm_v1)

# angle between vectors #

cos_theta=dot_pro/( np.linalg.norm(v1) * np.linalg.norm(v2))

angle=np.degrees(np.arccos(cos_theta))
print("angle of vector is=",angle)

# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for chi-square test
table = np.array([
    [30, 20], # Condition A
    [10, 40]  # Condition B
])

# chi-square test
chi2, p, dof, expected = stats.chi2_contingency(table)
print(f"Chi-square test: chi2={chi2}, p={p}, dof)={dof}")

# Bar plot
labels = ['Condition A', 'Condition B']
counts = [table[0][0], table[1][0]]
plt.bar(labels, counts)
plt.ylabel("Counts")
plt.show()

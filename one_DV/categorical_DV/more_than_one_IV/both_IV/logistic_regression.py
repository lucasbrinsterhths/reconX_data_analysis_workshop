# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 1 categorical DV, 1 continuous IV, and 1 categorical IV

n = 50
IV_A = ['A', 'B']

rows = []

for group in IV_A:
    for _ in range(n):
        IV_B = np.random.normal(50, 10)

        log_odds = -5 + 0.08 + IV_B + (1.2 if group == 'B' else 0)
        prob = 1 / (1 + np.exp(-log_odds))
        dv = np.random.binomial(1, prob)
        rows.append([group, IV_B, dv])

df = pd.DataFrame(rows, columns=['IV_A', 'IV_B', 'DV'])

# Fit logistic regression model
model = smf.logit("DV ~ IV_B + C(IV_A)", data=df).fit()
print(model.summary())

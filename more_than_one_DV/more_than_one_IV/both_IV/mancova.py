# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Generate sample data for multiple IVs (continuous and categorical) and multiple DVs
n = 20
IV_A_conditions = ['A1', 'A2']

rows = []

for a in IV_A_conditions:
    for _ in range(n):
        IV_B = np.random.normal(50, 10)

        dv1 = 30 + (8 if a == 'A2' else 0) + 0.5 * IV_B + np.random.normal(0, 5)
        dv2 = 20 + (5 if a == 'A2' else 0) - 0.3 * IV_B + np.random.normal(0, 5)
        rows.append([a, IV_B, dv1, dv2])

df = pd.DataFrame(rows, columns=['IV_A', 'IV_B', 'DV_1', 'DV_2'])

# mancova
manova = MANOVA.from_formula('DV_1 + DV_2 ~ IV_A + IV_B', data=df)
print(manova.mv_test())

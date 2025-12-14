# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Generate sample data for multiple categorial IVs and multiple DVs
n = 20
IV_A_conditions = ['A1', 'A2']
IV_B_conditions = ['B1', 'B2', 'B3']

rows = []

for a in IV_A_conditions:
    for b in IV_B_conditions:
        for _ in range(n):
            dv1 = 50 + (5 if a == 'A2' else 0) + (3 if b == 'B2' else 0) + (6 if b == 'B3' else 0) + np.random.normal(0, 5)
            dv2 = 30 + (4 if a == 'A2' else 0) + (2 if b == 'B2' else 0) + (5 if b == 'B3' else 0) + np.random.normal(0, 5)
            rows.append([a, b, dv1, dv2])

df = pd.DataFrame(rows, columns=['IV_A', 'IV_B', 'DV_1', 'DV_2'])

# factorial manova
manova = MANOVA.from_formula('DV_1 + DV_2 ~ IV_A + IV_B', data=df)
print(manova.mv_test())

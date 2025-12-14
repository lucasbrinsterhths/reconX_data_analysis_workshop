# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Generate random data (2 IV, 2 DV)
n = 100

IV_A = np.random.normal(50, 10, n)  # Continuous IV
IV_B = np.random.normal(30, 5, n)   # Continuous IV

DV_A = 5 + 0.6 * IV_A + 0.3 * IV_B + np.random.normal(0, 5, n)
DV_B = 10 + 0.4 * IV_A - 0.2 * IV_B + np.random.normal(0, 5, n) 

df = pd.DataFrame({
    'IV_A': IV_A,
    'IV_B': IV_B,
    'DV_A': DV_A,
    'DV_B': DV_B
})

# mancova
manova = MANOVA.from_formula('DV_A + DV_B ~ IV_A + IV_B', data=df)
print(manova.mv_test())

# Plot DV_A and DV_B against IV_A and IV_B
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].scatter(df['IV_A'], df['DV_A'], alpha=0.6)
axs[0].scatter(df['IV_A'], df['DV_B'], alpha=0.6, color ='orange')
axs[0].set_xlabel('IV_A')
axs[0].set_ylabel('DV_A')
axs[0].set_title('DV vs IV_A')
axs[1].scatter(df['IV_B'], df['DV_A'], alpha=0.6)
axs[1].scatter(df['IV_B'], df['DV_B'], alpha=0.6, color ='orange')
axs[1].set_xlabel('IV_B')
axs[1].set_ylabel('DV_B')
axs[1].set_title('DV vs IV_B')
plt.tight_layout()
plt.show()
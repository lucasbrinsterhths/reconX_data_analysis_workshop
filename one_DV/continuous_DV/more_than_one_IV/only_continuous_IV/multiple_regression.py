# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 2 DVs and 1 IV
df = pd.DataFrame({
    'DV_1': np.random.normal(40, 10, 100),
    "DV_2": np.random.normal(25, 4, 100)
})

df["IV"] = 0.5 * df.DV_1 + 2 * df.DV_2 + np.random.normal(0, 10, 100)

# multiple regression
model = smf.ols('IV ~ DV_1 + DV_2', data=df).fit()
print(model.summary())

# 3D Scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.DV_1, df.DV_2, df.IV)
ax.set_xlabel('DV 1')
ax.set_ylabel('DV 2')
ax.set_zlabel('IV')
plt.show()

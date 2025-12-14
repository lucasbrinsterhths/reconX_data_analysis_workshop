# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for x and y
df = pd.DataFrame({
    'x': np.random.uniform(0, 10, 50),
})

df['y'] = 2*df.x + np.random.normal(0, 3, 50)

# Pearson correlation and regression analysis
r, p = stats.pearsonr(df.x, df.y)
print(f"Pearson correlation coefficient: {r}, p-value: {p}")

# Scatter plot with regression line
plt.scatter(df.x, df.y)
plt.plot(df.x, np.poly1d(np.polyfit(df.x, df.y, 1))(df.x), color='red')
plt.xlabel("X"); plt.ylabel("Y")
plt.show()
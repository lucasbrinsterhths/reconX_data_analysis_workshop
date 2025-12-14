# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data
df = pd.DataFrame({
    'x': np.random.normal(0, 1, 100),
})

df['p'] = 1/(1+np.exp(-df.x))
df['y'] = np.random.binomial(1, df.p)

# logistic regression
model = sm.Logit(df.y, sm.add_constant(df.x)).fit()
print(model.summary())

# Scatter plot with logistic regression curve
plt.scatter(df.x, df.y, alpha=0.5)
x_vals = np.linspace(df.x.min(), df.x.max(), 100)
y_vals = model.predict(sm.add_constant(x_vals))
plt.plot(x_vals, y_vals, color='red')
plt.xlabel("X")
plt.ylabel("Probability of Y=1")
plt.show()

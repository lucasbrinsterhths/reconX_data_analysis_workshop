# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate IV Data
df = pd.DataFrame({
    'IV_1': np.random.normal(50, 12, 200),
    'IV_2': np.random.normal(27, 4, 200),
    'IV_3': np.random.normal(75, 20, 200)
})

# Generate DV Data
log_odds = (
    -8
    + 0.04 * df.IV_1
    + 0.012 * df.IV_2
    + 0.03 * df.IV_3
)

prob = 1 / (1 + np.exp(-log_odds))
df['DV'] = np.random.binomial(1, prob)

# Multiple Logistic Regression
model = sm.Logit(df.DV, sm.add_constant(df[['IV_1', 'IV_2', 'IV_3']])).fit()
print(model.summary())

a_range = np.linspace(df.IV_1.min(), df.IV_1.max(), 100)

X_plot = pd.DataFrame({
    'IV_1': a_range,
    'IV_2': df.IV_2.mean(),
    'IV_3': df.IV_3.mean()
})

X_plot = sm.add_constant(X_plot, has_constant = 'add')
pred_prob = model.predict(X_plot)

plt.scatter(df.IV_1, df.DV, alpha=0.3, label='Data')
plt.plot(a_range, pred_prob, color='red', label='Logistic Regression Fit')
plt.xlabel("IV_1")
plt.ylabel("Probability of DV=1")
plt.legend()
plt.show()

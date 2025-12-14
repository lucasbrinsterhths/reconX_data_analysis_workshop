# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 2 IVs (1 cont, 1 cat) and 1 DV
n = 30

IV_B = ['A', 'B', 'C']

rows = []

for group in IV_B:
    for _ in range(n):
        
        IV_A = np.random.normal(50, 10)
        dv = (
            30
            + (8 if group == 'B' else 0)
            + 0.6 * IV_A
            + np.random.normal(0, 5)
        )

        rows.append([group, IV_A, dv])

df = pd.DataFrame(rows, columns=['IV_B', 'IV_A', 'DV'])

# ancova
model = smf.ols('DV ~ C(IV_B) + IV_A', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

plt.figure(figsize=(7, 5))

for g in IV_B:
    sub = df[df["IV_B"] == g]
    plt.scatter(sub["IV_A"], sub["DV"], alpha=0.6, label=g)

# Common slope visualization
x = np.linspace(df.IV_A.min(), df.IV_A.max(), 100)
for g in IV_B:
    intercept = model.params["Intercept"] + (
        model.params["C(IV_B)[T.Treatment]"] if g == "Treatment" else 0
    )
    y = intercept + model.params["IV_A"] * x
    plt.plot(x, y)

plt.xlabel("IV_A")
plt.ylabel("DV")
plt.title("ANCOVA: Group Differences Adjusted for Covariate")
plt.legend()
plt.show()

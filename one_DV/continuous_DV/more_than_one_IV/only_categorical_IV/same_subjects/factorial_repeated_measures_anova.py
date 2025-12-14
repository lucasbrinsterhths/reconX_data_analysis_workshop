# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for 2 IVs and 1 DV
n_subjects = 30
subjects = np.arange(n_subjects)

IV_A = ["A1", "A2"]
IV_B = ["B1", "B2", "B3"]

rows = []
for subject in subjects:
    subject_offset = np.random.normal(0, 5)

    for a in IV_A:
        for b in IV_B:

            dv = (
                50
                + (5 if a == "A2" else 0)
                + ({"B1": 0, "B2": 3, "B3": 6}[b])
                + subject_offset
                + np.random.normal(0, 2)
            )

            rows.append([subject, a, b, dv])

df = pd.DataFrame(rows, columns=["subject", "IV_A", "IV_B", "DV"])

anova = AnovaRM(df, 'DV', 'subject', within=['IV_A', 'IV_B']).fit()
print(anova)

means = (
    df.groupby(['IV_A', 'IV_B'])['DV']
    .mean()
    .unstack()
)

plt.plot(means.T, marker='o')
plt.xlabel('IV_B')
plt.ylabel('Mean DV')
plt.title("Interaction Plot")
plt.legend(means.index, title='IV_A')
plt.show()

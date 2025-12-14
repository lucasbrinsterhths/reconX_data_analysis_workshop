# Imports (Using the same for all files in the repo for consistency)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA

# Gernerate sample data for multiple categorial IVs and one categorical DV

IV_A_conditions = ['A1', 'A2']
IV_B_conditions = ['B1', 'B2', 'B3']
DV_conditions = ['Yes', 'No']

rows = []
for a in IV_A_conditions:
    for b in IV_B_conditions:
        for dv in DV_conditions:
            count = np.random.poisson(lam = 20)
            rows.append([a, b, dv, count])

df = pd.DataFrame(rows, columns=['IV_A', 'IV_B', 'DV', 'Count'])    

# Loglinear Model
model = smf.glm(
    formula = 'Count ~ IV_A * IV_B * DV',
    data = df,
    family = sm.families.Poisson()
).fit()

print(model.summary())

df['Predicted'] = model.predict(df)

# heatmap visualization
pivot_table = df.pivot_table(values='Predicted', index='IV_A', columns='IV_B', aggfunc='sum')
plt.imshow(pivot_table, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Predicted Counts')
plt.xticks(ticks=np.arange(len(IV_B_conditions)), labels=IV_B_conditions)
plt.yticks(ticks=np.arange(len(IV_A_conditions)), labels=IV_A_conditions)
plt.xlabel('IV_B')
plt.ylabel('IV_A')
plt.title('Loglinear Model Predicted Counts Heatmap')
plt.show()
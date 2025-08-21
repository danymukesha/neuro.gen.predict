import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

genotype_data = pd.DataFrame({
    'APOE_e4': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.25, 0.05]),
    'APOE_e2': np.random.choice([0, 1, 2], n_samples, p=[0.85, 0.14, 0.01]),
    'TREM2': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
    'BIN1': np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.5, 0.2]),
    'CLU': np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.45, 0.15]),
    'PICALM': np.random.choice([0, 1, 2], n_samples, p=[0.35, 0.5, 0.15])
})

risk_factors = (genotype_data['APOE_e4'] * 0.5 + 
                genotype_data['TREM2'] * 1.0 + 
                genotype_data['BIN1'] * 0.1)

y = np.random.binomial(1, 1 / (1 + np.exp(-risk_factors + 1.5)))

genotype_data.to_csv('examples/example_genotypes.csv', index=False)
pd.DataFrame({'label': y}).to_csv('examples/example_labels.csv', index=False)

print("Example data generated: example_genotypes.csv and example_labels.csv")

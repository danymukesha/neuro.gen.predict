import pandas as pd
from neurogenpredict import NeuroGenPredict

# Load example data (replace with your real data)
genotype_data = pd.read_csv('examples/example_genotypes.csv')
y = pd.read_csv('examples/example_labels.csv')['label'].values

predictor = NeuroGenPredict(population="EUR")
X = predictor.prepare_features(genotype_data)
cv_scores = predictor.train_ensemble(X, y)
print("CV Scores:", cv_scores)

# Save trained model
predictor.save('examples/trained_model.pkl')

# Predict on first sample
test_X = X.iloc[[0]]  # Single sample
predictions = predictor.predict_risk(test_X)
report = predictor.generate_report(predictions, "Test_Sample", index=0)
print(report)

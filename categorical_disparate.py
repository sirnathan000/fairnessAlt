import pandas as pd
from sklearn.preprocessing import LabelEncoder
from aif360.sklearn.metrics import disparate_impact_ratio

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
                "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "income"]
data = pd.read_csv(url, names=column_names, na_values=" ?", skipinitialspace=True)

# Clean the dataset
data.dropna(inplace=True)

# Convert income to binary outcome
data['income'] = data['income'].apply(lambda x: 1 if x == ">50K" else 0)

print(type(data['sex']))

# Encode the protected attribute (e.g., 'sex')
protected_attr = data['sex']
le = LabelEncoder()
protected_attr = le.fit_transform(protected_attr)

outcomes = data['income']

# Check counts of protected attribute and favorable outcomes
print("Counts of protected attribute (sex):")
print(pd.Series(protected_attr).value_counts())
print("Counts of favorable outcomes (income >50K):")
print(outcomes.value_counts())

# Calculate disparate impact ratio
ratio = disparate_impact_ratio(y_true=outcomes, y_pred=outcomes, prot_attr=protected_attr)
print(f"Disparate Impact Ratio (Race): {ratio:.2f}")
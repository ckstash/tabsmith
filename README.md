# TabSmith

**TabSmith** is a Python library that can fit multi-target tabular ML classifiers with overlapping inputs and targets. It is a flexible, scikit‑learn‑compatible framework for repurposing multi-target classifiers to work as both conventional classifiers and denoisers depending on the inputs. The concept is described in [this article](https://medium.com/data-science/dawn-of-the-denoisers-multi-output-ml-models-for-tabular-data-imputation-317711d7a193).

---

## ✨ Features

- **Auto‑encoding of categoricals**: Automatically detects and encodes `object`/categorical columns with `LabelEncoder`.
- **Masking for denoising**: Randomly mask input values during training to teach the model to impute missing data.
- **Overlap‑aware prediction**: If a column is both an input and a target, known values are preserved and only missing values are predicted.
- **Multi‑target support**: Train and predict multiple target columns at once.
- **Holdout & cross‑validation**: Built‑in evaluation on a holdout set and K‑fold CV with per‑target metrics.
- **Feature importances**: Retrieve and plot raw or normalized feature importances.
- **Human‑readable decoding**: Map numeric predictions and probabilities back to original labels.

---

## 📦 Installation

```bash
pip install tabsmith
```

Built for Python 3.12 or above.

---

## 🚀 Quick Start

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tabsmith.model import TSModel
from tabsmith.utils import mask_df

# Load your dataset
df = pd.read_csv("titanic.csv")

input_features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
target_features = ["Survived", "Pclass", "Sex", "SibSp"]

# Initialize base model and TabSmith wrapper
base_model = RandomForestClassifier(n_estimators=100, random_state=42)
model = TSModel(base_model=base_model)

# Fit with auto‑encoding and masking
model.fit(
    df,
    input_columns=input_features,
    target_columns=target_features,
    test_prop=0.2,
    masking_value=-1,
    masking_prob=0.5,
    random_seed=42,
    upsampling_factor=2,
)

# Evaluate on holdout set
metrics = model.evaluate_holdout()
print(metrics)

# Predict on masked data
X_masked = mask_df(df.iloc[:5], masking_value=-1, masking_prob=0.5, seed=42)
preds = model.predict(X_masked)
print(model.decode_predictions(preds))

# Feature importances
feature_importances = model.feature_importances(normalized=True)
print(feature_importances)
```

---

## 📊 Plotting Feature Importances

```python
from tabsmith.utils import plot_feature_importances

plot_feature_importances(feature_importances, title="Normalized Feature Importances")
```

---

## 📈 Cross‑Validation

```python
cv_results = model.cross_validate_kfold(
    df,
    input_columns=input_features,
    target_columns=target_features,
    k=3,
    masking_value=-1,
    masking_prob=0.5,
    random_seed=42,
)
print(cv_results)
```

---

## 🧪 Utilities

- `encode_dataframe(df)`: Encode all categorical columns with `LabelEncoder`.
- `mask_df(df, masking_value, masking_prob, seed)`: Randomly mask entries in a DataFrame.
- `pretty_print_holdout(metrics)`: Nicely format holdout metrics.
- `plot_feature_importances(importances)`: Plot feature importances from a dict or list.

---

## 📚 API Reference

See [here](https://github.com/ckstash/tabsmith/blob/main/API.md)

---

## 📜 License

This project is licensed under the terms of the [MIT License](https://github.com/ckstash/tabsmith/blob/main/LICENSE).


<a id="tabsmith"></a>

# tabsmith

<a id="tabsmith.model"></a>

# tabsmith.model

<a id="tabsmith.model.TSModel"></a>

## TSModel Objects

```python
class TSModel()
```

TSModel trains a multi-output classifier to predict original (clean) values
from stochastically masked inputs.

This supports:
1) Typical supervised learning with explicit input/target columns.
2) Tabular imputation when some columns are both inputs and targets. At
inference, overlapping columns are treated as inputs if a value is provided,
or as targets (to be imputed) if missing.

A "missing" value is any entry equal to `masking_value` or `np.nan`.

**Attributes**:

- `base_model` _ClassifierMixin_ - User-provided multi-output classifier prototype.
- `mask_prob` _float_ - Probability used to mask any given input cell during training.
- `fitted_model` _Optional[ClassifierMixin]_ - Trained classifier after `fit`.
- `input_columns` _Optional[List[Union[str, int]]]_ - Columns used as inputs.
- `target_columns` _Optional[List[Union[str, int]]]_ - Columns used as targets.
- `masking_value` _Optional[Union[float, int]]_ - Placeholder used to denote missingness.
- `df_holdout` _Optional[pd.DataFrame]_ - Held-out evaluation dataframe (unmodified).
- `df_holdout_masked` _Optional[pd.DataFrame]_ - Masked inputs corresponding to the holdout.

<a id="tabsmith.model.TSModel.__init__"></a>

#### \_\_init\_\_

```python
def __init__(base_model: ClassifierMixin, mask_prob: float = 0.5) -> None
```

Initialize TSModel.

**Arguments**:

- `base_model` _ClassifierMixin_ - Any multi-output-capable classifier
  (e.g., DecisionTreeClassifier, RandomForestClassifier). It is cloned in fit().
- `mask_prob` _float_ - Probability of masking an input cell during denoising training.
  Must be in [0, 1].
  

**Raises**:

- `ValueError` - If mask_prob is not in [0, 1].

<a id="tabsmith.model.TSModel.fit"></a>

#### fit

```python
def fit(df: Union[pd.DataFrame, np.ndarray],
        input_columns: Optional[Iterable] = None,
        target_columns: Optional[Iterable] = None,
        test_prop: float = 0.2,
        masking_value: Union[float, int, None] = -1.0,
        random_seed: int = 42,
        upsampling_factor: int = 1) -> "TSModel"
```

Train the classifier as a denoiser on masked inputs, with automatic encoding of categorical columns.

This method:
* Detects and label-encodes string/categorical columns in both inputs and targets.
* Ensures the masking value is part of each encoder's classes.
* Optionally splits into training and holdout sets.
* Optionally upsamples the training set.
* Masks inputs randomly according to `mask_prob`.
* Drops rows with missing targets before fitting.

**Arguments**:

- `df` - Full dataset containing both input and target columns.
- `input_columns` - Columns to use as inputs. If None, inferred.
- `target_columns` - Columns to predict. If None, inferred.
- `test_prop` - Fraction of data to hold out for evaluation.
- `masking_value` - Value used to represent masked entries.
- `random_seed` - Random seed for reproducibility.
- `upsampling_factor` - Multiplier for training rows before masking.
  

**Returns**:

- `self` - The fitted model.

<a id="tabsmith.model.TSModel.predict"></a>

#### predict

```python
def predict(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame
```

Predict target columns from masked inputs, with overlap-aware imputation.

This method generates predictions for the target columns based on the provided
(already masked) input data. If a column is both an input and a target, any
non-missing values in the input are preserved in the output; missing values
are imputed using the model's predictions.

**Arguments**:

- `X` _Union[pd.DataFrame, np.ndarray]_ - Input data containing at least the
  columns specified in `input_columns` during `fit()`. May also include
  overlapping target columns.
  

**Returns**:

- `pd.DataFrame` - A DataFrame of predictions for the target columns. For
  overlapping columns, provided non-missing input values are retained;
  missing values are filled with model predictions.
  

**Raises**:

- `RuntimeError` - If the model has not been fitted.
- `ValueError` - If any required input columns are missing from `X`.
  

**Notes**:

  - `X` must already be masked using the same `masking_value` provided to `fit()`.
  This method does not perform additional masking.
  - Categorical columns are automatically encoded using the encoders learned
  during `fit()`, with masking values and unseen categories handled gracefully.

<a id="tabsmith.model.TSModel.predict_proba"></a>

#### predict\_proba

```python
def predict_proba(
        X: Union[pd.DataFrame,
                 np.ndarray]) -> Dict[Union[str, int], np.ndarray]
```

Return class probability distributions for each target variable.

For multi-output classifiers, scikit-learn typically returns a list of arrays
(one per output) from `predict_proba`. This method standardizes that output
into a dictionary mapping each target column name to its corresponding
probability array.

**Arguments**:

- `X` _Union[pd.DataFrame, np.ndarray]_ - Input data containing at least the
  columns specified in `input_columns` during `fit()`.
  

**Returns**:

  Dict[Union[str, int], np.ndarray]: A mapping from each target column name
  to a 2D NumPy array of shape `(n_samples, n_classes_for_target)`, where
  each row contains the predicted class probabilities for that sample.
  

**Raises**:

- `RuntimeError` - If the model has not been fitted.
- `AttributeError` - If the fitted base model does not implement `predict_proba`.
- `ValueError` - If any required input columns are missing from `X`.
- `RuntimeError` - If the `predict_proba` output format is unexpected for a
  multi-output model.
  

**Notes**:

  - `X` must already be masked using the same `masking_value` provided to `fit()`.
  - Categorical columns are automatically encoded using the encoders learned
  during `fit()`, with masking values and unseen categories handled gracefully.
  - For single-output models, the returned dictionary contains a single key
  corresponding to the target column.

<a id="tabsmith.model.TSModel.decode_predictions"></a>

#### decode\_predictions

```python
def decode_predictions(df_preds: pd.DataFrame) -> pd.DataFrame
```

Decode numeric predictions back to original categorical labels.

This method uses the label encoders stored during `fit()` to convert
integer-encoded predictions into their original string or categorical
representations for each encoded column.

**Arguments**:

- `df_preds` _pd.DataFrame_ - DataFrame containing model predictions in
  numeric (encoded) form. Column names should match those used
  during training.
  

**Returns**:

- `pd.DataFrame` - A copy of `df_preds` with encoded columns decoded back
  to their original labels.
  

**Notes**:

  - Only columns present in both `df_preds` and `self.encoders_` are decoded.
  - Columns without a stored encoder are returned unchanged.

<a id="tabsmith.model.TSModel.decode_predict_proba"></a>

#### decode\_predict\_proba

```python
def decode_predict_proba(probas: dict) -> dict
```

Map `predict_proba` outputs back to original class labels.

Converts the probability arrays returned by `predict_proba` into
dictionaries keyed by the original class labels from the stored
encoders. This makes the output human-readable and consistent with
the original data.

**Arguments**:

- `probas` _dict_ - Mapping from target column name to a 2D NumPy array
  of shape `(n_samples, n_classes_for_target)` containing class
  probabilities.
  

**Returns**:

- `dict` - A mapping from each target column name to a dictionary where
  keys are original class labels and values are 1D NumPy arrays of
  probabilities for that class.
  

**Notes**:

  - If the number of probability columns is fewer than the number of
  classes in the encoder (e.g., due to missing classes in training),
  the label list is trimmed to match the probability array width.
  - Columns without a stored encoder are returned with stringified
  integer indices as keys.

<a id="tabsmith.model.TSModel.feature_importances"></a>

#### feature\_importances

```python
def feature_importances(
        normalized: bool = False) -> Dict[Union[str, int], float]
```

Return feature importances aggregated across outputs if needed.

If the fitted model exposes `feature_importances_`, those are returned.
If it is a multi-output wrapper exposing per-output importances (e.g., via
`estimators_`), importances are averaged across outputs.

**Arguments**:

- `normalized` _bool_ - If True, min-max normalize the importances to [0, 1].
  

**Returns**:

  Dict[Union[str, int], float]: Mapping input feature -> importance.
  

**Raises**:

- `RuntimeError` - If model is not fitted.
- `AttributeError` - If no feature importance information can be extracted.

<a id="tabsmith.model.TSModel.evaluate_holdout"></a>

#### evaluate\_holdout

```python
def evaluate_holdout(
        average: str = "macro",
        zero_division: int = 0) -> Dict[str, Union[float, Dict[str, float]]]
```

Evaluate model performance on the internal holdout set.

Computes accuracy, precision, recall, and F1-score for each target column
in the holdout set, as well as macro-averaged metrics across all targets.

**Arguments**:

- `average` _str, optional_ - Averaging method for precision, recall, and F1-score.
  Passed directly to scikit-learn's metric functions. Defaults to `"macro"`.
- `zero_division` _int, optional_ - Value to return when there is a zero division
  in precision or recall calculation. Passed directly to scikit-learn's
  metric functions. Defaults to `0`.
  

**Returns**:

  Dict[str, Union[float, Dict[str, float]]]: A dictionary containing:
  - **accuracy** (float): Macro-averaged accuracy across all targets.
  - **precision** (float): Macro-averaged precision across all targets.
  - **recall** (float): Macro-averaged recall across all targets.
  - **f1** (float): Macro-averaged F1-score across all targets.
  - **per_target** (dict): Mapping from each target column name to its own
  metrics dictionary with keys `"accuracy"`, `"precision"`, `"recall"`,
  and `"f1"`.
  

**Raises**:

- `RuntimeError` - If the model has not been fitted or if holdout data is not
  available (e.g., `fit()` was not called with `test_prop > 0`).
  

**Notes**:

  - The holdout set is created during `fit()` when `test_prop > 0`.
  - Predictions are generated using the current fitted model and compared
  against the true labels in the holdout set.

<a id="tabsmith.model.TSModel.cross_validate_kfold"></a>

#### cross\_validate\_kfold

```python
def cross_validate_kfold(df: pd.DataFrame,
                         input_columns=None,
                         target_columns=None,
                         k: int = 5,
                         masking_value=-1,
                         random_seed: int = 42,
                         upsampling_factor: int = 1)
```

Perform K-fold cross-validation with clean per-target metrics and masked-value handling.

<a id="tabsmith.utils"></a>

# tabsmith.utils

<a id="tabsmith.utils.encode_dataframe"></a>

#### encode\_dataframe

```python
def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame
```

Encode all object or categorical columns in a DataFrame as integer labels.

This function creates a copy of the input DataFrame and applies
`sklearn.preprocessing.LabelEncoder` to each column with dtype `object`
or `CategoricalDtype`. All values are converted to strings before encoding.

**Arguments**:

- `df` _pd.DataFrame_ - The input DataFrame containing columns to encode.
  

**Returns**:

- `pd.DataFrame` - A copy of the input DataFrame with categorical columns
  replaced by integer-encoded values.
  

**Notes**:

  - The encoders used are not returned or stored; this is intended for
  quick, stateless encoding (e.g., in tests).
  - All non-object, non-categorical columns are left unchanged.

<a id="tabsmith.utils.mask_df"></a>

#### mask\_df

```python
def mask_df(df: pd.DataFrame, mask_prob: float, masking_value,
            seed: int) -> pd.DataFrame
```

Randomly mask entries in a DataFrame.

This utility is intended for testing. It randomly replaces a proportion
of entries in the DataFrame with a specified masking value or NaN.

**Arguments**:

- `df` _pd.DataFrame_ - The input DataFrame to mask.
- `mask_prob` _float_ - Probability of masking each individual cell
  (between 0 and 1).
- `masking_value` - Value to insert in masked positions. If `None` or NaN,
  masked entries are set to `np.nan`.
- `seed` _int_ - Random seed for reproducibility.
  

**Returns**:

- `pd.DataFrame` - A copy of the input DataFrame with some entries masked.
  

**Notes**:

  - Masking is applied independently to each cell.
  - If `masking_value` is a float NaN or `None`, pandas' `.mask()` is used.

<a id="tabsmith.utils.pretty_print_holdout"></a>

#### pretty\_print\_holdout

```python
def pretty_print_holdout(metrics: dict) -> None
```

Pretty-print overall and per-target holdout metrics.

Formats and prints the metrics dictionary returned by a model's
`evaluate_holdout()` method in a tabular form.

**Arguments**:

- `metrics` _dict_ - Dictionary containing overall metrics (`accuracy`,
  `precision`, `recall`, `f1`) and a `per_target` sub-dictionary
  mapping each target name to its own metrics.
  

**Returns**:

- `None` - This function prints to stdout.

<a id="tabsmith.utils.plot_feature_importances"></a>

#### plot\_feature\_importances

```python
def plot_feature_importances(importances: Union[Dict[str, float],
                                                List[Tuple[str, float]]],
                             title: str = "Feature Importances",
                             color: str = "steelblue") -> None
```

Plot feature importances as a horizontal bar chart.

Accepts either a dictionary mapping feature names to importance values
or a list of (feature, importance) tuples, sorts them in descending order
of importance, and plots them.

**Arguments**:

- `importances` _Union[Dict[str, float], List[Tuple[str, float]]]_ - Feature
  importances as a dict or list of tuples.
- `title` _str, optional_ - Title for the plot. Defaults to "Feature Importances".
- `color` _str, optional_ - Color of the bars. Defaults to "steelblue".
  

**Returns**:

- `None` - Displays a matplotlib plot.
  

**Raises**:

- `TypeError` - If `importances` is not a dict or list of tuples.
  

**Notes**:

  - Numpy scalar values are converted to Python floats for plotting.
  - The y-axis is inverted so the most important feature appears at the top.


import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Tuple
from tabulate import tabulate


def encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Encode all object or categorical columns in a DataFrame as integer labels.

    This function creates a copy of the input DataFrame and applies
    `sklearn.preprocessing.LabelEncoder` to each column with dtype `object`
    or `CategoricalDtype`. All values are converted to strings before encoding.

    Args:
        df (pd.DataFrame): The input DataFrame containing columns to encode.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with categorical columns
        replaced by integer-encoded values.

    Notes:
        - The encoders used are not returned or stored; this is intended for
          quick, stateless encoding (e.g., in tests).
        - All non-object, non-categorical columns are left unchanged.
    """
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or isinstance(df_encoded[col].dtype, CategoricalDtype):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded


def mask_df(df: pd.DataFrame, masking_value: Union[float, int, None] = -1.0, masking_prob: float = 0.5, seed: int = 42) -> pd.DataFrame:
    """Randomly mask entries in a DataFrame.

    This utility is intended for testing purposes. It creates a copy of the
    input DataFrame and randomly replaces a proportion of its entries with
    a specified masking value or ``NaN``.

    Args:
        df (pd.DataFrame):
            The input DataFrame to apply masking to.
        masking_value (Union[float, int, None], optional):
            Value to insert in masked positions. If ``None`` or ``NaN``,
            masked entries are set to ``numpy.nan``. Defaults to ``-1.0``.
        masking_prob (float, optional):
            Probability of masking each individual cell, between 0 and 1.
            Defaults to ``0.5``.
        seed (int, optional):
            Random seed for reproducibility. Defaults to ``42``.

    Returns:
        pd.DataFrame:
            A copy of the input DataFrame with some entries masked.

    Notes:
        - Masking is applied independently to each cell.
        - If ``masking_value`` is ``None`` or a float ``NaN``, pandas'
          :meth:`DataFrame.mask` is used to insert ``NaN`` values.
    """
    rng = np.random.default_rng(seed)
    mask = rng.random(size=df.shape) < masking_prob
    out = df.copy()
    if masking_value is None or (isinstance(masking_value, float) and np.isnan(masking_value)):
        return out.mask(mask, other=np.nan)
    vals = out.values
    vals[mask] = masking_value
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


def pretty_print_holdout(metrics: dict) -> None:
    """Pretty-print overall and per-target holdout metrics.

    Formats and prints the metrics dictionary returned by a model's
    `evaluate_holdout()` method in a tabular form.

    Args:
        metrics (dict): Dictionary containing overall metrics (`accuracy`,
            `precision`, `recall`, `f1`) and a `per_target` sub-dictionary
            mapping each target name to its own metrics.

    Returns:
        None: This function prints to stdout.
    """
    # Overall metrics
    overall = [
        ["Accuracy", f"{metrics['accuracy']:.4f}"],
        ["Precision", f"{metrics['precision']:.4f}"],
        ["Recall", f"{metrics['recall']:.4f}"],
        ["F1-score", f"{metrics['f1']:.4f}"],
    ]
    print("\n=== Overall Metrics ===")
    print(tabulate(overall, headers=["Metric", "Value"], tablefmt="github"))

    # Per-target metrics
    per_target_rows = []
    for target, vals in metrics["per_target"].items():
        per_target_rows.append([
            target,
            f"{vals['accuracy']:.4f}",
            f"{vals['precision']:.4f}",
            f"{vals['recall']:.4f}",
            f"{vals['f1']:.4f}",
        ])
    print("\n=== Per-Target Metrics ===")
    print(tabulate(per_target_rows, headers=["Target", "Accuracy", "Precision", "Recall", "F1"], tablefmt="github"))


def plot_feature_importances(
    importances: Union[Dict[str, float], List[Tuple[str, float]]],
    title: str = "Feature Importances",
    color: str = "steelblue"
) -> None:
    """Plot feature importances as a horizontal bar chart.

    Accepts either a dictionary mapping feature names to importance values
    or a list of (feature, importance) tuples, sorts them in descending order
    of importance, and plots them.

    Args:
        importances (Union[Dict[str, float], List[Tuple[str, float]]]): Feature
            importances as a dict or list of tuples.
        title (str, optional): Title for the plot. Defaults to "Feature Importances".
        color (str, optional): Color of the bars. Defaults to "steelblue".

    Returns:
        None: Displays a matplotlib plot.

    Raises:
        TypeError: If `importances` is not a dict or list of tuples.

    Notes:
        - Numpy scalar values are converted to Python floats for plotting.
        - The y-axis is inverted so the most important feature appears at the top.
    """
    # Normalise to list of (feature, importance) pairs
    if isinstance(importances, dict):
        items = list(importances.items())
    elif isinstance(importances, list):
        items = importances
    else:
        raise TypeError("importances must be a dict or list of (feature, importance) tuples")

    # Convert any numpy scalar to Python float
    items = [(feat, float(val)) for feat, val in items]

    # Sort by importance descending
    items.sort(key=lambda x: x[1], reverse=True)

    features, scores = zip(*items)

    # Plot horizontal bar chart
    plt.figure(figsize=(8, 6))
    plt.barh(features, scores, color=color)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title(title)
    plt.gca().invert_yaxis()  # highest importance at top
    plt.tight_layout()
    plt.show()


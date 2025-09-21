import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, Iterable, List, Optional, Tuple, Union
import warnings


class TSModel:
    """TSModel trains a multi-output classifier to predict original (clean) values
    from stochastically masked inputs. 
    
    This supports:
      1) Typical supervised learning with explicit input/target columns.
      2) Tabular imputation when some columns are both inputs and targets. At
         inference, overlapping columns are treated as inputs if a value is provided,
         or as targets (to be imputed) if missing.

    A "missing" value is any entry equal to `masking_value` or `np.nan`.

    Attributes:
        base_model (ClassifierMixin): User-provided multi-output classifier prototype.
        mask_prob (float): Probability used to mask any given input cell during training.
        fitted_model (Optional[ClassifierMixin]): Trained classifier after `fit`.
        input_columns (Optional[List[Union[str, int]]]): Columns used as inputs.
        target_columns (Optional[List[Union[str, int]]]): Columns used as targets.
        masking_value (Optional[Union[float, int]]): Placeholder used to denote missingness.
        df_holdout (Optional[pd.DataFrame]): Held-out evaluation dataframe (unmodified).
        df_holdout_masked (Optional[pd.DataFrame]): Masked inputs corresponding to the holdout.
    """

    def __init__(self, base_model: ClassifierMixin, mask_prob: float = 0.5) -> None:
        """Initialize TSModel.

        Args:
            base_model (ClassifierMixin): Any multi-output-capable classifier
                (e.g., DecisionTreeClassifier, RandomForestClassifier). It is cloned in fit().
            mask_prob (float): Probability of masking an input cell during denoising training.
                Must be in [0, 1].

        Raises:
            ValueError: If mask_prob is not in [0, 1].
        """
        if not (0.0 <= mask_prob <= 1.0):
            raise ValueError("mask_prob must be in [0, 1].")
        self.base_model = base_model
        self.mask_prob = mask_prob

        self.fitted_model: Optional[ClassifierMixin] = None
        self.input_columns: Optional[List[Union[str, int]]] = None
        self.target_columns: Optional[List[Union[str, int]]] = None
        self.masking_value: Optional[Union[float, int]] = None

        self.df_holdout: Optional[pd.DataFrame] = None
        self.df_holdout_masked: Optional[pd.DataFrame] = None

    # ------------------------
    # Internal utilities
    # ------------------------
    @staticmethod
    def _to_dataframe(X: Union[pd.DataFrame, np.ndarray], columns: Optional[Iterable] = None) -> pd.DataFrame:
        """Convert input to DataFrame, preserving or assigning columns.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.
            columns (Optional[Iterable]): Column names to assign if X is array-like.

        Returns:
            pd.DataFrame: DataFrame representation of X.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        X = np.asarray(X)
        if columns is None:
            columns = list(range(X.shape[1]))
        return pd.DataFrame(X, columns=columns)

    def _is_missing_series(self, s: pd.Series) -> pd.Series:
        """Return boolean mask of missing values based on masking_value or NaN.

        Args:
            s (pd.Series): Series to check.

        Returns:
            pd.Series: Boolean mask, True where missing.
        """
        m = s.isna()
        if self.masking_value is not None and not (isinstance(self.masking_value, float) and np.isnan(self.masking_value)):
            m = m | (s == self.masking_value)
        return m

    def _random_masking(self, df_inputs: pd.DataFrame, random_seed: int) -> pd.DataFrame:
        """Apply random masking to inputs for denoising training.

        Args:
            df_inputs (pd.DataFrame): Input feature dataframe to mask.
            random_seed (int): Seed for reproducibility.

        Returns:
            pd.DataFrame: Masked inputs.
        """
        rng = np.random.default_rng(random_seed)
        mask = rng.random(size=df_inputs.shape) < self.mask_prob
        df_masked = df_inputs.copy()

        # Use np.nan if masking_value is None or NaN; else use masking_value
        if self.masking_value is None or (isinstance(self.masking_value, float) and np.isnan(self.masking_value)):
            df_masked = df_masked.mask(mask, other=np.nan)
        else:
            # .values for fast boolean indexing
            vals = df_masked.values
            vals[mask] = self.masking_value
            df_masked = pd.DataFrame(vals, index=df_inputs.index, columns=df_inputs.columns)

        return df_masked

    def _resolve_columns(
        self,
        df: pd.DataFrame,
        input_columns: Optional[Iterable] = None,
        target_columns: Optional[Iterable] = None,
    ) -> Tuple[List, List]:
        """Resolve default columns and validate overlap semantics.

        Args:
            df (pd.DataFrame): Dataset containing all candidate columns.
            input_columns (Optional[Iterable]): Input feature columns.
            target_columns (Optional[Iterable]): Target columns.

        Returns:
            Tuple[List, List]: (input_columns, target_columns)

        Raises:
            ValueError: If provided columns are not present in df.
        """
        if input_columns is None and target_columns is None:
            cols = list(df.columns)
            return cols, cols

        if input_columns is None:
            input_columns = list(set(df.columns) - set(target_columns))
        if target_columns is None:
            target_columns = list(set(df.columns) - set(input_columns))

        input_columns = list(input_columns)
        target_columns = list(target_columns)

        missing_inputs = [c for c in input_columns if c not in df.columns]
        missing_targets = [c for c in target_columns if c not in df.columns]
        if missing_inputs:
            raise ValueError(f"Input columns not in dataframe: {missing_inputs}")
        if missing_targets:
            raise ValueError(f"Target columns not in dataframe: {missing_targets}")

        return input_columns, target_columns

    # ------------------------
    # Public API
    # ------------------------
    def fit(
        self,
        df: Union[pd.DataFrame, np.ndarray],
        input_columns: Optional[Iterable] = None,
        target_columns: Optional[Iterable] = None,
        test_prop: float = 0.2,
        masking_value: Union[float, int, None] = -1.0,
        random_seed: int = 42,
        upsampling_factor: int = 1,
    ) -> "TSModel":
        """Train the classifier as a denoiser on masked inputs, with automatic encoding of categorical columns.

        This method:
        * Detects and label-encodes string/categorical columns in both inputs and targets.
        * Ensures the masking value is part of each encoder's classes.
        * Optionally splits into training and holdout sets.
        * Optionally upsamples the training set.
        * Masks inputs randomly according to `mask_prob`.
        * Drops rows with missing targets before fitting.

        Args:
            df: Full dataset containing both input and target columns.
            input_columns: Columns to use as inputs. If None, inferred.
            target_columns: Columns to predict. If None, inferred.
            test_prop: Fraction of data to hold out for evaluation.
            masking_value: Value used to represent masked entries.
            random_seed: Random seed for reproducibility.
            upsampling_factor: Multiplier for training rows before masking.

        Returns:
            self: The fitted model.
        """
        if not (0.0 <= test_prop < 1.0):
            raise ValueError("test_prop must be in [0, 1).")

        self.masking_value = masking_value
        df = self._to_dataframe(df)
        self.input_columns, self.target_columns = self._resolve_columns(df, input_columns, target_columns)

        # Auto-encode categoricals
        self.encoders_ = {}
        df_encoded = df.copy()
        for col in self.input_columns + self.target_columns:
            if df_encoded[col].dtype == 'object' or isinstance(df_encoded[col].dtype, CategoricalDtype):
                le = LabelEncoder()
                series_str = df_encoded[col].astype(str)
                le.fit(series_str)

                # Ensure masking value string is in classes_
                mask_str = str(self.masking_value)
                if mask_str not in le.classes_:
                    le.classes_ = np.append(le.classes_, mask_str)

                df_encoded[col] = le.transform(series_str.where(series_str.isin(le.classes_), mask_str))
                self.encoders_[col] = le

        # Holdout split
        if test_prop > 0:
            df_train, df_holdout = train_test_split(
                df_encoded, test_size=test_prop, random_state=random_seed, shuffle=True
            )
        else:
            df_train = df_encoded
            df_holdout = pd.DataFrame(columns=df_encoded.columns)

        # Optional upsampling
        if upsampling_factor > 1:
            df_train = resample(
                df_train,
                replace=True,
                n_samples=len(df_train) * upsampling_factor,
                random_state=random_seed,
            )

        # Prepare masked inputs and targets
        X_train = df_train[self.input_columns]
        Y_train = df_train[self.target_columns]

        # Drop rows with NaN in targets
        mask_complete_targets = ~Y_train.isna().any(axis=1)
        n_dropped = (~mask_complete_targets).sum()
        if n_dropped > 0:
            warnings.warn(f"Dropped {n_dropped} training rows with NaN in target columns.")
        X_train = X_train.loc[mask_complete_targets]
        if len(self.target_columns) == 1:
            Y_train = Y_train[self.target_columns[0]].to_numpy().ravel()
        else:
            Y_train = Y_train.loc[mask_complete_targets]

        X_train_masked = self._random_masking(X_train, random_seed=random_seed)

        # Fit base model
        model = clone(self.base_model)
        self.fitted_model = model.fit(X_train_masked, Y_train)

        # Store holdout for evaluation
        self.df_holdout = df_holdout.copy()
        if not df_holdout.empty:
            self.df_holdout_masked = self._random_masking(df_holdout[self.input_columns], random_seed=random_seed)
        else:
            self.df_holdout_masked = pd.DataFrame(columns=self.input_columns)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Predict target columns from masked inputs, with overlap-aware imputation.

        This method generates predictions for the target columns based on the provided
        (already masked) input data. If a column is both an input and a target, any
        non-missing values in the input are preserved in the output; missing values
        are imputed using the model's predictions.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data containing at least the
                columns specified in `input_columns` during `fit()`. May also include
                overlapping target columns.

        Returns:
            pd.DataFrame: A DataFrame of predictions for the target columns. For
            overlapping columns, provided non-missing input values are retained;
            missing values are filled with model predictions.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If any required input columns are missing from `X`.

        Notes:
            - `X` must already be masked using the same `masking_value` provided to `fit()`.
            This method does not perform additional masking.
            - Categorical columns are automatically encoded using the encoders learned
            during `fit()`, with masking values and unseen categories handled gracefully.
        """
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        X_df = self._to_dataframe(X)
        missing_needed = [c for c in self.input_columns if c not in X_df.columns]
        if missing_needed:
            raise ValueError(f"Missing required input columns in X: {missing_needed}")

        # Encode categoricals using stored encoders, skip already-numeric
        for col, le in getattr(self, "encoders_", {}).items():
            if col in X_df.columns:
                if X_df[col].dtype == 'object' or isinstance(X_df[col].dtype, CategoricalDtype):
                    mask_missing = (
                        self._is_missing_series(X_df[col])
                        | (X_df[col] == self.masking_value)
                        | (X_df[col].astype(str) == str(self.masking_value))
                    )
                    if mask_missing.any():
                        to_encode = X_df.loc[~mask_missing, col].astype(str)
                    else:
                        to_encode = X_df[col].astype(str)

                    known_classes = set(le.classes_)
                    to_encode = to_encode.where(to_encode.isin(known_classes), other=str(self.masking_value))

                    X_df.loc[~mask_missing, col] = le.transform(to_encode)

        # Use inputs as provided (already masked by user) and predict
        X_used = X_df[self.input_columns].copy()
        Y_pred = self.fitted_model.predict(X_used)
        Y_pred = self._to_dataframe(Y_pred, columns=self.target_columns)
        
        # For overlapping columns: where input is not missing, prefer provided values
        overlap = [c for c in self.target_columns if c in self.input_columns]
        for c in overlap:
            provided = X_df[c] if c in X_df.columns else pd.Series([np.nan] * len(X_df), index=X_df.index)
            not_missing = ~self._is_missing_series(provided)
            mask_array = not_missing.to_numpy()  # positional mask
            if c in Y_pred.columns:
                Y_pred.loc[mask_array, c] = provided.to_numpy()[mask_array].astype(Y_pred[c].dtype, copy=False)

        return Y_pred[self.target_columns].copy()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[Union[str, int], np.ndarray]:
        """Return class probability distributions for each target variable.

        For multi-output classifiers, scikit-learn typically returns a list of arrays
        (one per output) from `predict_proba`. This method standardizes that output
        into a dictionary mapping each target column name to its corresponding
        probability array.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data containing at least the
                columns specified in `input_columns` during `fit()`.

        Returns:
            Dict[Union[str, int], np.ndarray]: A mapping from each target column name
            to a 2D NumPy array of shape `(n_samples, n_classes_for_target)`, where
            each row contains the predicted class probabilities for that sample.

        Raises:
            RuntimeError: If the model has not been fitted.
            AttributeError: If the fitted base model does not implement `predict_proba`.
            ValueError: If any required input columns are missing from `X`.
            RuntimeError: If the `predict_proba` output format is unexpected for a
                multi-output model.

        Notes:
            - `X` must already be masked using the same `masking_value` provided to `fit()`.
            - Categorical columns are automatically encoded using the encoders learned
            during `fit()`, with masking values and unseen categories handled gracefully.
            - For single-output models, the returned dictionary contains a single key
            corresponding to the target column.
        """
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if not hasattr(self.fitted_model, "predict_proba"):
            raise AttributeError("Base model does not support predict_proba.")

        X_df = self._to_dataframe(X)
        missing_needed = [c for c in self.input_columns if c not in X_df.columns]
        if missing_needed:
            raise ValueError(f"Missing required input columns in X: {missing_needed}")

        # Encode categoricals using stored encoders, skip masking values
        for col, le in getattr(self, "encoders_", {}).items():
            if col in X_df.columns:
                if X_df[col].dtype == 'object' or isinstance(X_df[col].dtype, CategoricalDtype):
                    mask_missing = (
                        self._is_missing_series(X_df[col])
                        | (X_df[col] == self.masking_value)
                        | (X_df[col].astype(str) == str(self.masking_value))
                    )
                    if mask_missing.any():
                        to_encode = X_df.loc[~mask_missing, col].astype(str)
                    else:
                        to_encode = X_df[col].astype(str)

                    known_classes = set(le.classes_)
                    to_encode = to_encode.where(to_encode.isin(known_classes), other=str(self.masking_value))

                    X_df.loc[~mask_missing, col] = le.transform(to_encode)

        X_used = X_df[self.input_columns].copy()
        probs = self.fitted_model.predict_proba(X_used)

        out: Dict[Union[str, int], np.ndarray] = {}
        if isinstance(probs, list) and len(probs) == len(self.target_columns):
            for col, arr in zip(self.target_columns, probs):
                out[col] = arr
        else:
            if len(self.target_columns) == 1:
                out[self.target_columns[0]] = probs  # type: ignore[assignment]
            else:
                raise RuntimeError("Unexpected predict_proba output format for multi-output model.")
        return out

    def decode_predictions(self, df_preds: pd.DataFrame) -> pd.DataFrame:
        """Decode numeric predictions back to original categorical labels.

        This method uses the label encoders stored during `fit()` to convert
        integer-encoded predictions into their original string or categorical
        representations for each encoded column.

        Args:
            df_preds (pd.DataFrame): DataFrame containing model predictions in
                numeric (encoded) form. Column names should match those used
                during training.

        Returns:
            pd.DataFrame: A copy of `df_preds` with encoded columns decoded back
            to their original labels.

        Notes:
            - Only columns present in both `df_preds` and `self.encoders_` are decoded.
            - Columns without a stored encoder are returned unchanged.
        """
        df_decoded = df_preds.copy()
        for col, le in self.encoders_.items():
            if col in df_decoded.columns:
                df_decoded[col] = le.inverse_transform(df_decoded[col].astype(int))
        return df_decoded

    def decode_predict_proba(self, probas: dict) -> dict:
        """Map `predict_proba` outputs back to original class labels.

        Converts the probability arrays returned by `predict_proba` into
        dictionaries keyed by the original class labels from the stored
        encoders. This makes the output human-readable and consistent with
        the original data.

        Args:
            probas (dict): Mapping from target column name to a 2D NumPy array
                of shape `(n_samples, n_classes_for_target)` containing class
                probabilities.

        Returns:
            dict: A mapping from each target column name to a dictionary where
            keys are original class labels and values are 1D NumPy arrays of
            probabilities for that class.

        Notes:
            - If the number of probability columns is fewer than the number of
            classes in the encoder (e.g., due to missing classes in training),
            the label list is trimmed to match the probability array width.
            - Columns without a stored encoder are returned with stringified
            integer indices as keys.
        """
        decoded = {}
        for col, arr in probas.items():
            if col in self.encoders_:
                le = self.encoders_[col]
                labels = list(le.classes_)
                n_cols = arr.shape[1]
                if len(labels) > n_cols:
                    labels = labels[:n_cols]
                decoded[col] = {label: arr[:, idx] for idx, label in enumerate(labels)}
            else:
                decoded[col] = {str(idx): arr[:, idx] for idx in range(arr.shape[1])}
        return decoded

    def feature_importances(self, normalized: bool = False) -> Dict[Union[str, int], float]:
        """Return feature importances aggregated across outputs if needed.

        If the fitted model exposes `feature_importances_`, those are returned.
        If it is a multi-output wrapper exposing per-output importances (e.g., via
        `estimators_`), importances are averaged across outputs.

        Args:
            normalized (bool): If True, min-max normalize the importances to [0, 1].

        Returns:
            Dict[Union[str, int], float]: Mapping input feature -> importance.

        Raises:
            RuntimeError: If model is not fitted.
            AttributeError: If no feature importance information can be extracted.
        """
        if self.fitted_model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        importances = None

        # Direct attribute
        if hasattr(self.fitted_model, "feature_importances_"):
            importances = np.asarray(self.fitted_model.feature_importances_)

        # Aggregate from per-output estimators (e.g., MultiOutputClassifier)
        elif hasattr(self.fitted_model, "estimators_"):
            vals = []
            for est in getattr(self.fitted_model, "estimators_"):
                if hasattr(est, "feature_importances_"):
                    vals.append(np.asarray(est.feature_importances_))
            if vals:
                importances = np.mean(np.vstack(vals), axis=0)

        if importances is None:
            raise AttributeError("Feature importances not available for the fitted model.")

        if normalized:
            min_v, max_v = float(np.min(importances)), float(np.max(importances))
            if max_v > min_v:
                importances = (importances - min_v) / (max_v - min_v)
            else:
                importances = np.zeros_like(importances, dtype=float)

        return dict(zip(self.input_columns, importances))  # type: ignore[arg-type]

    # ------------------------
    # Evaluation utilities
    # ------------------------
    def evaluate_holdout(
        self,
        average: str = "macro",
        zero_division: int = 0,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """Evaluate model performance on the internal holdout set.

        Computes accuracy, precision, recall, and F1-score for each target column
        in the holdout set, as well as macro-averaged metrics across all targets.

        Args:
            average (str, optional): Averaging method for precision, recall, and F1-score.
                Passed directly to scikit-learn's metric functions. Defaults to `"macro"`.
            zero_division (int, optional): Value to return when there is a zero division
                in precision or recall calculation. Passed directly to scikit-learn's
                metric functions. Defaults to `0`.

        Returns:
            Dict[str, Union[float, Dict[str, float]]]: A dictionary containing:
                - **accuracy** (float): Macro-averaged accuracy across all targets.
                - **precision** (float): Macro-averaged precision across all targets.
                - **recall** (float): Macro-averaged recall across all targets.
                - **f1** (float): Macro-averaged F1-score across all targets.
                - **per_target** (dict): Mapping from each target column name to its own
                metrics dictionary with keys `"accuracy"`, `"precision"`, `"recall"`,
                and `"f1"`.

        Raises:
            RuntimeError: If the model has not been fitted or if holdout data is not
                available (e.g., `fit()` was not called with `test_prop > 0`).

        Notes:
            - The holdout set is created during `fit()` when `test_prop > 0`.
            - Predictions are generated using the current fitted model and compared
            against the true labels in the holdout set.
        """
        if self.fitted_model is None or self.df_holdout is None or self.df_holdout_masked is None:
            raise RuntimeError("Holdout data not available. Ensure fit() was called with test_prop > 0.")

        Y_true = self._to_dataframe(self.df_holdout[self.target_columns], columns=self.target_columns)
        Y_pred = self.predict(self.df_holdout_masked)

        per_target: Dict[str, Dict[str, float]] = {}
        accs, precs, recs, f1s = [], [], [], []
        for col in self.target_columns:
            y_t = Y_true[col]
            y_p = Y_pred[col]
            accs.append(accuracy_score(y_t, y_p))
            precs.append(precision_score(y_t, y_p, average=average, zero_division=zero_division))
            recs.append(recall_score(y_t, y_p, average=average, zero_division=zero_division))
            f1s.append(f1_score(y_t, y_p, average=average, zero_division=zero_division))
            per_target[col] = {
                "accuracy": float(accs[-1]),
                "precision": float(precs[-1]),
                "recall": float(recs[-1]),
                "f1": float(f1s[-1]),
            }

        metrics = {
            "accuracy": float(np.mean(accs)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "f1": float(np.mean(f1s)),
            "per_target": per_target,
        }
        return metrics

    def cross_validate_kfold(
        self,
        df: pd.DataFrame,
        input_columns=None,
        target_columns=None,
        k: int = 5,
        masking_value=-1,
        random_seed: int = 42,
        upsampling_factor: int = 1,
    ):
        """Perform K-fold cross-validation with clean per-target metrics and masked-value handling."""
        df = self._to_dataframe(df)
        input_columns, target_columns = self._resolve_columns(df, input_columns, target_columns)

        kf = KFold(n_splits=k, shuffle=True, random_state=random_seed)
        results = []

        for train_idx, test_idx in kf.split(df):
            df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]

            cm = self.__class__(base_model=self.base_model, mask_prob=self.mask_prob)
            cm.fit(
                df_train,
                input_columns=input_columns,
                target_columns=target_columns,
                test_prop=0.0,
                masking_value=masking_value,
                random_seed=random_seed,
                upsampling_factor=upsampling_factor,
            )

            X_test_masked = cm._random_masking(df_test[input_columns], random_seed=random_seed)
            Y_pred = cm.predict(X_test_masked)

            fold_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

            for col_idx, col in enumerate(target_columns):
                y_true = df_test[col].to_numpy()
                y_pred = Y_pred[col].to_numpy()

                mask_str = str(masking_value)
                
                # Get the set of classes the model was trained on for this target
                if hasattr(cm.fitted_model, "estimators_"):
                    trained_classes = cm.fitted_model.estimators_[col_idx].classes_
                else:
                    trained_classes = cm.fitted_model.classes_
                
                # Robust flatten without NumPy trying to infer shape
                trained_classes_flat = []
                for cls in trained_classes:
                    if np.ndim(cls) == 0:
                        trained_classes_flat.append(cls)
                    else:
                        trained_classes_flat.extend(list(cls))

                trained_classes_set = set(trained_classes_flat)

                valid_mask = (
                    (y_pred != masking_value) & (y_pred.astype(str) != mask_str) &
                    (y_true != masking_value) & (y_true.astype(str) != mask_str) &
                    np.isin(y_true, list(trained_classes_set)) &
                    np.isin(y_pred, list(trained_classes_set))
                )

                if not np.any(valid_mask):
                    continue

                y_pred_valid = y_pred[valid_mask].astype(y_true.dtype, copy=False)
                y_true_valid = y_true[valid_mask]

                fold_metrics["accuracy"].append(accuracy_score(y_true_valid, y_pred_valid))
                fold_metrics["precision"].append(
                    precision_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
                )
                fold_metrics["recall"].append(
                    recall_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
                )
                fold_metrics["f1"].append(
                    f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
                )

            results.append({
                "accuracy": np.mean(fold_metrics["accuracy"]) if fold_metrics["accuracy"] else np.nan,
                "precision": np.mean(fold_metrics["precision"]) if fold_metrics["precision"] else np.nan,
                "recall": np.mean(fold_metrics["recall"]) if fold_metrics["recall"] else np.nan,
                "f1": np.mean(fold_metrics["f1"]) if fold_metrics["f1"] else np.nan,
            })

        return results


# ----------------------------------------------------------------------------
# 1.1 Shared Components: Consistent Training and Inference
# ----------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from xgboost import XGBClassifier

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):

    def __init__(self, cols_to_check=None):
        self.cols_to_check = cols_to_check

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Detect missing values before imputation
        if self.cols_to_check:
            for col in self.cols_to_check:
                if col not in X_out.columns:
                    raise ValueError(f"Missing required column: {col}")

                X_out[f"{col}_isnan"] = X_out[col].isna().astype(int)

        return X_out


class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()

        # Temperature difference feature
        if (
            "Process_temperature_K" in X_out.columns
            and "Air_temperature_K" in X_out.columns
        ):
            X_out["Temp_diff_K"] = (
                X_out["Process_temperature_K"]
                - X_out["Air_temperature_K"]
            )

        return X_out



class FoldWeightedXGBClassifier(XGBClassifier):
    """
    XGBoost classifier with fold-specific class weighting.
    """

    def fit(self, X, y, **kwargs):
        y_array = np.asarray(y)

        n_negative = np.sum(y_array == 0)
        n_positive = np.sum(y_array == 1)

        if n_positive == 0:
            raise ValueError(
                "Training fold contains no positive samples."
            )

        self.set_params(
            scale_pos_weight=float(n_negative / n_positive)
        )

        return super().fit(X, y, **kwargs)

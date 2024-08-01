import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


# Simple Example of a custom estimator
class TemplateEstimator(BaseEstimator, TransformerMixin):
    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        "Empty fit method"
        return self

    def transform(self, X):
        """Remove outliers from the dataset. Outliers are:
        All elements with index greater than 5
        """
        X_ = X.copy()
        X_ = X_.reset_index()
        col = X_.columns[0]
        X = X_[(X_[col] < 5)]
        assert X.shape[0] != 0, "No data left after removing outliers"
        return X.reset_index(drop=True)


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan
        )
        self.X_fit = None
        self.X_original = None
        self.X_encoded = None

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        self.X_fit = X.copy()
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        self.X_original = X[self.columns].copy()
        X_encoded = self.encoder.transform(X[self.columns])
        self.X_encoded = pd.DataFrame(X_encoded, columns=self.columns)
        return self.X_encoded


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.new_columns = None
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.X_fit = None
        self.X_original = None
        self.X_encoded = None

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        self.X_fit = X.copy()
        self.encoder.fit(self.X_fit[self.columns])
        return self

    def transform(self, X):
        self.X_original = X[self.columns].copy()
        X_encoded = self.encoder.transform(X[self.columns])
        self.X_encoded = self.feature_adjust(X_encoded)
        self.new_columns = self.X_encoded.columns
        return self.X_encoded

    def feature_adjust(self, X_encoded):
        result = pd.DataFrame()
        last_len = 0
        for i, features in enumerate(self.encoder.categories_):
            len_feature = len(features)
            formated_features = [
                f"{self.columns[i]}_{feat}".replace(" ", "_") for feat in features
            ]
            builded_features = pd.DataFrame(
                X_encoded[:, last_len : last_len + len_feature],
                columns=formated_features,
            )
            result = pd.concat([result, builded_features], axis=1)
            last_len += len_feature

        result.columns = result.columns.str.replace(
            r"[^\w\s]", "_", regex=True
        ).str.replace("__+", "_", regex=True)
        return result


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=MinMaxScaler):
        self.scaler = scaler()
        self.X_fit = None
        self.X_original = None
        self.X_encoded = None

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        self.X_fit = X.copy()
        self.scaler.fit(X)
        return self

    def transform(self, X):
        self.X_original = X.copy()
        X_encoded = self.scaler.transform(X)
        self.X_encoded = pd.DataFrame(X_encoded, columns=self.X_fit.columns)
        return self.X_encoded

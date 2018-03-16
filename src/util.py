import pandas as pd
import numpy as np
import string as s
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from src.dftransformers import (ColumnSelector, Identity,
                            FeatureUnion, MapFeature,
                            Intercept)
from basis_expansions import NaturalCubicSpline
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler

class XyScaler(BaseEstimator, TransformerMixin):
    """Standardize a training set of data along with a vector of targets.  """

    def __init__(self):
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()

    def fit(self, X, y, *args, **kwargs):
        """Fit the scaler to data and a target vector."""
        self.X_scaler.fit(X)
        self.y_scaler.fit(y.reshape(-1, 1))
        return self

    def transform(self, X, y, *args, **kwargs):
        """Transform a new set of data and target vector."""
        return (self.X_scaler.transform(X),
                self.y_scaler.transform(y.reshape(-1, 1)).flatten())

def catagorical_plot(ax, x, y):

    def convert_to_numeric(catagorical):
        classes = catagorical.unique()
        classes_mapping = {cls: i for i, cls in enumerate(classes)}
        classes_inv_mapping = {i: cls for i, cls in enumerate(classes)}
        classes_numeric = catagorical.apply(lambda cls: classes_mapping[cls])
        return classes_numeric, classes_inv_mapping

    numeric, classes_mapping = convert_to_numeric(x)

    noise = np.random.uniform(-0.3, 0.3, size=len(x))
    ax.scatter(numeric + noise, y, color="grey", alpha=0.5)
    box_data = list(y.groupby(x))
    ax.boxplot([data for _, data in box_data], positions=range(len(box_data)))
    ax.set_xticks(list(classes_mapping))
    ax.set_xticklabels(list(x.unique()), fontsize = 20)

def categorical_plot(df, ax, var_name,field_comp,title):
    catagorical_plot(ax, df[var_name], df[field_comp])
    ax.set_ylabel(title)
    ax.set_xlabel(var_name, fontweight='bold', fontsize=14)
    ax.set_title(title + " by {}".format(var_name), fontweight='bold', fontsize=20)

def simple_spline_specification(name, knots):
    select_name = "{}_select".format(name)
    spline_name = "{}_spline".format(name)
    return Pipeline([
        (select_name, ColumnSelector(name=name)),
        (spline_name, NaturalCubicSpline(knots=knots))
    ])

def residual_plot(ax, x, y, y_hat, n_bins=50):
    residuals = y - y_hat
    ax.axhline(0, color="black", linestyle="--")
    ax.scatter(x, residuals, color="grey", alpha=0.5)
    ax.set_ylabel("Residuals ($y - \hat y$)")

def rss(y, y_hat):
    return np.mean((y  - y_hat)**2)

def rsq(y, y_hat):
    ss_tot = rss(y, np.mean(y))
    ss_res = rss(y, y_hat)
    return 1 - (ss_res / ss_tot)

def bootstrap_rsq(X, y, pipeline, n_boot=10000):
    rsqs = []
    for _ in range(n_boot):
        X_boot, y_boot = resample(X, y)
        X_transform = pipeline.transform(X_boot)
        model = LinearRegression(fit_intercept=False)
        model.fit(X_transform.values, y_boot)
        y_boot_hat = model.predict(X_transform.values)
        rsqs.append(rsq(y_boot, y_boot_hat))
    return rsqs

def rmse(true, predicted):

    sum_list = []
    for idx, each in enumerate(predicted):
        sum_list.append((predicted.iloc[idx] - true.iloc[idx])**2)
    sum_mean = np.mean(sum_list)
    rmse = np.sqrt(sum_mean)

    return rmse

def cv(X, y, base_estimator, n_folds, random_seed=154):
    kf = KFold(n_splits=n_folds, random_state=random_seed)
    test_cv_errors, train_cv_errors = np.empty(n_folds), np.empty(n_folds)
    for idx, (train, test) in enumerate(kf.split(X)):
        # Split into train and test
        X_cv_train, y_cv_train = X[train], y[train]
        X_cv_test, y_cv_test = X[test], y[test]
        # Standardize data.
        standardizer = XyScaler()
        standardizer.fit(X_cv_train, y_cv_train)
        X_cv_train_std, y_cv_train_std = standardizer.transform(X_cv_train, y_cv_train)
        X_cv_test_std, y_cv_test_std = standardizer.transform(X_cv_test, y_cv_test)
        # Fit estimator
        estimator = clone(base_estimator)
        estimator.fit(X_cv_train_std, y_cv_train_std)
        # Measure performance
        y_hat_train = estimator.predict(X_cv_train_std)
        y_hat_test = estimator.predict(X_cv_test_std)
        # Calclate the error metrics
        train_cv_errors[idx] = rss(y_cv_train_std, y_hat_train)
        test_cv_errors[idx] = rss(y_cv_test_std, y_hat_test)
    return train_cv_errors, test_cv_errors

def get_optimal_alpha(mean_cv_errors_test):
    alphas = mean_cv_errors_test.index
    optimal_idx = np.argmin(mean_cv_errors_test.values)
    optimal_alpha = alphas[optimal_idx]
    return optimal_alpha

def train_at_various_alphas(X, y, model, alphas, n_folds=10, **kwargs):
    cv_errors_train = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                     columns=alphas)
    cv_errors_test = pd.DataFrame(np.empty(shape=(n_folds, len(alphas))),
                                        columns=alphas)
    for alpha in alphas:
        train_fold_errors, test_fold_errors = cv(X, y, model(alpha=alpha, **kwargs), n_folds=n_folds)
        cv_errors_train.loc[:, alpha] = train_fold_errors
        cv_errors_test.loc[:, alpha] = test_fold_errors
    return cv_errors_train, cv_errors_test

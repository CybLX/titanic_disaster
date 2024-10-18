from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer,KNNImputer
import numpy as np

#https://scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html#sphx-glr-auto-examples-impute-plot-missing-values-py

def get_scores_for_imputer(regressor,imputer, X_missing, y_missing, N_SPLITS):
    estimator = make_pipeline(imputer, regressor)
    impute_scores = cross_val_score(
        estimator, X_missing, y_missing, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return impute_scores

def get_full_score(regressor,X_full, y_full, N_SPLITS):
    full_scores = cross_val_score(
        regressor, X_full, y_full, scoring="neg_mean_squared_error", cv=N_SPLITS
    )
    return full_scores.mean(), full_scores.std()

def get_impute_iterative(regressor,X_missing, y_missing,N_SPLITS):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=3,
        max_iter=1,
        sample_posterior=True,
    )
    iterative_impute_scores = get_scores_for_imputer(regressor,imputer, X_missing, y_missing,N_SPLITS)
    return iterative_impute_scores.mean(), iterative_impute_scores.std()

def get_impute_zero_score(regressor,X_missing, y_missing,N_SPLITS):
    imputer = SimpleImputer(
        missing_values=np.nan, add_indicator=True, strategy="constant", fill_value=0
    )
    zero_impute_scores = get_scores_for_imputer(regressor,imputer, X_missing, y_missing,N_SPLITS)
    return zero_impute_scores.mean(), zero_impute_scores.std()

def get_impute_mean(regressor,X_missing, y_missing,N_SPLITS):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean", add_indicator=True)
    mean_impute_scores = get_scores_for_imputer(regressor,imputer, X_missing, y_missing,N_SPLITS)
    return mean_impute_scores.mean(), mean_impute_scores.std()

def get_impute_KNN_score(regressor,X_missing, y_missing,N_SPLITS):
    imputer = KNNImputer(missing_values=np.nan, add_indicator=True)
    KNN_impute_scores = get_scores_for_imputer(regressor,imputer, X_missing, y_missing,N_SPLITS)
    return KNN_impute_scores.mean(), KNN_impute_scores.std()

def get_impute_iterative(regressor,X_missing, y_missing,N_SPLITS):
    imputer = IterativeImputer(
        missing_values=np.nan,
        add_indicator=True,
        random_state=0,
        n_nearest_features=3,
        max_iter=1,
        sample_posterior=True,
    )
    iterative_impute_scores = get_scores_for_imputer(regressor,imputer, X_missing, y_missing,N_SPLITS)
    return iterative_impute_scores.mean(), iterative_impute_scores.std()

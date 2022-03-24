import os
import pickle

import numpy as np

from kndetect.predict_features import get_feature_names
from kndetect.utils import get_data_dir_path


def load_classifier(classifier_name):
    """Function to load trained classifier

    Parameters
    ----------
    classifier_name: str
        name of classifier to be loaded from the data folder
        options are: ['complete.pkl', 'partial.pkl']

    Returns
    -------
    clf: sklearn.ensemble.RandomForestClassifier
        trained classifier that is loaded from the data folder
    """
    if classifier_name not in ["complete.pkl", "partial.pkl"]:
        raise ValueError("Such a model does not exist")

    data_dir = get_data_dir_path()
    fn = os.path.join(data_dir, classifier_name)

    clf = pickle.load(open(fn, "rb"))

    return clf


def filter_no_coeff_events(features_df):
    """Function to filter out events that have no predictions in both `g` and `r` bands

    Parameters
    ----------
    features_df: pd.DataFrame
        dataframe with the optimised set of coefficients and features.
        this dataframe must contain "coeff1_g" and "coeff1_r" columns

    Returns
    _______
    filtered_indices: list
        list with boolean values, that are True for events where coefficients are predicted
        i.e., they are non-zero
    """

    if not {"coeff1_g", "coeff1_r"}.issubset(features_df.columns):
        raise ValueError("The features column names are incorrect")
    zeros = np.logical_and(
        features_df["coeff1_g"].values == 0, features_df["coeff1_r"].values == 0
    )
    filtered_indices = ~zeros
    return filtered_indices


def predict_kn_score(clf, features_df):
    """Function to predict kn_scores

    Parameters
    ----------
    clf: sklearn.ensemble.RandomForestClassifier
        trained classifier to be used for classifying events
    features_df: pd.DataFrame
        A dataframe containing at least the columns mentioned
        in kndetect.predict_features.get_feature_names.
        Etra columns are ignored while making a classfication.

    Returns
    -------
    probabilities_: np.array
        array with index [0] corresponding to events being non-kne
        and index [1] corresponding to events being kn
    filtered_indices: list
        List with bool values as True for events with non-zero predicted coefficients.
        This list includes the events that are actually classified by the classifier.
    """
    feature_col_names = get_feature_names()
    filtered_indices = filter_no_coeff_events(features_df)
    # If all alerts are flagged as bad

    if len(features_df[filtered_indices]) == (0, len(feature_col_names)):
        probabilities_ = np.zeros((len(features_df), 2), dtype=float)
        filtered_indices = [False] * len(features_df)
        return probabilities_, filtered_indices

    probabilities = clf.predict_proba(
        features_df[filtered_indices][feature_col_names].values
    )
    probabilities_notkne = np.zeros(len(features_df))
    probabilities_kne = np.zeros(len(features_df))

    probabilities_notkne[filtered_indices] = probabilities.T[0]
    probabilities_kne[filtered_indices] = probabilities.T[1]
    probabilities_ = np.array([probabilities_notkne, probabilities_kne]).T

    return probabilities_, filtered_indices

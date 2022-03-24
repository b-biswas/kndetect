import numpy as np
from sklearn.ensemble import RandomForestClassifier

from kndetect.features import get_feature_names
from kndetect.utils import get_event_type


def get_y_true_values(event_types, prediction_type_nos):
    """function to return the y_true values

    event_types: list
        event types for which the y_true value is to be obtained
    prediction_type_nos: list
        list of event type nos that are to be predicted as y_true.

    Returns
    -------
    y_true: np.array
        array with the y_true values
    """
    assert isinstance(event_types, list)
    assert isinstance(prediction_type_nos, list)

    y_true = np.isin(np.array(event_types), prediction_type_nos)

    return y_true


def append_y_true_col(
    features_df, prediction_type_nos, meta_df, meta_key_col_name, meta_type_col_name
):
    """Function to add column with y_true values to features_df

    Parameters
    ----------
    features_df: pd.DataFrame
        dataframe with the optimised set of coefficients and features.
        this dataframe must contain all the columns that are used as features and a `y_true` column
    prediction_type_nos: list
        list of event type nos that are to be predicted as y_true.
    meta_key_col_name: str
        column name against which keys are to be matched
    meta_type_col_name: str
        column name for event type, in df_meta
    fetch_type_name: bool
        To idtentify id the event type name corresponding to event types are to be returned.

    Returns
    -------
    features_df: pd.DataFrame
        returns the features_df with the following columns added:
        `type`, `type_names` and `y_ture`
    """
    event_types, event_type_names = get_event_type(
        list(features_df["key"].values),
        meta_df=meta_df,
        meta_key_col_name=meta_key_col_name,
        meta_type_col_name=meta_type_col_name,
        fetch_type_name=True,
    )
    features_df["type"] = event_types
    features_df["type_names"] = event_type_names
    features_df["y_true"] = get_y_true_values(
        event_types=event_types, prediction_type_nos=prediction_type_nos
    )

    return features_df


def train_classifier(features_df):
    feature_names = get_feature_names()
    assert set(feature_names + ["y_true"]).issubset(features_df.columns)

    clf = RandomForestClassifier(n_estimators=30, max_depth=13)

    features = features_df[feature_names]
    clf.fit(features.values, features_df["y_true"].values)

    y_pred = clf.predict(features)
    y_score = clf.predict_proba(features)

    features_df["y_pred"] = y_pred
    features_df["y_score"] = y_score[:, 1]

    return clf, features_df

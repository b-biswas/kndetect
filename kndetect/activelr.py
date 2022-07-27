import os

import actsnclass
import numpy as np
import pandas as pd
from actsnclass import DataBase

from kndetect.features import get_feature_names


# this was taken from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/database.py
def build_samples(
    features: pd.DataFrame, initial_training: int, frac_Ia=0.5, screen=False
):
    """Build initial samples for Active Learning loop.

    Parameters
    ----------
    features: pd.DataFrame
        Complete feature matrix. Columns are: ['id', 'type',
        'g_pc_1',  'g_pc_2', 'g_pc_3', 'g_residual', 'g_maxflux',
         'r_pc_1', 'r_pc_2', 'r_pc_3', 'r_residual', 'r_maxflux']

    initial_training: int
        Number of objects in the training sample.
    frac_Ia: float (optional)
        Fraction of Ia in training. Default is 0.5.
    screen: bool (optional)
        If True, print intermediary information to screen.
        Default is False.


    Returns
    -------
    actsnclass.DataBase
        DataBase for active learning loop
    """
    data = DataBase()

    # initialize the temporary label holder
    train_indexes = np.random.choice(
        np.arange(0, features.shape[0]), size=initial_training, replace=False
    )

    Ia_flag = features["type"] == 1
    Ia_indx = np.arange(0, features.shape[0])[Ia_flag]
    nonIa_indx = np.arange(0, features.shape[0])[~Ia_flag]

    indx_Ia_choice = np.random.choice(
        Ia_indx, size=max(1, initial_training // 2), replace=False
    )
    indx_nonIa_choice = np.random.choice(
        nonIa_indx, size=initial_training - max(1, initial_training // 2), replace=False
    )
    train_indexes = list(indx_Ia_choice) + list(indx_nonIa_choice)

    temp_labels = features["type"][np.array(train_indexes)]

    if screen:
        print("\n temp_labels = ", temp_labels, "\n")

    # set training
    train_flag = np.array([item in train_indexes for item in range(features.shape[0])])

    train_Ia_flag = features["type"][train_flag] == 1
    data.train_labels = train_Ia_flag.astype(int)
    data.train_features = features[train_flag].values[:, 2:]
    data.train_metadata = features[["id", "type"]][train_flag]

    # set test set as all objs apart from those in training
    test_indexes = np.array(
        [i for i in range(features.shape[0]) if i not in train_indexes]
    )
    test_ia_flag = features["type"][test_indexes].values == 1
    data.test_labels = test_ia_flag.astype(int)
    data.test_features = features[~train_flag].values[:, 2:]
    data.test_metadata = features[["id", "type"]][~train_flag]

    # set metadata names
    data.metadata_names = ["id", "type"]

    # set everyone to queryable
    data.queryable_ids = data.test_metadata["id"].values

    if screen:
        print("Training set size: ", data.train_metadata.shape[0])
        print("Test set size: ", data.test_metadata.shape[0])
        print("  from which queryable: ", len(data.queryable_ids))

    return data


# This was slightly modified from https://github.com/COINtoolbox/ActSNClass/blob/master/actsnclass/learn_loop.py
def learn_loop(
    data: actsnclass.DataBase,
    nloops: int,
    strategy: str,
    output_metrics_file: str,
    output_queried_file: str,
    classifier="RandomForest",
    batch=1,
    screen=True,
    output_prob_root=None,
    seed=42,
    nest=1000,
    max_depth=None,
):
    """Perform the active learning loop. All results are saved to file.

    Parameters
    ----------
    data: actsnclass.DataBase
        Output from the build_samples function.
    nloops: int
        Number of active learning loops to run.
    strategy: str
        Query strategy. Options are 'UncSampling' and 'RandomSampling'.
    output_metrics_file: str
        Full path to output file to store metric values of each loop.
    output_queried_file: str
        Full path to output file to store the queried sample.
    classifier: str (optional)
        Machine Learning algorithm.
        Currently only 'RandomForest' is implemented.
    batch: int (optional)
        Size of batch to be queried in each loop. Default is 1.
    n_est: int (optional)
        Number of trees. Default is 1000.
    output_prob_root: str or None (optional)
        If str, root to file name where probabilities without extension!
        Default is None.
    screen: bool (optional)
        If True, print on screen number of light curves processed.
    seed: int (optional)
        Random seed.
    max_depth: None or int (optional)
        The maximum depth of the tree. Default is None.
    """

    for loop in range(nloops):

        if screen:
            print("Processing... ", loop)

        # classify
        data.classify(method=classifier, seed=seed, n_est=nest)

        if isinstance(output_prob_root, str):
            data_temp = data.test_metadata.copy(deep=True)
            data_temp["prob_Ia"] = data.classprob[:, 1]
            data_temp.to_csv(
                output_prob_root + "_loop_" + str(loop) + ".csv", index=False
            )

        # calculate metrics
        data.evaluate_classification(screen=screen)

        # choose object to query
        indx = data.make_query(strategy=strategy, batch=batch, seed=seed, screen=screen)
        print("indx: ", indx)

        # update training and test samples
        data.update_samples(indx, loop=loop)

        # save metrics for current state
        data.save_metrics(
            loop=loop, output_metrics_file=output_metrics_file, batch=batch, epoch=loop
        )

        # save query sample to file
        data.save_queried_sample(output_queried_file, loop=loop, full_sample=False)


def run_AL_loops(
    features_df,
    save_data,
    initial_training=10,
    strategy="UncSampling",
    nloops=1500,
    screen=True,
    data_base_path=None,
    max_depth=42,
    n_estimators=30,
):

    features_df = features_df.rename(columns={"key": "id", "y_true": "type"})
    if data_base_path is None:
        data_base_path = "results/"

    data_base_path = os.path.join(data_base_path, "AL")

    output_prob_root = None

    output_metrics_file = os.path.join(data_base_path, "metrics.dat")
    output_queried_file = os.path.join(data_base_path, "queries.dat")

    fname_ini_train = os.path.join(data_base_path, "initial_train.csv")
    fname_fulltrain = os.path.join(data_base_path, "train.csv")

    # build database
    database = build_samples(
        features_df, initial_training=initial_training, screen=screen
    )
    database.features_names = get_feature_names()

    train = pd.DataFrame(database.train_features, columns=database.features_names)
    train["key"] = database.train_metadata["id"].values
    train["y_true"] = database.train_metadata["type"].values
    if save_data:
        train.to_csv(fname_ini_train, index=False)

    # perform learning loop
    learn_loop(
        database,
        nloops=nloops,
        strategy=strategy,
        output_metrics_file=output_metrics_file,
        output_queried_file=output_queried_file,
        classifier="RandomForest",
        seed=None,
        batch=1,
        screen=True,
        output_prob_root=output_prob_root,
        nest=n_estimators,
        max_depth=max_depth,
    )

    # save final training
    full_train = pd.DataFrame(database.train_features, columns=database.features_names)
    full_train["key"] = database.train_metadata["id"]
    full_train["y_true"] = database.train_metadata["type"]

    if save_data:
        full_train.to_csv(fname_fulltrain, index=False)

    return full_train

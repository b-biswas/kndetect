import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def get_ids_of_eqally_spaced_objects(data_ob, object_ids=None):
    """
    returns a list of ids among the object_ids provided that has readings every 2 days.

    Parameters
    ----------
    data_ob: Data.Data
        Data object with the light curve data
    object_ids: list
        list of ids among which search is to be performed. if left as none, all ids in the data_ob is considered.

    Returns
    -------
    equally_spaced_ids: list
        list of ids with data points at an interval of 2 days.
    """
    equally_spaced_ids = []

    if object_ids is None:
        object_ids = data_ob.get_all_object_ids()

    for object_id in object_ids:
        mjds = data_ob.get_data_of_event(object_id)["MJD"]
        min_date = np.amin(mjds)
        max_date = np.amax(mjds)
        tot_num_days = (max_date - min_date) / 2 + 1

        if (len(mjds) / 6) == tot_num_days:
            equally_spaced_ids.append(object_id)

    return equally_spaced_ids


def get_event_distirb(data_ob, object_ids=None):
    """
    returns a dictionary showing the distribution of objects among various types.

    Parameters
    ----------
    data_ob: Data.Data
        Data object with the light curve data
    object_ids: list of ids among which distribution is to be searched for. if left as none, all ids in the data_ob is considered.

    Returns
    -------
    event_distrib: dict
        dict that holds the number of objects corresponding to each type.
    """

    event_distrib = {}

    if object_ids is None:
        object_ids = data_ob.get_all_object_ids()

    for object_id in object_ids:
        type_number = data_ob.get_object_type_number(object_id)
        if type_number not in event_distrib:
            event_distrib[type_number] = 0
        event_distrib[type_number] = event_distrib[type_number] + 1

    return event_distrib


def get_ids_for_target_distrib(data_ob, target_distrib, object_ids=None):
    """
    randomly samples events to match the target dsitribution of events.
    Number of events from each type is the minimum of corresponding target value
    and number events for that type in the original dataset.

    Parameters
    ----------
    data_ob: Data.Data
        Data object with the light curve data
    target_distrib: dict
        target distribution that is to be obtained.
    object_ids:
        object ids from which target distribution is to be drawn.
        if left as none, is samples from all the events in the dataset

    Returns
    -------
    new_ids: list
        A list of object ids that obeys rhe target distribution.
    """
    if object_ids is None:
        object_ids = data_ob.get_all_object_ids()

    np.random.seed(13)
    object_ids = np.random.permutation(object_ids)
    counter_distrib = {}

    new_ids = []
    for object_id in object_ids:
        current_event_type = data_ob.get_object_type_number(object_id)
        if current_event_type not in counter_distrib:
            counter_distrib[current_event_type] = 0
        if counter_distrib[current_event_type] < target_distrib[current_event_type]:
            new_ids.append(object_id)
            counter_distrib[current_event_type] = (
                counter_distrib[current_event_type] + 1
            )

    return new_ids


def stack_training_data(
    data_ob,
    object_ids,
    number_of_days_in_lc=100,
    time_step=2,
    num_days_tolerance=2,
    plot_results=False,
):
    """
    function to generate bandwise stacked dataset for generating PCs.

    Parameters
    ----------
    data_ob: Data.Data
        data of the light curves
    object_ids: list
        object ids to be used for the stacking
    number_of_days_in_lc: int
        number of days for which light curves are to be predicted
    time_step: float
        frequency of readings in days
    num_days_tolerance: int
        number of days of variation allowed in the mid point determination.
        For example, if num_days_tolerance is 2 the midpoint will lie between day 48 and 52.
        Note that this must be a numtiple to 2 because of the current convention of 1 points in every 2 days.
    plot_results: bool
        plots only the u band

    Returns
    -------
    final_array: np.ndarray
        array of segments with 100 days of data
    """
    final_array = {}
    for band in data_ob.bands:
        final_array[band] = np.zeros((len(object_ids), 51))

    np.random.seed(23)

    for i, object_id in enumerate(object_ids):

        event_df = data_ob.get_data_of_event(object_id)

        for band in data_ob.bands:

            band_index = event_df[data_ob.band_col_name] == band
            band_df = event_df[band_index]

            if len(band_df) > 0:
                loc = np.argmax(band_df[data_ob.brightness_col_name])
                rand_int = 0
                if num_days_tolerance != 0:
                    rand_int = np.random.randint(0, num_days_tolerance + 1) - int(
                        num_days_tolerance / 2
                    )
                while ((loc + rand_int) < 0) | ((loc + rand_int) >= len(band_df)):
                    rand_int = np.random.randint(num_days_tolerance + 1) - int(
                        num_days_tolerance / 2
                    )
                mid_point = band_df[data_ob.time_col_name][loc + rand_int]
                start_date = mid_point - number_of_days_in_lc / 2
                end_date = mid_point + number_of_days_in_lc / 2
                start_index = band_df["MJD"] >= start_date
                end_index = band_df["MJD"] <= end_date
                band_df = band_df[start_index * end_index]

                if len(band_df) > 0:
                    loc = np.argmax(band_df[data_ob.brightness_col_name])

                    shifted_flux = band_df["FLUXCAL"]

                    flux_data = np.zeros(51)

                    flux_data[
                        25 - loc - rand_int : 25 - loc - rand_int + len(band_df["MJD"])
                    ] = shifted_flux

                    final_array[band][i] = flux_data

                    if band == "r" and plot_results:
                        fig = plt.figure(figsize=(12, 6))
                        ax = fig.add_subplot(1, 1, 1)

                        ax.scatter(
                            np.arange(0, 102, 2) - 50, flux_data, color="C4", label=band
                        )

                        type_name = data_ob.df_metadata["type"][
                            data_ob.df_metadata[data_ob.object_id_col_name] == object_id
                        ][0]

                        ax = fig.add_subplot(1, 1, 1)
                        plt.gca().annotate(
                            "Type: " + type_name,
                            xy=(0.12, 0.83),
                            xycoords="figure fraction",
                            fontsize=20,
                        )
                        ax.axvline(x=0, ymin=0, ymax=1, label="Day 0", linestyle="--")
                        plt.xlabel("Days since maximum", fontsize=25)
                        plt.ylabel("FLUXCAL", fontsize=25)
                        plt.rc("xtick", labelsize=17)
                        plt.rc("ytick", labelsize=17)
                        plt.rc("legend", fontsize=15)
                        ax.legend()
                        plt.tight_layout()
                        fig.savefig(
                            "results/Perfect_plots/" + str(int(object_id)) + "_uband"
                        )
                        plt.show()
                        plt.close("all")

        final_array["object_ids"] = object_ids

    return final_array


# function to generate PCs


def gen_components(train_data, bands=["g", "r"], bands_combined=True):
    """
    function to geenrate the PCs

    Parameters
    ----------
    train_data: np.ndarray
        stacked data used to generate the PCs
    bands: list
        The bands which are taken into account for generating PCs.
        if bands_combined is True, then the bands are stacked together to generate
        the PCs and the key for the dict is set to 'all'.
    bands_combined: bool
        consider data of all bands together/individually.
        Defult behavior is to combine the data from different bands before taking PCA
    """
    PC_dict = {}
    PC_var_ratio = {}
    clf = PCA(n_components=10)
    if bands_combined:
        fit_arr = None
        for pb_name in bands:
            if fit_arr is None:
                fit_arr = train_data[pb_name]
            else:
                fit_arr = np.concatenate((fit_arr, train_data[pb_name]))
        print(np.shape(fit_arr))

        clf.fit(fit_arr)
        PC_var_ratio["all"] = clf.explained_variance_ratio_
        PC_dict["all"] = clf.components_

    else:
        for pb_name in bands:
            fit_arr = train_data[pb_name]
            clf.fit(fit_arr)
            PC_var_ratio[pb_name] = clf.explained_variance_ratio_
            PC_dict[pb_name] = clf.components_
            # print(pca.explained_variance_ratio_)
            print(PC_dict[pb_name].shape)
    return PC_dict, PC_var_ratio

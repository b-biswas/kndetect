import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn import metrics

from kndetect.features import calc_prediction
from kndetect.utils import extract_mimic_alerts_region, snana_ob_type_name


def plot_light_curve(
    color_band_dict,
    lc,
    bands,
    band_map=None,
    fig=None,
    band=None,
    start_date=None,
    end_date=None,
    plot_points=True,
    mark_label=True,
    label_postfix="",
    clip_xlims=None,
    markers={},
    markerfacecolor=None,
    alpha=1.0,
    min_points_for_plot=1,
):
    """
    plots either only one band of a light curve or all the bands

    Parameters
    ----------
    color_band_dict: dict
        mapping from band/filter name to color with which it is to be drawn
    band_map: dict
        map for the names of filters to be displayed on the plot eg. {0:u, 1:g, 2:r, 3:i, 4:z, 5:y}
    fig: matplotlib.figure
        fig on which plot is to be made. New figure is created if nothing is passed
    band: list
        band for which light curve is to be drawn (else plots are made for all the bands)
    start_date: int/float
        start of plot region
    end_date: int/float
        end of plot region
    plot_points: bool
        mark the recorded datapoints on the curve
    mark_label: bool
        to put label or not
    mark_maximum: bool
        if True, marks the point with highest flux reading for each band
    label_postfix: str
        post fix on label after band name
    clip_xlims: bool
        plots only the region of prediction if set to true
    markers: dict
        dictionary of markers to be used for the plotting dta from different bands
    markerfacecolor: str
    either None/"none". When "none" the markers are not filled but if left as None the
        markers are filled with solid color
    alpha: float
        alpha value of the lines/points that are to be plotted
    min_points_for_plot: int
        minimum number of points that must be present in the bands to be plotted

    Returns
    -------
    fig: matplotlib.figure
        Figure with the plots
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.gca()

    if start_date is None:
        start_date = np.amin(lc["MJD"])
    if end_date is None:
        end_date = np.amax(lc["MJD"])

    data_points_found = 0

    for band in bands:

        pb_name = band
        if band_map is not None:
            pb_name = band_map[band]

        band_index = lc["FLT"] == band
        start_index = lc["MJD"] >= start_date
        end_index = lc["MJD"] <= end_date

        index = band_index & start_index & end_index

        # print(sum(index))
        if sum(index) > 0:
            data_points_found = 1
            df_plot_data = lc[index]

            if plot_points:
                ax.errorbar(
                    df_plot_data["MJD"],
                    df_plot_data["FLUXCAL"],
                    df_plot_data["FLUXCALERR"],
                    color=color_band_dict[band],
                    markersize=8,
                    markerfacecolor=markerfacecolor,
                    label=pb_name + " " + label_postfix if mark_label else "",
                    fmt=".",
                    marker=markers[band] if markers else "o",
                    alpha=alpha,
                )
            else:
                ax.errorbar(
                    df_plot_data["MJD"],
                    df_plot_data["FLUXCAL"],
                    df_plot_data["FLUXCALERR"],
                    markersize=8,
                    markerfacecolor=markerfacecolor,
                    color=color_band_dict[band],
                    marker=markers[band] if markers else "o",
                    label=pb_name + " " + label_postfix if mark_label else "",
                    alpha=alpha,
                )

    if data_points_found == 0:
        print("There are no data points in the given date range")

    # ax.plot([start_date, end_date], [0, 0], label='y=0')
    if clip_xlims is not None:
        ax.set_xlim([start_date, end_date])

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("MJD", fontsize=30)
    plt.ylabel("FLUXCAL", fontsize=30)
    plt.legend(prop={"size": 35})
    plt.legend()
    # fig.close()

    return fig


def plot_predicted_bands(
    lc,
    all_band_coeff_dict,
    features,
    color_band_dict,
    flux_lim,
    pcs,
    title=None,
    object_name=None,
    axes_lims=True,
    buffer_days=30,
    num_prediction_points=401,
    mark_threshold=True,
    linestyle="solid",
    fig=None,
    duration=None,
    band_map=None,
    plot_unused_points=True,
    legend_title="",
):
    """
    plot of predictions of each band

    Parameters
    ----------
    all_band_coeff_dict:
        dictionary with coefficients corresponding to each band
    color_band_dict:
        dict with colors corresponding to each band
    title: str
        Title of the legend
    mark_maximum: bool
        option to mark maximum flux recorded in each band
    object_name: str
        name/object type of the current object
    axes_lims: bool
        boolean to limit the axes on in the region of prediction
    buffer_days: int
        buffer region beyond the prediction where plot is made. So if prediction in between the days
        120 - 220 and buffer_days is 5, plot is limited to days 115 - 225.
    mark_threshold: bool
        Mark the minimum threshold below which no prediction is made for a band
    linestyle:
        linetyle of the plot
    fig:
        fig on which plot is to be made. If nothing is passed, new fig is created.

    Returns
    -------
    fig:
        figure with the plots
    """
    bands = all_band_coeff_dict.keys()

    if plot_unused_points:
        fig = plot_light_curve(
            lc=lc,
            bands=bands,
            fig=fig,
            color_band_dict=color_band_dict,
            alpha=0.3,
            mark_label=True,
            plot_points=True,
            label_postfix="light curve",
            band_map=band_map,
        )

    prediction_start_date = np.inf
    prediction_end_date = -np.inf

    current_date = None
    if "current_dates" in features.keys():
        current_date = features["current_dates"]
        lc, _ = extract_mimic_alerts_region(
            lc=lc,
            current_date=features["current_dates"],
            flux_lim=flux_lim,
            duration=duration,
        )

    prediction_made = False
    for band in bands:
        band_df = lc[lc["FLT"] == band]

        if all_band_coeff_dict[band][0] != 0:
            max_loc = np.argmax(band_df["FLUXCAL"])
            mid_point_date = band_df["MJD"].iloc[max_loc]

            end_date = mid_point_date + 50
            start_date = mid_point_date - 50

            fig = plot_light_curve(
                lc=band_df,
                bands=[band],
                color_band_dict=color_band_dict,
                fig=fig,
                start_date=start_date,
                end_date=end_date,
                band=band,
                alpha=1,
                plot_points=True,
                label_postfix=" prediction region",
                band_map=band_map,
            )

            predicted_lc = calc_prediction(all_band_coeff_dict[band], pcs) * np.amax(
                band_df["FLUXCAL"].values
            )
            time_data = (
                np.linspace(-50, 50, num=num_prediction_points, endpoint=True)
                + mid_point_date
            )

            prediction_start_date = min(prediction_start_date, start_date)
            prediction_end_date = max(prediction_end_date, end_date)

            prediction_made = True

        else:
            predicted_lc = []
            time_data = []

        plt.plot(
            time_data,
            predicted_lc,
            color=color_band_dict[band],
            linestyle=linestyle,
            label=band_map[band] + " prediction",
        )

    if prediction_made and axes_lims:
        if prediction_start_date is not np.inf:
            plt.xlim(left=prediction_start_date - buffer_days)
        if prediction_end_date != -np.inf:
            plt.xlim(right=prediction_end_date + buffer_days)

    xmin, xmax, ymin, ymax = plt.axis()
    if prediction_made and ("current_dates" in features.keys()):
        plt.plot(
            [current_date, current_date],
            [ymin / 2, ymax / 2],
            color="darkorange",
            ls="dashed",
            label="alert end",
        )

    ax = plt.gca()
    if mark_threshold:
        ax.axhline(
            y=flux_lim, color="#3C8DFF", linestyle="--", label="min amplitude threshold"
        )
    # plt.text(.01, .94, "ID: " + str(self.lc.object_id), fontsize=15, transform=ax.transAxes)
    # if object_name is not None:
    #    plt.text(.01, .88, "Type: " + object_name, fontsize=15, transform=ax.transAxes)

    ax.tick_params(axis="both", which="major", labelsize=12)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel("MJD", fontsize=30)
    plt.ylabel("FLUXCAL", fontsize=30)
    if title is not None:
        ax.legend(loc="upper left", fontsize=15, title=title, title_fontsize=15)
    else:
        ax.legend(loc="upper left", fontsize=15)
    ax.legend(loc="upper left", fontsize=17, title=legend_title, title_fontsize=22)
    plt.tight_layout()

    return fig


def plot_contamination_statistics(
    ax, test_features_df, prediction_type_nos, save_fig_prefix=None
):
    """
    plot displaying total number of events and number of events correctly classified for each event type.
    Parameters
    ----------
    ax: matplotlib.axes
        axes on which plot is to be made
    """
    sns.set_theme(
        style={
            "axes.grid": False,
            "axes.labelcolor": "black",
            "figure.facecolor": "1",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "image.cmap": "viridis",
        }
    )
    performance_statistics_df = get_performance_statistics_df(
        test_features_df, prediction_type_nos
    )

    col_type_nos = np.array(performance_statistics_df.columns)
    # print(col_type_nos)
    pred_col_names = [snana_ob_type_name(item) for item in prediction_type_nos]
    non_pred_types = col_type_nos[~np.isin(col_type_nos, prediction_type_nos)]
    # print(pred_col_names)
    non_pred_type_names = [snana_ob_type_name(item) for item in non_pred_types]
    col_type_names = [snana_ob_type_name(item) for item in col_type_nos]

    # print(col_type_nos)
    # print(non_pred_types)

    # print(np.where(np.in1d(non_pred_types, col_type_nos)))
    ax.barh(
        col_type_names,
        performance_statistics_df.loc[1],
        alpha=0.6,
        tick_label=col_type_names,
        color="#D5D5D3",
        ec="black",
        linewidth=1,
        label="Total events",
    )
    ax.barh(
        non_pred_type_names,
        performance_statistics_df[non_pred_types].loc[0],
        alpha=1,
        color="#F5622E",
        ec="black",
        label="Correct non-KN",
    )
    ax.barh(
        pred_col_names,
        performance_statistics_df[prediction_type_nos].loc[0],
        alpha=1,
        color="#15284F",
        ec="black",
        label="Correct KN",
    )
    # plt.rc('ytick', labelsize=15)
    # plt.rc('xtick', labelsize=15)
    plt.xlabel("Number of events", fontsize=25)
    ax.tick_params(axis="both", labelsize=25)
    # print(col_type_nos)
    for i, v in enumerate(col_type_nos):
        ax.text(
            performance_statistics_df[v].values[1] + 10,
            i - 0.2,
            f"{performance_statistics_df[v].values[0] / performance_statistics_df[v].values[1] * 100:.2f}%",
            color="black",
            fontsize=20,
        )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fancybox=True,
        shadow=True,
        prop={"size": 25},
    )
    plt.xlim(right=np.max(performance_statistics_df.loc[1].values) * 125 / 100)
    if save_fig_prefix is not None:
        plt.savefig(
            os.path.join("results", save_fig_prefix, "contamination_plot"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join("results", save_fig_prefix, "contamination_plot.pdf"),
            bbox_inches="tight",
        )


def get_performance_statistics_df(test_features_df, prediction_type_nos):
    """
    functions to evaluate performance for each event type

    Returns
    -------
    performance_stats: pandas.DataFrame
        df with number of events of each type: correctly classified, total number of events of the type and
        number of events of the type in training set
    """
    prediction_stat = {}
    for i, object_id in enumerate(test_features_df["key"]):
        # print(1)
        type_no = test_features_df["type"].values[
            np.where(test_features_df["key"] == object_id)
        ][0]
        # print(self.train_sample_numbers)
        # num_training_events = sample_numbers_train[type_no]
        # if num_training_events == 0:
        #    type_no = 0

        if type_no not in prediction_stat:
            # prediction_stat[type_no] = [0, 1, num_training_events]
            prediction_stat[type_no] = [0, 1]
        else:
            prediction_stat[type_no][1] = prediction_stat[type_no][1] + 1

        if (type_no in prediction_type_nos) & (
            test_features_df["y_pred_score"].values[i] >= 0.5
        ):
            prediction_stat[type_no][0] = prediction_stat[type_no][0] + 1

        elif (test_features_df["y_pred_score"].values[i] < 0.5) & (
            type_no not in prediction_type_nos
        ):
            prediction_stat[type_no][0] = prediction_stat[type_no][0] + 1
    stat_df = pd.DataFrame(prediction_stat)
    performance_stats = stat_df.reindex(sorted(stat_df.columns), axis=1)
    return performance_stats


def plot_confusion_matrix(
    ax,
    y_true,
    y_pred,
    cmap=LinearSegmentedColormap.from_list("", ["#FFFFFF", "#15284F"]),
    save_fig_prefix=None,
):
    """
    This function prints and plots the confusion matrix.

    Parameters
    ----------
    ax: matplotlib.axes
        axes on which plot is to be generated
    cmap: matplotlib colormap
        color map for plotting
    :return:
    """

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    cm = cm / len(y_true)
    print(cm)
    # Only use the labels that appear in the data
    classes = ["non-KN", "KN"]

    # fig, ax = plt.subplots()
    _ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
    )

    ax.tick_params(axis="both", labelsize=25)
    ax.set_xlabel("Predicted label", fontsize=30)
    ax.set_ylabel("True label", fontsize=30)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    plt.yticks(rotation="vertical", va="center")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]*100:.2f}%",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=25,
            )
    # fig.tight_layout()
    ax.axis("equal")

    if save_fig_prefix is not None:

        plt.savefig(
            os.path.join("results", save_fig_prefix, "confusion_matrix"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join("results", save_fig_prefix, "confusion_matrix.pdf"),
            bbox_inches="tight",
        )


def plot_features_correlation_helper(
    class_features_df,
    bands,
    num_pc_components,
    color,
    fig=None,
    x_limits=None,
    y_limits=None,
    mark_xlabel=False,
    mark_ylabel=False,
    band_map=None,
    marker="o",
    set_ax_title=False,
    label="",
):
    """
    plots correlations between PCs of each band for only 1 class of data: ex KN
    class_features_df: dataframe of events of current class (KN and non-KN)
    bands: bands for which plots are to be generated
    color: colors make the plot
    fig: fig on which plot is generated. If None, new fig is created
    x_limits: x limits of the plot
    y_limits: y limits of the plot
    mark_xlabel: mark x label or not
    mark_ylabel: to mark y label or not
    band_map: renaming bands/filter/channel name in plots
    set_ax_title: title of the axes ojbect on which plot is made
    label: string label of the current class (ex "KN" or "non-KN")
    :return: figure with the plots
    """
    num_rows = len(bands)
    num_cols = int(num_pc_components * (num_pc_components - 1) / 2)
    if fig is None:
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_rows * 5, num_cols * 5)
        )
        # fig.subplots_adjust(wspace=.5,hspace=.5)
        ax_list = fig.axes
    else:
        ax_list = fig.axes

    for i, band in enumerate(bands):
        for x in range(num_pc_components):
            for y in range(x):
                ax_current = ax_list[int(i * num_cols + (x - 1) * (x) / 2 + y)]
                colx_name = "coeff" + str(x + 1) + "_" + str(band)
                coly_name = "coeff" + str(y + 1) + "_" + str(band)
                if mark_xlabel:
                    ax_current.set_xlabel(f"$c_{x + 1:d}$", fontsize=20)
                if mark_ylabel:
                    ax_current.set_ylabel(f"$c_{y + 1:d}$", fontsize=20)
                PCx = class_features_df[colx_name].values
                PCy = class_features_df[coly_name].values

                ax_current.scatter([], [], label=label, marker=marker, color=color)

                ax_current.scatter(
                    PCx,
                    PCy,
                    color=color,
                    alpha=0.3,
                    marker=marker,
                )
                ax_current.tick_params(axis="both", labelsize=14)

                if x_limits is not None:
                    ax_current.set_xlim(x_limits)
                if y_limits is not None:
                    ax_current.set_ylim(y_limits)
                if set_ax_title:
                    if band_map is None:
                        ax_current.set_title(
                            "PCs for " + str(band) + "-band", fontsize=20
                        )
                    else:
                        ax_current.set_title(
                            "PCs for " + str(band_map[band]) + "-band", fontsize=20
                        )

                if label != "":
                    if band_map is None:
                        title = "Correlation in " + str(band) + "-band"
                    else:
                        title = "Correlation in " + str(band_map[band]) + "-band"

                    ax_current.legend(
                        loc="lower left", fontsize=14, title=title, title_fontsize=14
                    )
                ax_current.set_aspect("equal", "box")
                # ax_current.axis('square')
    fig.tight_layout()
    return fig


def plot_features_correlation(
    features_df,
    bands,
    color_dict=None,
    x_limits=None,
    y_limits=None,
    mark_xlabel=True,
    mark_ylabel=True,
    band_map=None,
    set_ax_title=False,
    num_kn_points=None,
    num_non_kn_points=None,
    num_pc_components=3,
    save_fig_prefix=None,
):
    """
    plots correlations between the PCs of each band (with the training set features)

    features_df: dataframe of events of current class (KN and non-KN)
    bands: bands for which plots are to be generated
    color_dict:test colors to be used for corresponding classes
    x_limits: x limits of the plot
    y_limits: y limits of the plot
    mark_xlabel: mark x label or not
    mark_ylabel: to mark y label or not
    band_map: renaming bands/filter/channel name in plots
    set_ax_title: title of the axes ojbect on which plot is made
    :return: figure with the plots
    """
    sns.set_theme(
        style={
            "axes.grid": True,
            "axes.labelcolor": "black",
            "figure.facecolor": "1",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "image.cmap": "viridis",
        }
    )
    if color_dict is None:
        color_dict = {"non_kn": "#F5622E", "kn": "#15284F"}

    kn_df = features_df[features_df["y_true"] == 1]
    non_kn_df = features_df[features_df["y_true"] == 0]

    if (num_kn_points is not None) & (num_kn_points < len(kn_df)):
        mask = np.zeros((len(kn_df)), dtype=bool)
        mask[0:num_kn_points] = True
        np.random.shuffle(mask)
        kn_df = kn_df[mask]

    non_kn_df = features_df[features_df["y_true"] == 0]

    if (num_non_kn_points is not None) & (num_non_kn_points < len(non_kn_df)):
        mask = np.zeros((len(non_kn_df)), dtype=bool)
        mask[0:num_non_kn_points] = True
        np.random.shuffle(mask)
        non_kn_df = non_kn_df[mask]

    num_rows = len(bands)
    num_cols = int(num_pc_components * (num_pc_components - 1) / 2)
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(num_pc_components * 5, len(bands) * 5)
    )
    # fig.subplots_adjust(wspace=.5,hspace=.5)
    plot_features_correlation_helper(
        non_kn_df,
        fig=fig,
        band_map=band_map,
        color=color_dict["non_kn"],
        bands=bands,
        x_limits=x_limits,
        y_limits=y_limits,
        mark_xlabel=mark_xlabel,
        mark_ylabel=mark_ylabel,
        set_ax_title=set_ax_title,
        label="non-KN",
        marker="o",
        num_pc_components=num_pc_components,
    )
    plot_features_correlation_helper(
        kn_df,
        fig=fig,
        band_map=band_map,
        color=color_dict["kn"],
        bands=bands,
        x_limits=x_limits,
        y_limits=y_limits,
        mark_xlabel=mark_xlabel,
        mark_ylabel=mark_ylabel,
        set_ax_title=set_ax_title,
        label="KN",
        marker="v",
        num_pc_components=num_pc_components,
    )

    if save_fig_prefix is not None:
        plt.savefig(
            os.path.join("results", save_fig_prefix, "features_corrlation"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join("results", save_fig_prefix, "features_corrlation.pdf"),
            bbox_inches="tight",
        )
    return fig


def plot_band_correlation_helper(
    current_class_df,
    bands,
    color=None,
    fig=None,
    x_limits=None,
    y_limits=None,
    mark_xlabel=False,
    mark_ylabel=False,
    band_map=None,
    set_ax_title=False,
    label="",
    marker=None,
    pc_names=None,
    num_pc_components=3,
):
    """
    current_class_df: dataframe of events of current class (KN and non-KN)
    bands: bands for which plots are to be generated
    color: color for the class
    fig: fig on which plot is generated. If None, new fig is created
    x_limits: x limits of the plot
    y_limits: y limits of the plot
    mark_xlabel: mark x label or not
    mark_ylabel: to mark y label or not
    band_map: renaming bands/filter/channel name in plots
    set_ax_title: title of the axes ojbect on which plot is made
    label: string label of the current class (ex "KN" or "non-KN")
    num_pc_components: number of pcs
    :return: figure with the plots
    """
    num_rows = int(len(bands) * (len(bands) - 1) / 2)
    num_cols = num_pc_components
    if fig is None:
        fig, axs = plt.subplots(
            num_rows, num_cols, figsize=(num_pc_components * 5, len(bands) * 5)
        )
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        ax_list = fig.axes
    else:
        ax_list = fig.axes
    for i in range(num_pc_components):
        # print("pc "+str(i))
        for x, band in enumerate(bands):
            for y in range(x):
                x_band = bands[x]
                y_band = bands[y]
                # print("x " +str(x))
                # print("y " +str(y))
                # print(int(i*len(bands)*(len(bands)-1)/2 + (x-1)*(x-2)/2 +y))
                ax_current = ax_list[int(i * num_rows + (x - 1) * (x - 2) / 2 + y)]
                colx_name = "coeff" + str(i + 1) + "_" + str(x_band)
                coly_name = "coeff" + str(i + 1) + "_" + str(y_band)
                # print(coeff_plot_data)
                PCx = current_class_df[colx_name].values
                PCy = current_class_df[coly_name].values

                sns.kdeplot(
                    x=PCx,
                    y=PCy,
                    color=color,
                    marker=marker,
                    alpha=0.9,
                    ax=ax_current,
                    levels=4,
                )
                ax_current.scatter([], [], label=label, marker=marker, color=color)

                if pc_names is None:
                    title = f"Correlation for $c_{i + 1:d}$"
                else:
                    title = "Correlation for " + pc_names[i]

                ax_current.legend(
                    loc="lower left", fontsize=14, title=title, title_fontsize=14
                )

                ax_current.tick_params(axis="both", labelsize=14)

                if x_limits is not None:
                    ax_current.set_xlim(x_limits)
                if y_limits is not None:
                    ax_current.set_ylim(y_limits)

                if band_map is None:
                    if mark_xlabel:
                        ax_current.set_xlabel(x_band + " band", fontsize=20)
                    if mark_ylabel:
                        ax_current.set_ylabel(y_band + " band", fontsize=20)
                else:
                    if mark_xlabel:
                        ax_current.set_xlabel(band_map[x_band] + " band", fontsize=20)
                    if mark_ylabel:
                        ax_current.set_ylabel(band_map[y_band] + " band", fontsize=20)

                if set_ax_title:
                    ax_current.set_title("correlation for PC" + str(i + 1), fontsize=20)

                ax_current.set_aspect("equal", "box")

    fig.tight_layout()
    return fig


def plot_band_correlation(
    features_df,
    bands,
    color_dict=None,
    x_limits=None,
    y_limits=None,
    mark_xlabel=True,
    mark_ylabel=True,
    band_map=None,
    set_ax_title=False,
    pc_names=None,
    num_kn_points=None,
    num_non_kn_points=None,
    num_pc_components=3,
    save_fig_prefix=None,
):
    """
    plots correlations between 2 bands for each PC (with the training set features)

    bands: bands among which correlation is to be plotted
    color_dict: colors to be used for corresponding classes
    fig: fig on which plot is generated. If None, new fig is created
    x_limits: x limits of the plot
    y_limits: y limits of the plot
    mark_xlabel: mark x label or not
    mark_ylabel: to mark y label or not
    band_map: renaming bands/filter/channel name in plots
    set_ax_title: title of the axes ojbect on which plot is made
    num_pc_components: number of pcs
    :return: figure on which the correlations are plotted
    """

    sns.set_theme(
        style={
            "axes.grid": True,
            "axes.labelcolor": "black",
            "figure.facecolor": "1",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "image.cmap": "viridis",
        }
    )

    if color_dict is None:
        color_dict = {"non_kn": "#F5622E", "kn": "#15284F"}

    kn_df = features_df[features_df["y_true"] == 1]
    if (num_kn_points is not None) & (num_kn_points < len(kn_df)):
        mask = np.zeros((len(kn_df)), dtype=bool)
        mask[0:num_kn_points] = True
        np.random.shuffle(mask)
        kn_df = kn_df[mask]

    non_kn_df = features_df[features_df["y_true"] == 0]

    if (num_non_kn_points is not None) & (num_non_kn_points < len(non_kn_df)):
        mask = np.zeros((len(non_kn_df)), dtype=bool)
        mask[0:num_non_kn_points] = True
        np.random.shuffle(mask)
        non_kn_df = non_kn_df[mask]

    num_rows = int(len(bands) * (len(bands) - 1) / 2)
    num_cols = num_pc_components

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    # fig.subplots_adjust(wspace=space_between_axes,hspace=space_between_axes)
    plot_band_correlation_helper(
        non_kn_df,
        bands=bands,
        fig=fig,
        band_map=band_map,
        x_limits=x_limits,
        y_limits=y_limits,
        mark_xlabel=mark_xlabel,
        mark_ylabel=mark_ylabel,
        set_ax_title=set_ax_title,
        label="non-KN",
        pc_names=pc_names,
        marker="o",
        color=color_dict["non_kn"],
        num_pc_components=num_pc_components,
    )
    plot_band_correlation_helper(
        kn_df,
        bands=bands,
        fig=fig,
        band_map=band_map,
        x_limits=x_limits,
        y_limits=y_limits,
        mark_xlabel=mark_xlabel,
        mark_ylabel=mark_ylabel,
        set_ax_title=set_ax_title,
        label="KN",
        pc_names=pc_names,
        marker="v",
        color=color_dict["kn"],
        num_pc_components=num_pc_components,
    )
    if save_fig_prefix is not None:

        plt.savefig(
            os.path.join("results", save_fig_prefix, "band_correlation_plot.pdf"),
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join("results", save_fig_prefix, "band_correlation_plot"),
            bbox_inches="tight",
        )
    # plt.xlabel(" correlation ")
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    return fig

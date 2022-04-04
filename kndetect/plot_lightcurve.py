import matplotlib.pyplot as plt
import numpy as np

from kndetect.features import calc_prediction
from kndetect.utils import extract_mimic_alerts_region


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
            time_data, predicted_lc, color=color_band_dict[band], linestyle=linestyle
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
        ax.legend(loc="upper right", fontsize=17, title=title, title_fontsize=17)
    else:
        ax.legend(loc="upper right", fontsize=17)
    ax.legend()
    plt.tight_layout()

    return fig

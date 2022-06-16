import copy
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table


class LightCurve:
    """
    Holds data/lightcurves for all bands/filters/channels of a single event: used to make plots and calculate properties
    of individual lightcurve (ex: periodic penalty)

    extracts data of a light curve from an object with a given object id, of Data class
    """

    def __init__(self, data_ob, object_id):

        self.df = Table()
        self.object_id = object_id
        self.df = data_ob.get_data_of_event(object_id)
        self.time_col_name = data_ob.time_col_name
        self.brightness_col_name = data_ob.brightness_col_name
        self.brightness_err_col_name = data_ob.brightness_err_col_name
        self.band_col_name = data_ob.band_col_name
        self.bands = data_ob.bands
        self.points_of_maximum, self.dates_of_maximum = self.get_dates_of_maximum()
        self.priority_regions = None

    def get_band_data(self, band):
        """
        Extracts data for a particular band from the dataset.

        Parameters
        ----------
        band: list
            bands for which data is to be extracted

        Returns
        -------

        data corresponding to the band
        """
        index = self.df[self.band_col_name] == band
        return self.df[index]

    def get_dates_of_maximum(self):
        """
        returns max flux dates and points for the bands present in self.df

        Returns
        -------
        points_of_maximum: dict
            the keys of the dict represent the name of the band.
            and each key contains  a tuple with the date of maximum flux and the value of maximum flux
            in that band
        dates_of_maximum: list
            list containing the dates of the maximum recorded flux
        """
        dates_of_maximum = []
        points_of_maximum = {}
        for band in self.bands:
            # pb_name = band
            current_band_data = self.get_band_data(band)

            if len(current_band_data) > 0:
                current_max_index = np.argmax(
                    current_band_data[self.brightness_col_name]
                )

                current_max_date = current_band_data[self.time_col_name][
                    current_max_index
                ]
                dates_of_maximum.append(current_max_date)
                points_of_maximum[band] = [
                    current_max_date,
                    current_band_data[self.brightness_col_name][current_max_index],
                ]

        return points_of_maximum, dates_of_maximum

    def plot_light_curve(
        self,
        color_band_dict,
        band_map=None,
        fig=None,
        band=None,
        start_date=None,
        end_date=None,
        plot_points=False,
        mark_label=True,
        mark_maximum=True,
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
            start_date = np.amin(self.df[self.time_col_name])
        if end_date is None:
            end_date = np.amax(self.df[self.time_col_name])

        if band is not None:

            if band in self.bands:

                event_df = self.get_time_sliced_df(
                    start_date=start_date, end_date=end_date
                )
                band_df = self.extract_band_data(band=band, event_df=event_df)

                if len(band_df) >= min_points_for_plot:

                    pb_name = band
                    if band_map is not None:
                        pb_name = band_map[band]

                    if plot_points:
                        ax.errorbar(
                            band_df[self.time_col_name],
                            band_df[self.brightness_col_name],
                            band_df[self.brightness_err_col_name],
                            color=color_band_dict[band],
                            fmt=".",
                            marker=markers[band] if markers else "o",
                            markersize=8,
                            markerfacecolor=markerfacecolor,
                            label=pb_name + label_postfix if mark_label else "",
                            alpha=alpha,
                        )
                    else:
                        ax.errorbar(
                            band_df[self.time_col_name],
                            band_df[self.brightness_col_name],
                            band_df[self.brightness_err_col_name],
                            markersize=8,
                            markerfacecolor=markerfacecolor,
                            color=color_band_dict[band],
                            label=pb_name + label_postfix if mark_label else "",
                            marker=markers[band] if markers else "o",
                            alpha=alpha,
                        )

                    if mark_maximum:
                        fig = self.mark_maximum_in_plot(
                            color_band_dict=color_band_dict,
                            fig=fig,
                            band=band,
                            start_date=start_date,
                            end_date=end_date,
                        )
                    if clip_xlims is not None:
                        ax.set_xlim([start_date, end_date])

            else:
                print("the band requested is not present")

        else:

            data_points_found = 0

            for band in self.bands:

                pb_name = band
                if band_map is not None:
                    pb_name = band_map[band]

                band_index = self.df[self.band_col_name] == band
                start_index = self.df[self.time_col_name] >= start_date
                end_index = self.df[self.time_col_name] <= end_date

                index = band_index * start_index * end_index

                # print(sum(index))
                if sum(index) > 0:
                    data_points_found = 1
                    df_plot_data = self.df[index]

                    if plot_points:
                        ax.errorbar(
                            df_plot_data[self.time_col_name],
                            df_plot_data[self.brightness_col_name],
                            df_plot_data[self.brightness_err_col_name],
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
                            df_plot_data[self.time_col_name],
                            df_plot_data[self.brightness_col_name],
                            df_plot_data[self.brightness_err_col_name],
                            markersize=8,
                            markerfacecolor=markerfacecolor,
                            color=color_band_dict[band],
                            marker=markers[band] if markers else "o",
                            label=pb_name + " " + label_postfix if mark_label else "",
                            alpha=alpha,
                        )

                if mark_maximum:
                    fig = self.mark_maximum_in_plot(
                        color_band_dict=color_band_dict,
                        fig=fig,
                        band=band,
                        start_date=start_date,
                        end_date=end_date,
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

    def get_time_sliced_df(self, start_date=None, end_date=None):
        """
        gets time sliced data between start date and end date

        Parameters
        ----------
        start_date: start date of the slice to be extracted
        end_date: end date of the slice to be extracted

        Returns
        -------
        extracted_region: astropy.Table
            time sliced data
        """
        event_df = self.df
        if start_date is None:
            if end_date is None:
                return event_df
            start_date = np.amax(event_df[self.time_col_name])
        if end_date is None:
            end_date = np.amax(event_df[self.time_col_name])
        start_index = event_df[self.time_col_name] >= start_date
        end_index = event_df[self.time_col_name] <= end_date

        extracted_region = event_df[start_index & end_index]
        return extracted_region

    def extract_band_data(self, band, event_df=None):
        """
        extracts data from a particular band

        Parameters
        ----------
        band: name of the filter whose data is to be extracted
        event_df: data from which band is to be extracted. If nothing is passed, set to self.df

        Returns
        -------
        band_data: astropy.Table
            data of a given band
        """
        if event_df is None:
            event_df = self.df
        band_index = event_df[self.band_col_name] == band

        band_data = event_df[band_index]
        return band_data

    def get_max_point_of_band(
        self, band, start_date=None, end_date=None, event_df=None
    ):
        """
        returns  a tuple with the date of maximum flux and the maximum flux

        Parameters
        ----------
        band: str
            band whose maximum point is to be extracted
        start_date: float
            start date of the extraction region. Default: start date of the data
        end_date: float
            end date of the extraction region. Default: end date of the data
        event_df: astropy.Table
            data from which extraction is to be done. Default: self.df

        Returns
        -------
        max_time: float
            if there is at least one data point, it returns the date of maximum flux. else returns None.
        max_flux: float
            maximum flux recoded in the band, if there is at least 1 point. else returns None.
        """
        if event_df is None:
            event_df = self.get_time_sliced_df(start_date, end_date)
        band_df = self.extract_band_data(band, event_df)
        if len(band_df) > 0:
            loc = np.argmax(band_df[self.brightness_col_name])
            max_time = band_df[self.time_col_name][loc]
            max_flux = band_df[self.brightness_col_name][loc]
            return max_time, max_flux
        else:
            return None, None

    def mark_maximum_in_plot(
        self, color_band_dict, fig, band=None, start_date=None, end_date=None
    ):
        """
        Marks the maximum point of each band on the plots (as circles)

        Parameters
        ----------
        color_band_dict: dict
            dict with color corresponding to each band
        fig: matplotlib.figure
            plot on which the maximum is to be marked
        band: list
            band on which maximum is to be marked
        start_date: float
            start date of the interval in which max is to be calculated. Default: start day of data
        end_date: float
            end date of the interval in which maximum is to be calculated. Default: end day of data

        Returns
        -------
        fig: matplotlib.figure
            figure with the plots
        """

        ax = fig.gca()
        if band is None:
            for band in self.bands:
                max_point = self.get_max_point_of_band(
                    band=band, start_date=start_date, end_date=end_date
                )
                if max_point is not None:
                    ax.plot(
                        max_point[0],
                        max_point[1],
                        color=color_band_dict[band],
                        marker="o",
                        markersize=15,
                    )
        else:
            max_point = self.get_max_point_of_band(
                band=band, start_date=start_date, end_date=end_date
            )
            if max_point is not None:
                ax.plot(
                    max_point[0],
                    max_point[1],
                    color=color_band_dict[band],
                    marker="o",
                    markersize=15,
                )
        return fig

    def find_region_priority(self, total_days_range=100):
        """
        finds a region of total_days range, where the maximum flux of most bands are located. This is particularly be
        useful if the light-curves are of a long duration (finding this region ensures that the maximums of the bands
        that we are considering for generating features are not located too far from each other)

        Parameters
        ----------
        total_days_range: int
            size of the bin to be considered while extracting data

        Returns
        -------
        priority_regions: list
            a list of a lists containing clusters
        """

        dates_of_maximum_copy = copy.copy(self.dates_of_maximum)
        dates_of_maximum_copy.sort()
        priority_regions = [[]]

        for date in dates_of_maximum_copy:

            if len(priority_regions[0]) == 0:
                priority_regions[0].append(date)

            else:
                region_flag = 0
                for region in priority_regions:

                    modified_region = copy.copy(region)
                    modified_region.append(date)

                    new_median = median(modified_region)

                    for region_date in region:

                        if ((date - region_date) <= 14) | (
                            (date - new_median) <= total_days_range / 2
                        ):
                            region.append(date)
                            region_flag = 1
                            break

                if region_flag != 1:
                    priority_regions.append([date])

        def find_len(e) -> int:
            return len(e)

        priority_regions.sort(reverse=True, key=find_len)
        return priority_regions

    def plot_max_flux_regions(
        self,
        color_band_dict,
        event_days_range=100,
        plot_points=False,
        priority=None,
        band=None,
        mark_label=True,
        mark_maximum=True,
        label_postfix="",
        clip_xlims=None,
        alpha=1.0,
    ):
        """
        plots the region of light curve where most of the band maximums are located.

        Parameters
        ----------
        color_band_dict: dict
            mapping from band/filter name to color with which it is to be drawn
        event_days_range: int
            size of bin in which the maximum fluxes of most bands should lie
        plot_points: bool
            mark the recorded data points on the curve
        priority: int
            integer that states the number of priority regions that are to be plotted.
            for example, if priority = 1, only the highest priority region is plotted. (this parameter is useful
            only for lightcurves with large temporal duration)
        band: list
            bands for which plots are to be drawn
        mark_label: bool
            to put label or not
        mark_maximum: bool
            if True, marks the point with highest flux reading for each band
        label_postfix: str
            post fix on label after band name
        clip_xlims: bool
            plots only the region of prediction if set to true
        alpha: float
            alpha value of the lines/points that are to be plotted

        Returns
        -------
        fig: matplotlib.figure
            figure with the plots
        """
        self.priority_regions = self.find_region_priority(event_days_range)
        if priority is not None:
            if priority <= 0:
                print("Error in priority value, priority number must be greater than 1")

        fig = plt.figure(figsize=(12, 6))

        for i, ranges in enumerate(self.priority_regions):

            mid_pt = median(ranges)
            # print(mid_pt)
            start_date = mid_pt - event_days_range / 2
            end_date = mid_pt + event_days_range / 2

            if priority is None:
                fig = self.plot_light_curve(
                    color_band_dict,
                    start_date=start_date,
                    end_date=end_date,
                    plot_points=plot_points,
                )

            else:
                if (i < priority) | (len(ranges) == len(self.priority_regions[i - 1])):
                    single_band_plot = self.plot_light_curve(
                        color_band_dict,
                        start_date=start_date,
                        end_date=end_date,
                        plot_points=plot_points,
                        band=band,
                        mark_label=mark_label,
                        mark_maximum=mark_maximum,
                        label_postfix=label_postfix,
                        clip_xlims=clip_xlims,
                        alpha=alpha,
                    )
                    ax = single_band_plot.gca()
                    ax.remove()
                    ax.figure = fig
                    fig.axes.append(ax)
                    fig.add_axes(ax)
                    plt.close(single_band_plot)
                    del single_band_plot

                    for j in range(i):
                        fig.axes[j].change_geometry(i + 1, 1, j + 1)

                    dummy = fig.add_subplot(i + 1, 1, i + 1)
                    ax.set_position(dummy.get_position())
                    dummy.remove()
                    del dummy

                else:
                    break

        return fig

import numpy as np
from astropy.table import vstack


class Data:
    """
    class that stores the data of all light curves. This class helps to code to adapt to different datasets
    :param df_data: data of with mjd and flux values
    :param object_id_col_name: name of the column containing object ids
    :param time_col_name: name of column storing time data
    :param band_col_name: name of column with band/filter/channel information
    :param df_metadata: dataframe with te metadata
    :param bands: list of bands used to be used for predictions
    :param brightness_col_name: column name of the col that stores flux values
    :param brightness_err_col_name: column name of the col that stores flux error values
    :param target_col_name: column name of the col that stores event type (None if not available)
    """

    def __init__(
        self,
        df_data,
        object_id_col_name,
        time_col_name,
        band_col_name,
        bands,
        df_metadata=None,
        brightness_col_name=None,
        brightness_err_col_name=None,
        target_col_name=None,
    ):

        self.object_id_col_name = object_id_col_name
        self.time_col_name = time_col_name
        self.brightness_col_name = brightness_col_name
        self.brightness_err_col_name = brightness_err_col_name
        self.band_col_name = band_col_name
        self.target_col_name = target_col_name
        self.df_metadata = df_metadata
        self.df_data = df_data

        self.bands = bands

        self.prediction_stat_df = None
        self.sample_numbers = None
        self.num_pc_components = None

    def get_all_object_ids(self):
        """
        :return: np array with all object ids
        """
        if self.df_metadata is not None:
            return np.array(self.df_metadata[self.object_id_col_name])
        else:
            return np.unique(np.array(self.df_data[self.object_id_col_name]))

    def get_ids_of_event_type(self, target):
        """
        :param target: event types whose ids we want to extract
        :return: numpy array with list of ids
        """
        class_ids = []
        if isinstance(target, int):
            if self.target_col_name is None:
                print("Target name not given")
            else:
                event = self.df_metadata[self.target_col_name]
                index = event == target
                object_ids = self.get_all_object_ids()
                class_ids = object_ids[index]
        else:
            class_ids = []

            for target_id in target:
                event = self.df_metadata[self.target_col_name]
                index = event == target_id
                object_ids = self.get_all_object_ids()
                if class_ids is None:
                    class_ids = object_ids[index]
                else:
                    class_ids = vstack([class_ids, object_ids[index]])

        return class_ids

    def get_data_of_event(self, object_id):
        """
        :param object_id: object id of the event that we want to extract
        :return: data of the required event
        """
        index = self.df_data[self.object_id_col_name] == object_id
        return self.df_data[index]

    def get_band_data(self, band):
        """
        :param band: band whose events we want ot extract
        :return: data of events belonging to a particular band
        """
        index = self.df_data[self.band_col_name] == band
        return self.df_data[index]

    def get_object_type_number(self, object_id):
        """
        :param object_id: object id whose event type we want o extract
        :return: event type of the object selected
        """
        if self.target_col_name is None:
            print("Target name not given")
        index = self.df_metadata[self.object_id_col_name] == object_id
        object_type = np.array(self.df_metadata[self.target_col_name][index])
        return object_type[0]

import pandas as pd
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import datetime
from datetime import timedelta

coefficients_COSTIC = {
    "month": {
        "January": 1.11,
        "February": 1.20,
        "March": 1.11,
        "April": 1.06,
        "May": 1.03,
        "June": 0.93,
        "July": 0.84,
        "August": 0.72,
        "September": 0.92,
        "October": 1.03,
        "November": 1.04,
        "December": 1.01
    },
    "week": {
        "Monday": 0.97,
        "Tuesday": 0.95,
        "Wednesday": 1.00,
        "Thursday": 0.97,
        "Friday": 0.96,
        "Saturday": 1.02,
        "Sunday": 1.13
    },
    "day": {
        "hour_weekday": np.array([
            0.264, 0.096, 0.048, 0.024, 0.144, 0.384,
            1.152, 2.064, 1.176, 1.08, 1.248, 1.224,
            1.296, 1.104, 0.84,0.768, 0.768, 1.104,
            1.632, 2.088, 2.232, 1.608, 1.032, 0.624
        ]),
        "hour_saturday": np.array([
            0.408, 0.192, 0.072, 0.048, 0.072, 0.168,
            0.312, 0.624, 1.08, 1.584, 1.872, 1.992,
            1.92, 1.704,1.536, 1.2, 1.248, 1.128,
            1.296, 1.32, 1.392, 1.2, 0.936, 0.696
        ]),
        "hour_sunday": np.array([
            0.384, 0.168, 0.096, 0.048, 0.048, 0.048,
            0.12, 0.216, 0.576, 1.128, 1.536, 1.752,
            1.896, 1.872, 1.656, 1.296, 1.272, 1.248,
            1.776, 2.016, 2.04, 1.392, 0.864, 0.552
        ]),
    },
}
coefficients_RE2020 = {
    "month": {
        "January": 1.05,
        "February": 1.05,
        "March": 1.05,
        "April": 0.95,
        "May": 0.95,
        "June": 0.95,
        "July": 0.95,
        "August": 0.95,
        "September": 0.95,
        "October": 1.05,
        "November": 1.05,
        "December": 1.05
    },
    "day": {
        "hour_weekday": np.array([
            0, 0, 0, 0, 0, 0,
            0, 0.028, 0.029, 0, 0, 0,
            0, 0, 0, 0, 0, 0.007,
            0.022, 0.022, 0.022, 0.007, 0.007, 0.007
        ]),
        "hour_weekend": np.array([
            0, 0, 0, 0, 0, 0,
            0, 0.028, 0.029, 0, 0, 0,
            0, 0, 0, 0, 0, 0.011,
            0.011, 0.029, 0.011, 0.0011, 0.0011, 0.0
        ]),
    },
}


class DHWaterConsumption:
    """
    A class that calculates the water consumption for a building that uses district heating.

    Attributes:
    ----------
    n_dwellings: int
        The number of dwellings in the building
    v_per_dwelling: int, optional
        The volume of water used per dwelling in liters per day, default value is 110
    ratio_bath_shower: float, optional
        The ratio of water used for shower to the total volume of water used in a dwelling per day, default value is 0.8
    t_shower: int, optional
        The average duration of a shower in minutes, default value is 7
    d_shower: int, optional
        The number of showers taken per person per day, default value is 8
    s_moy_dwelling: float, optional
        The average surface area per dwelling in square meters, default value is 49.6
    s_tot_building: float, optional
        The total surface area of the building in square meters, default value is 2480
    method: str, optional
        The method used to calculate the water consumption. Possible values are "COSTIC" and "RE2020". Default value is "COSTIC".

    Methods:
    -------
    get_coefficient_calc_from_df(df)
        Calculates the coefficients for the water consumption calculation based on the method specified
    COSTIC_shower_distribution(df)
        Calculates the water consumption based on the COSTIC method for the given dataframe
    COSTIC_random_shower_distribution(df, optional_columns=False)
        Calculates the water consumption based on the COSTIC method with random shower distribution for the given dataframe
    """

    def __init__(self,
                 n_dwellings,
                 v_per_dwelling=110,
                 ratio_bath_shower=0.8,
                 t_shower=7,
                 d_shower=8,
                 s_moy_dwelling=49.6,
                 s_tot_building=2480,
                 method="COSTIC"):

        self.method = method
        self.n_dwellings = n_dwellings
        self.v_per_dwelling = v_per_dwelling
        self.ratio_bath_shower = ratio_bath_shower
        self.t_shower = t_shower
        self.d_shower = d_shower
        self.s_moy_dwelling = s_moy_dwelling
        self.s_tot_building = s_tot_building

        self.df_coefficient = None
        self.df_daily_sum = None
        self.df_re2020 = None
        self.df_costic = None
        self.df_costic_random = None
        self.df_all = None

        if self.method == "COSTIC":
            self.coefficients = coefficients_COSTIC
        elif self.method == "RE2020":
            self.coefficients = coefficients_RE2020

        self.v_used = self.t_shower * self.d_shower
        self.v_liters_day = self.n_dwellings * self.v_per_dwelling
        self.v_shower_bath_per_day = self.ratio_bath_shower * self.v_liters_day

    def get_coefficient_calc_from_period(self, start, end):

        # if not pd.Timestamp(start) or not pd.Timestamp(end):
        #     raise ValueError("Les valeurs start et end doivent être des timestamps valides.")

        # periods_a = pd.date_range(start=start, end=end, freq='H')

        # end = end + timedelta(days=1)
        date_index = pd.date_range(start=start, end=end, freq='H')
        coefficients_c = []

        self.df_coefficient = pd.DataFrame(
            data=np.zeros(date_index.shape[0]),
            index=date_index,
            columns=["coef"]
        )

        val_list = []
        # for val in self.df_coefficient.index:
        #     if val.day_of_week in range(5):
        #         hour_coefficients = self.coefficients["day"]["hour_weekday"]
        #     elif val.day_of_week == 5:
        #         hour_coefficients = self.coefficients["day"]["hour_saturday"]
        #     else:
        #         hour_coefficients = self.coefficients["day"]["hour_sunday"]
        #
        #     h24 = (
        #         hour_coefficients
        #         * self.coefficients["month"][str(val.month_name())]
        #         * self.coefficients["week"][str(val.day_name())]
        #     )
        #
        #     val_list.append(h24[val.hour])
        #
        # self.df_coefficient['coef'] = val_list

        if self.method == "COSTIC":

            for val in self.df_coefficient.index:
                if val.day_of_week in range(5):
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                elif val.day_of_week == 5:
                    hour_coefficients = self.coefficients["day"]["hour_saturday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_sunday"]

                h24 = (
                        hour_coefficients
                        * self.coefficients["month"][str(val.month_name())]
                        * self.coefficients["week"][str(val.day_name())]
                )
                val_list.append(h24[val.hour])

        elif self.method == "RE2020":

            for val in self.df_coefficient.index:
                if val.day_of_week in range(5):
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_weekend"]

                h24 = (
                        hour_coefficients
                        * self.coefficients["month"][str(val.month_name())]
                )
                val_list.append(h24[val.hour])

        self.df_coefficient['coef'] = val_list
        self.df_daily_sum = self.df_coefficient.resample('D').sum()
        self.df_daily_sum.columns = ["coef_daily_sum"]
        return self.df_coefficient

        self.df_daily_sum = self.df_coefficient.resample('D').sum()
        self.df_daily_sum.columns = ["coef_daily_sum"]
        return self.df_coefficient

    def costic_shower_distribution(self, start, end):

        periods = pd.date_range(start=start, end=end, freq='H')
        # Concaténation des coefficients et de leur daily sum
        self.get_coefficient_calc_from_period(start, end)
        df_co = pd.concat([self.df_coefficient, self.df_daily_sum], axis=1)
        df_co.fillna(method='ffill', inplace=True)
        df_co = df_co.dropna(axis=0)

        # Calcul du nombre de douches par heure
        df_co['consoECS_COSTIC'] = df_co['coef'] * self.v_shower_bath_per_day / df_co['coef_daily_sum']
        return df_co[['consoECS_COSTIC']]

    def costic_random_shower_distribution(self,
                                          start=None,
                                          end=None,
                                          optional_columns=False,
                                          seed=None):

        if seed is not None:
            rs = RandomState(MT19937(SeedSequence(seed)))
        else:
            rs = RandomState()

        periods = pd.date_range(start=start, end=end, freq='T')
        df_costic = self.costic_shower_distribution(start, end)
        df_costic["nb_shower"] = df_costic['consoECS_COSTIC'] / self.v_used
        df_costic["t_shower_per_hour"] = df_costic["nb_shower"] * self.t_shower
        df_costic["nb_shower_int"] = np.round(df_costic["nb_shower"]).astype(int)

        rs_dd = rs.randint(
            0,
            60 - self.t_shower,
            (len(periods), df_costic["nb_shower_int"].max())
        )

        distribution_list = []
        for h, nb_shower in zip(rs_dd, df_costic["nb_shower_int"]):
            starts = h[:nb_shower]
            distribution = np.zeros(60)
            for start_shower in starts:
                distribution[start_shower: start_shower + self.t_shower] += 1
            distribution_list.append(distribution)

        df_costic_random = pd.DataFrame(
            data=np.concatenate(distribution_list),
            index=pd.date_range(
                df_costic["nb_shower_int"].index[0],
                freq='T',
                periods=df_costic.shape[0] * 60
            ),
            columns=['shower_per_minute']
        )

        df_costic_random["consoECS_COSTIC_random"] = df_costic_random["shower_per_minute"] * self.v_used / self.t_shower
        df_costic_random["consoECS_COSTIC_random"] = df_costic_random["consoECS_COSTIC_random"].astype(float)

        self.df_costic_random = df_costic_random

        if optional_columns:
            return df_costic_random
        else:
            return df_costic_random[["consoECS_COSTIC_random"]]


    def re2020_shower_distribution(self, start, end):
        periods = pd.date_range(start=start, end=end, freq='H')
        self.get_coefficient_calc_from_period(start, end)
        # N_calculation
        if self.s_moy_dwelling < 10:
            nmax = 1
        elif self.s_moy_dwelling in {10: 49.99}:
            nmax = 1.75 - 0.01875 * (50 - self.s_moy_dwelling)
        else:
            nmax = 0.035 * self.s_moy_dwelling

        n_adult = nmax * self.n_dwellings
        a = min(392, int(40 * (self.s_tot_building / self.s_moy_dwelling)))
        v_weekly = a * n_adult  # Liters
        v_shower_bath = v_weekly * self.ratio_bath_shower

        # Calcul du nombre de douches par heure
        to_return = self.df_coefficient.copy() * v_shower_bath
        to_return = to_return[:len(periods)]
        to_return.columns = ['consoECS_RE2020']
        return to_return

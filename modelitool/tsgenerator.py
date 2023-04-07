import pandas as pd
import numpy as np
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
        "hour_weekday": {
            '0': 0.264, '1': 0.096, '2': 0.048, '3': 0.024, '4': 0.144, '5': 0.384,
            '6': 1.152, '7': 2.064, '8': 1.176, '9': 1.08, '10': 1.248, '11': 1.224,
            '12': 1.296, '13': 1.104, '14': 0.84, '15': 0.768, '16': 0.768, '17': 1.104,
            '18': 1.632, '19': 2.088, '20': 2.232, '21': 1.608, '22': 1.032, '23': 0.624
        },
        "hour_saturday": {
            '0': 0.408, '1': 0.192, '2': 0.072, '3': 0.048, '4': 0.072, '5': 0.168,
            '6': 0.312, '7': 0.624, '8': 1.08, '9': 1.584, '10': 1.872, '11': 1.992,
            '12': 1.92, '13': 1.704, '14': 1.536, '15': 1.2, '16': 1.248, '17': 1.128,
            '18': 1.296, '19': 1.32, '20': 1.392, '21': 1.2, '22': 0.936, '23': 0.696
        },
        "hour_sunday": {
            '0': 0.384, '1': 0.168, '2': 0.096, '3': 0.048, '4': 0.048, '5': 0.048,
            '6': 0.12, '7': 0.216, '8': 0.576, '9': 1.128, '10': 1.536, '11': 1.752,
            '12': 1.896, '13': 1.872, '14': 1.656, '15': 1.296, '16': 1.272, '17': 1.248,
            '18': 1.776, '19': 2.016, '20': 2.04, '21': 1.392, '22': 0.864, '23': 0.552
        },
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
        "hour_weekday": {
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,
            '6': 0, '7': 0.028, '8': 0.029, '9': 0, '10': 0, '11': 0,
            '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0.007,
            '18': 0.022, '19': 0.022, '20': 0.022, '21': 0.007, '22': 0.007, '23': 0.007
        },
        "hour_weekend": {
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0,
            '6': 0, '7': 0.028, '8': 0.029, '9': 0, '10': 0, '11': 0,
            '12': 0, '13': 0, '14': 0, '15': 0, '16': 0, '17': 0.011,
            '18': 0.011, '19': 0.029, '20': 0.011, '21': 0.0011, '22': 0.0011, '23': 0.0
        },
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
        self.v_liters_day = self.n_dwellings * self.v_per_dwelling  # Volume par jour (L/j)
        self.v_shower_bath_per_day = self.ratio_bath_shower * self.v_liters_day  # Volume d'eau douche/bain/jour (L/j)

    def get_coefficient_calc_from_period(self, start, end):

        # if not pd.Timestamp(start) or not pd.Timestamp(end):
        #     raise ValueError("Les valeurs start et end doivent être des timestamps valides.")

        periods_a = pd.date_range(start=start, end=end, freq='H')

        end = end + timedelta(days=1)
        periods = pd.date_range(start=start, end=end, freq='H')
        coefficients_c = []

        while start <= end:
            current_month = start.strftime('%B')
            current_weekday = start.strftime('%A')
            current_weekday_nb = start.weekday()

            if self.method == "COSTIC":
                if current_weekday_nb < 5:
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                elif current_weekday_nb == 5:
                    hour_coefficients = self.coefficients["day"]["hour_saturday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_sunday"]
                for current_hour in range(24):
                    c = (self.coefficients["month"][str(current_month)]
                         * self.coefficients["week"][str(current_weekday)]
                         * hour_coefficients[str(current_hour)])
                    coefficients_c.append(c)
                start += timedelta(days=1)

            elif self.method == "RE2020":
                if current_weekday_nb < 5:
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_weekend"]
                for current_hour in range(24):
                    c = (self.coefficients["month"][str(current_month)]
                         * hour_coefficients[str(current_hour)])
                    coefficients_c.append(c)
                start += timedelta(days=1)

        self.df_coefficient = pd.DataFrame({'coef': coefficients_c}, index=periods)
        self.df_daily_sum = self.df_coefficient.resample('D').sum()
        self.df_daily_sum.columns = ["coef_daily_sum"]
        df_coefficient = self.df_coefficient[:len(periods_a)]
        return df_coefficient

    def costic_shower_distribution(self, start, end):

        periods = pd.date_range(start=start, end=end, freq='H')

        # Concaténation des coefficients et de leur daily sum
        self.get_coefficient_calc_from_period(start, end)
        self.df_daily_sum = self.df_daily_sum.resample('H').ffill()
        df_co = pd.concat([self.df_coefficient, self.df_daily_sum], axis=1)

        # Calcul du nombre de douches par heure
        df_co['consoECS_COSTIC'] = df_co['coef'] * self.v_shower_bath_per_day / df_co['coef_daily_sum']
        df_co = df_co.dropna(axis=0)
        df_co = df_co[:len(periods)]
        return df_co[['consoECS_COSTIC']]

    def costic_random_shower_distribution(self,
                                          start,
                                          end,
                                          optional_columns=False):

        periods = pd.date_range(start=start, end=end, freq='T')
        df_costic = self.costic_shower_distribution(start, end)

        df_costic["nb_shower"] = df_costic['consoECS_COSTIC'] / self.v_used
        df_costic["t_shower_per_hour"] = df_costic["nb_shower"] * self.t_shower
        df_costic["nb_shower_int"] = np.floor(df_costic["nb_shower"]).astype(int)
        df_costic_random = pd.DataFrame()

        minutes_per_hour = 60
        if self.t_shower > minutes_per_hour:
            raise ValueError("t_shower ne peut pas être supérieur à 60 minutes")

        # Distribution des douches par minute
        def distrib_shower_per_minute(nb_shower):
            distribution = np.zeros(minutes_per_hour)
            for i in range(nb_shower):
                start_shower = np.random.randint(minutes_per_hour - self.t_shower)
                distribution[start_shower: start_shower + self.t_shower] += 1
            return distribution

        df_costic_random['shower_per_minute'] = df_costic['nb_shower_int'].apply(
            distrib_shower_per_minute).values.tolist()
        df_costic_random = df_costic_random.explode('shower_per_minute')
        df_costic = df_costic.resample('1T').ffill()
        df_costic_random = df_costic_random[:len(periods)]
        df_costic_random.index = periods

        df_costic_random = pd.merge(
            df_costic, df_costic_random, left_index=True, right_index=True, how='right')
        df_costic_random["consoECS_COSTIC_random"] = df_costic_random["shower_per_minute"] * 480
        df_costic_random["consoECS_COSTIC_random"] = df_costic_random["consoECS_COSTIC_random"].astype(float)

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

import pandas as pd
import numpy as np
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
            "1": 0.264,"2":0.096,"3":0.048,"4":0.024,"5":0.144,"6":0.384,
            "7":1.152,"8":2.064,"9":1.176,"10":1.080,"11":1.248,"12":1.224,
            "13":1.296, "14":1.104, "15":0.840, "16":0.768, "17":0.768,"18":1.104,
            "19":1.632,"20":2.088,"21":2.232,"22":1.608,"23":1.032,"0":0.624,
        },
        "hour_saturday" : {
            "1": 0.408,"2":0.192,"3":0.072,"4":0.048,"5":0.072,"6":0.168,
            "7":0.312,"8":0.624,"9":1.08,"10":1.584,"11":1.872,"12":1.992,
            "13":1.92, "14":1.704, "15":1.536, "16":1.2, "17":1.248,"18":1.128,
            "19":1.296,"20":1.32,"21":1.392,"22":1.2,"23":0.936,"0":0.696,
        },
        "hour_sunday": {
            "1":0.384,"2":0.168,"3":0.096,"4":0.048,"5":0.048,"6":0.048,
            "7":0.12,"8":0.216,"9":0.576,"10":1.128,"11":1.536,"12":1.752,
            "13":1.896,"14":1.872,"15":1.656,"16":1.296,"17":1.272,"18":1.248,
            "19":1.776,"20":2.016,"21":2.04,"22":1.392,"23":0.864,"0":0.552,
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
            "1": 0,"2":0,"3":0,"4":0,"5":0,"6":0,
            "7":0,"8":0.028,"9":0.029,"10":0,"11":0,"12":0,
            "13":0, "14":0, "15":0, "16":0, "17":0,"18":0.007,
            "19":0.022,"20":0.022,"21":0.022,"22":0.007,"23":0.007,"0":0.007,
        },
        "hour_weekend" : {
            "1": 0,"2":0,"3":0,"4":0,"5":0,"6":0,
            "7":0,"8":0.028,"9":0.029,"10":0,"11":0,"12":0,
            "13":0, "14":0, "15":0, "16":0, "17":0,"18":0.011,
            "19":0.011,"20":0.029,"21":0.011,"22":0.0011,"23":0.0011,"0":0.000,
        },
    },
}

class DHWaterConsumption:
    def __init__(self,
                 df,
                 n_dwellings=50,
                 v_per_dwelling=110,
                 ratio_bath_shower=0.8,
                 t_shower=7,
                 d_shower=8,
                 s_moy_dwelling=49.6,
                 s_tot_building=2480,
                 method=None):
        assert method in ["COSTIC", "RE2020"], "Specifiez la méthode : 'COSTIC' ou 'RE2020'"
        assert isinstance(df, pd.DataFrame), "df doit être un DataFrame"
        self.df = df
        self.n_dwellings = n_dwellings
        self.v_per_dwelling = v_per_dwelling
        self.ratio_bath_shower = ratio_bath_shower
        self.t_shower = t_shower
        self.d_shower = d_shower
        self.s_moy_dwelling = s_moy_dwelling
        self.s_tot_building = s_tot_building
        self.method = method
        self.check_date_index()

        self.df_coefficient = None
        self.df_re2020 = None
        self.df_costic = None
        self.df_costic_random = None
        self.df_all = None

        if self.method == "COSTIC":
            self.coefficients = coefficients_COSTIC
        elif self.method == "RE2020":
            self.coefficients = coefficients_RE2020
            assert s_moy_dwelling, "s_moy_dwelling doit être renseigné pour la méthode RE2020"
            assert s_tot_building, "s_tot_building doit être renseigné pour la méthode RE2020"

        self.coefficient_calc()

        v_used = self.t_shower * self.d_shower
        v_liters_day = self.n_dwellings * self.v_per_dwelling  # Volume par jour (L/j)
        v_shower_bath_per_day = self.ratio_bath_shower * v_liters_day  # Volume d'eau douche/bain/jour (L/j)

        self.v_used = v_used
        self.v_liters_day = v_liters_day
        self.v_shower_bath_per_day = v_shower_bath_per_day

    def check_date_index(self):
        assert isinstance(self.df.index, pd.DatetimeIndex), "L'index doit être de type DatetimeIndex"

    def coefficient_calc(self):
        coefficients_c = []
        df = self.df.copy()

        df.index = pd.to_datetime(df.index)
        start_df = df.index[0]
        end_df = df.index[-1]

        while start_df <= end_df:
            current_month = start_df.strftime('%B')
            current_weekday = start_df.strftime('%A')
            current_weekday_nb = start_df.weekday()

            if self.method == "COSTIC":
                if current_weekday_nb < 5:
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                elif current_weekday_nb == 5:
                    hour_coefficients = self.coefficients["day"]["hour_saturday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_sunday"]
                for current_hour in range(24):
                    c = self.coefficients["month"][str(current_month)] * self.coefficients["week"][str(current_weekday)] * \
                        hour_coefficients[str(current_hour)]
                    coefficients_c.append(c)
                start_df += timedelta(days=1)

            elif self.method == "RE2020":
                if current_weekday_nb < 5:
                    hour_coefficients = self.coefficients["day"]["hour_weekday"]
                else:
                    hour_coefficients = self.coefficients["day"]["hour_weekend"]
                for current_hour in range(24):
                    c = self.coefficients["month"][str(current_month)] * hour_coefficients[str(current_hour)]
                    coefficients_c.append(c)
                start_df += timedelta(days=1)

        # self.coefficient_c_df = pd.DataFrame({'coef': coefficients_c})
        self.df_coefficient= pd.DataFrame({'coef': coefficients_c})
        self.df_coefficient.index = df.index[:len(self.df_coefficient)]
        self.df_daily_sum = self.df_coefficient.resample('D').sum()

    def COSTIC_shower_distribution(self):
        df_co = self.df.copy()
        # calcul de la daily sum des coefficients pour COSTIC
        # df.index = pd.to_datetime(df.index)
        df_co = pd.concat([df_co, self.df_coefficient], axis=1)
        df_co['coef_daily_sum'] =self.df_daily_sum.reindex(df_co.index, method='ffill')

        # Calcul du nombre de douches par heure
        df_co['consoECS_COSTIC'] = df_co['coef'] * self.v_shower_bath_per_day / df_co['coef_daily_sum']
        # df_co['consoECS_COSTIC'][-1] = df_co['consoECS_COSTIC'][-2] ## erreur sur dernier calcul, triche en remplacement la dernière valeur
        self.df_costic = df_co


    def COSTIC_random_shower_distribution(self):

        df_costic = self.df_costic.copy()
        df_costic.index = pd.to_datetime(df_costic.index)

        df_costic["nb_shower"] = self.df_costic['consoECS_COSTIC'] / self.v_used
        df_costic["t_shower_per_hour"] = df_costic["nb_shower"] * self.t_shower
        df_costic["nb_shower_int"] = np.floor(df_costic["nb_shower"]).astype(int)
        df_costic_random = pd.DataFrame()

        minutes_per_hour = 60

        def distrib_shower_per_minute(nb_shower):
            if nb_shower == 0:
                return np.zeros(minutes_per_hour)
            else:
                # Nombre de douches par minute en moyenne
                shower_per_minute = nb_shower / minutes_per_hour / self.t_shower
                # Génération d'une distribution aléatoire uniforme des douches par minutes
                distribution = np.zeros(minutes_per_hour)
                for i in range(nb_shower):
                    start_shower = np.random.randint(minutes_per_hour - self.t_shower)
                    distribution[start_shower: start_shower + self.t_shower] += 1
                return distribution

        df_costic_random['shower_per_minute'] = df_costic['nb_shower_int'].apply(
            distrib_shower_per_minute).values.tolist()
        df_costic_random = df_costic_random.explode('shower_per_minute')
        df_costic = df_costic.resample('1T').ffill()
        df_costic_random = df_costic_random[:len(df_costic)]
        df_costic_random.index = df_costic.index
        df_costic_random = pd.merge(
            df_costic, df_costic_random, left_index=True, right_index=True, how='right')
        df_costic_random["consoECS_COSTIC_random"] = df_costic_random["shower_per_minute"] * 480
        df_costic_random["consoECS_COSTIC_random"] = df_costic_random["consoECS_COSTIC_random"].astype(float)
        self.df_costic_random = df_costic_random

    def re_2020_shower_distribution(self):
        df_re = self.df.copy()
            # N_calculation
        if self.s_moy_dwelling < 10:
            nmax = 1
        elif self.s_moy_dwelling in {10: 49.99}:
            nmax = 1.75 - 0.01875 * (50 - self.s_moy_dwelling)
        else:
            nmax = 0.035 * self.s_moy_dwelling

        df_re = pd.concat([df_re, self.df_coefficient], axis=1)
        n_adult = nmax * self.n_dwellings
        a = min(392, int(40 * (self.s_tot_building / self.s_moy_dwelling)))
        v_weekly = a * n_adult  # Liters
        v_shower_bath = v_weekly * self.ratio_bath_shower

        # Calcul du nombre de douches par heure
        df_re['consoECS_RE2020'] = df_re['coef'] * v_shower_bath
        self.df_re2020 = df_re

    def get_complete_calc(self):
        if self.method == 'COSTIC':
            self.df_all = self.df_costic_random.drop(
                columns=["coef", "coef_daily_sum", "nb_shower", "t_shower_per_hour", "nb_shower_int", "shower_per_minute"])
        elif self.method == "RE2020":
            self.df_all = self.df_re2020.drop(columns = ["coef"])

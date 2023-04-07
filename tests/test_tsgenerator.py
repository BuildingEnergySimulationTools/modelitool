import pandas as pd
import numpy as np
from modelitool.tsgenerator import DHWaterConsumption
import datetime as dt


class TestDHWaterConsumption:

    def test_get_coefficient_calc_from_period(self):
        df = pd.DataFrame(
            index=pd.date_range("2023-01-01 00:00:00", freq='H', periods=8760)
        )
        start = df.index[0]
        end = df.index[-1]

        # test COSTIC
        dhw1 = DHWaterConsumption(n_dwellings=50)
        dhw1.get_coefficient_calc_from_period(start=start, end=end)
        assert round(dhw1.df_coefficient.loc["2023-04-05 00:00", 'coef'], 4) == round(1.06 * 0.264 * 1.00, 4)
        assert round(dhw1.df_coefficient.loc["2023-08-26 20:00", 'coef'], 4) == round(1.392 * 0.72 * 1.02, 4)
        assert round(dhw1.df_coefficient.loc["2023-09-10 11:00", 'coef'], 4) == round(1.752 * 0.92 * 1.13, 4)

        # #test RE2020
        dhw2 = DHWaterConsumption(n_dwellings=50, method='RE2020')
        dhw2.get_coefficient_calc_from_period(start=start, end=end)
        assert round(dhw2.df_coefficient.loc["2023-04-05 00:00", 'coef'], 4) == round(0 * 0.95, 4)
        assert round(dhw2.df_coefficient.loc["2023-08-25 20:00", 'coef'], 4) == round(0.022 * 0.95, 4)
        assert round(dhw2.df_coefficient.loc["2023-09-10 18:00", 'coef'], 4) == round(0.011 * 0.95, 4)

    def test_get_coefficient_calc_from_period_2(self):
        dhw = DHWaterConsumption(n_dwellings=50)
        start = dt.datetime(2022, 1, 1)
        end = dt.datetime(2022, 12, 31, 23, 0, 0)
        df = dhw.costic_random_shower_distribution(start=start, end=end)

        assert isinstance(df, pd.DataFrame)
        assert 'consoECS_COSTIC_random' in df.columns

        # vérifier que la somme de la consommation d'eau aléatoire est égale à la consommation totale estimée
        total_consoECS_COSTIC = dhw.costic_shower_distribution(start=start, end=end)['consoECS_COSTIC'].sum()
        assert np.isclose(df['consoECS_COSTIC_random'].sum(), total_consoECS_COSTIC, rtol=0.05)

        # vérifier que la consommation d'eau aléatoire est supérieure à zéro
        assert (df['consoECS_COSTIC_random'] > 0).all()

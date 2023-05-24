import requests
import json
import pandas as pd
from .combitabconvert import get_dymo_time_index

# TODO
# Lots of hardcoded things, elevation, GMT+1
# Use Lafrech devs instead

EPW_COLS = [
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Data Source",
    "Dry Bulb Temperature",
    "Dew Point Temperature",
    "Relative Humidity",
    "Atmospheric Station Pressure",
    "Extraterrestrial Horizontal Radiation",
    "Extraterrestrial Direct Normal Radiation",
    "Horizontal Infrared Radiation Intensity",
    "Global Horizontal Radiation",
    "Direct Normal Radiation",
    "Diffuse Horizontal Radiation",
    "Global Horizontal Illuminance",
    "Direct Normal Illuminance",
    "Diffuse Horizontal Illuminance",
    "Zenith Luminance",
    "Wind Direction",
    "Wind Speed",
    "Total Sky Cover",
    "Opaque Sky Cover",
    "Visibility",
    "Ceiling Height",
    "Present Weather Observation",
    "Present Weather Codes",
    "Precipitable Water",
    "Aerosol Optical Depth",
    "Snow Depth",
    "Days Since Last Snowfall",
    "Albedo",
    "Liquid Precipitation Quantity",
]

TMY5_COLS = [
    "seconds index",
    "Dry Bulb Temperature",
    "Dew Point Temperature",
    "Relative Humidity",
    "Atmospheric Station Pressure",
    "Extraterrestrial Horizontal Radiation",
    "Extraterrestrial Direct Normal Radiation",
    "Horizontal Infrared Radiation Intensity",
    "Global Horizontal Radiation",
    "Direct Normal Radiation",
    "Diffuse Horizontal Radiation",
    "Global Horizontal Illuminance",
    "Direct Normal Illuminance",
    "Diffuse Horizontal Illuminance",
    "Zenith Luminance",
    "Wind Direction",
    "Wind Speed",
    "Total Sky Cover",
    "Opaque Sky Cover",
    "Visibility",
    "Ceiling Height",
    "Present Weather Observation",
    "Present Weather Codes",
    "Precipitable Water",
    "Aerosol Optical Depth",
    "Snow Depth",
    "Days Since Last Snowfall",
    "Albedo",
    "Liquid precipitation depth in mm at indicated time",
    "Liquid Precipitation Quantity",
]


class OikolabWeatherData:
    def __init__(self, location_name, location_lat, location_long, start, end, api_key):
        self.location_name = location_name
        self.location_lat = location_lat
        self.location_long = location_long
        self.start = start
        self.end = end
        self.api_key = api_key
        self.data = None

    def _get_oikolab_json(self):
        return requests.get(
            "https://api.oikolab.com/weather",
            params={
                "param": [
                    # DryBulb {C}
                    "temperature",
                    # DewPoint {C}
                    "dewpoint_temperature",
                    # RelHum {%}
                    "relative_humidity",
                    # Atmos Pressure {Pa}
                    "surface_pressure",
                    # ExtHorzRad {Wh/m2}
                    # ExtDirRad {Wh/m2}
                    # HorzIRSky {Wh/m2}
                    # GloHorzRad {Wh/m2}
                    "surface_solar_radiation",
                    # DirNormRad {Wh/m2}
                    "direct_normal_solar_radiation",
                    # DifHorzRad {Wh/m2}
                    "surface_diffuse_solar_radiation",
                    # GloHorzIllum {lux}
                    # DirNormIllum {lux}
                    # DifHorzIllum {lux}
                    # ZenLum {Cd/m2}
                    # WindDir {deg}
                    "wind_direction",
                    # WindSpd {m/s}
                    "wind_speed",
                    # TotSkyCvr {.1}
                    "total_cloud_cover",
                    # OpaqSkyCvr {.1}
                    # Visibility {km}
                    # Ceiling Hgt {m}
                    # PresWeathObs
                    # PresWeathCodes
                    # Precip Wtr {mm}
                    # Aerosol Opt Depth {.001}
                    # SnowDepth {cm}
                    "snowfall",
                    # Rain {mm}
                    "total_precipitation",
                    # Days Last Snow Albedo {.01}
                    # Rain Quantity {hr}
                ],
                "start": self.start,
                "end": self.end,
                "lat": self.location_lat,
                "lon": self.location_long,
                "api-key": self.api_key,
            },
        )

    def _format_wea_df(self):
        wea_df = pd.DataFrame(index=self.data.index)
        wea_df["seconds index"] = get_dymo_time_index(wea_df)
        wea_df["Year"] = wea_df.index.year
        wea_df["Month"] = wea_df.index.month
        wea_df["Day"] = wea_df.index.day
        wea_df["Hour"] = wea_df.index.hour
        wea_df["Hour"] = wea_df["Hour"] + 1
        wea_df["Minute"] = wea_df.index.minute
        wea_df["Minute"] = wea_df["Minute"] + 60
        wea_df["Data Source"] = ["?"] * wea_df.shape[0]
        wea_df["Dry Bulb Temperature"] = self.data["temperature (degC)"]
        wea_df["Dew Point Temperature"] = self.data["dewpoint_temperature (degC)"]
        wea_df["Relative Humidity"] = self.data["relative_humidity (0-1)"] * 100
        wea_df["Atmospheric Station Pressure"] = self.data["surface_pressure (Pa)"]
        wea_df["Extraterrestrial Horizontal Radiation"] = [9999] * wea_df.shape[0]
        wea_df["Extraterrestrial Direct Normal Radiation"] = [9999] * wea_df.shape[0]
        wea_df["Horizontal Infrared Radiation Intensity"] = [9999] * wea_df.shape[0]
        wea_df["Global Horizontal Radiation"] = self.data[
            "surface_solar_radiation (W/m^2)"
        ]
        wea_df["Direct Normal Radiation"] = self.data[
            "direct_normal_solar_radiation (W/m^2)"
        ]
        wea_df["Diffuse Horizontal Radiation"] = self.data[
            "surface_diffuse_solar_radiation (W/m^2)"
        ]
        wea_df["Global Horizontal Illuminance"] = [999999] * wea_df.shape[0]
        wea_df["Direct Normal Illuminance"] = [999999] * wea_df.shape[0]
        wea_df["Diffuse Horizontal Illuminance"] = [999999] * wea_df.shape[0]
        wea_df["Zenith Luminance"] = [9999] * wea_df.shape[0]
        wea_df["Wind Direction"] = self.data["wind_direction (deg)"]
        wea_df["Wind Speed"] = self.data["wind_speed (m/s)"]
        wea_df["Total Sky Cover"] = self.data["total_cloud_cover (0-1)"]
        wea_df["Opaque Sky Cover"] = [99] * wea_df.shape[0]
        wea_df["Visibility"] = [9999] * wea_df.shape[0]
        wea_df["Ceiling Height"] = [99999] * wea_df.shape[0]
        wea_df["Present Weather Observation"] = [9] * wea_df.shape[0]
        wea_df["Present Weather Codes"] = ["'999999999"] * wea_df.shape[0]
        wea_df["Precipitable Water"] = [999] * wea_df.shape[0]
        wea_df["Aerosol Optical Depth"] = [0.999] * wea_df.shape[0]
        wea_df["Snow Depth"] = [999] * wea_df.shape[0]
        wea_df["Days Since Last Snowfall"] = [99] * wea_df.shape[0]
        wea_df["Albedo"] = [999] * wea_df.shape[0]
        wea_df["Liquid precipitation depth in mm at indicated time"] = [
            99
        ] * wea_df.shape[0]
        wea_df["Liquid Precipitation Quantity"] = [99] * wea_df.shape[0]

        return wea_df

    def get_data(self):
        r = self._get_oikolab_json()
        weather_data = json.loads(r.json()["data"])
        self.data = pd.DataFrame(
            index=pd.to_datetime(weather_data["index"], unit="s"),
            data=weather_data["data"],
            columns=weather_data["columns"],
        )
        self.data.index = self.data.index.tz_localize("UTC")
        self.data.drop(
            [
                "model (name)",
                "coordinates (lat,lon)",
                "model elevation (surface)",
                "utc_offset (hrs)",
            ],
            axis=1,
            inplace=True,
        )

    def generate_tmy5(self, file_path):
        tmy_df = self._format_wea_df()[TMY5_COLS]

        file = open(file_path, "w")
        file.write("#1 \n")
        file.write(f"double tab1({tmy_df.shape[0]}, {tmy_df.shape[1]})\n")
        file.write(
            f"#LOCATION,{self.location_name},empty,empty,ERA5_NBK_Oikolab,666,"
            f"{self.location_lat},{self.location_long},1.0,10\n"
        )
        file.write(
            f"#DATA PERIODS,1,1,Data,"
            f"{tmy_df.index[0].day_name()},"
            f"{tmy_df.index[0].month}/"
            f"{tmy_df.index[0].day},"
            f"{tmy_df.index[-1].month}/"
            f"{tmy_df.index[-1].day}\n"
        )
        file.write(
            "#C1 Time in seconds. Beginning of a year is 0s.\n"
            "#C2 Dry bulb temperature in Celsius at indicated time\n"
            "#C3 Dew point temperature in Celsius at indicated time\n"
            "#C4 Relative humidity in percent at indicated time\n"
            "#C5 Atmospheric station pressure in Pa at indicated time\n"
            "#C6 Extraterrestrial horizontal radiation in Wh/m2\n"
            "#C7 Extraterrestrial direct normal radiation in Wh/m2\n"
            "#C8 Horizontal infrared radiation intensity in Wh/m2\n"
            "#C9 Global horizontal radiation in Wh/m2\n"
            "#C10 Direct normal radiation in Wh/m2\n"
            "#C11 Diffuse horizontal radiation in Wh/m2\n"
            "#C12 Averaged global horizontal illuminance in\
            lux during minutes preceding the indicated time\n"
            "#C13 Direct normal illuminance in lux during\
            minutes preceding the indicated time\n"
            "#C14 Diffuse horizontal illuminance in lux\
            during minutes preceding the indicated time\n"
            "#C15 Zenith luminance in Cd/m2 during minutes\
            preceding the indicated time\n"
            "#C16 Wind direction at indicated time. N=0, E=90, S=180, W=270\n"
            "#C17 Wind speed in m/s at indicated time\n"
            "#C18 Total sky cover at indicated time\n"
            "#C19 Opaque sky cover at indicated time\n"
            "#C20 Visibility in km at indicated time\n"
            "#C21 Ceiling height in m\n"
            "#C22 Present weather observation\n"
            "#C23 Present weather codes\n"
            "#C24 Precipitable water in mm\n"
            "#C25 Aerosol optical depth\n"
            "#C26 Snow depth in cm\n"
            "#C27 Days since last snowfall\n"
            "#C28 Albedo\n"
            "#C29 Liquid precipitation depth in mm at indicated time\n"
            "#C30 Liquid precipitation quantity\n"
        )

        file.write(
            tmy_df.to_csv(header=False, index=False, sep="\t", line_terminator="\n")
        )
        file.close()

    def generate_epw(self, file_path):
        epw_df = self._format_wea_df()[EPW_COLS]

        file = open(file_path, "w")
        file.write(
            f"LOCATION,{self.location_name},empty,empty,ERA5_NBK_Oikolab,666,"
            f"{self.location_lat},{self.location_long},1.0,10\n"
        )
        # State, Country, Source, WMO number, location latitude,
        # location longitude, GMT +1 "france", Field elevation

        file.write("DESIGN CONDITIONS,0 \n")
        file.write("TYPICAL/EXTREME PERIODS,0 \n")
        line = "GROUND TEMPERATURES,1,1.0,,,"
        mon_mean = epw_df["Dry Bulb Temperature"].groupby(epw_df.index.month).mean()
        for t in mon_mean:
            line += "," + str(t)
        line += "\n"

        file.write(line)

        file.write("HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0 \n")
        file.write("COMMENTS 1,  NOBATEK V1 \n")
        file.write("COMMENTS 2,\n")
        file.write(
            f"DATA PERIODS,1,1,Data,"
            f"{self.data.index[0].day_name()},"
            f"{self.data.index[0].month}/"
            f"{self.data.index[0].day},"
            f"{self.data.index[-1].month}/"
            f"{self.data.index[-1].day}\n"
        )
        file.write(epw_df.to_csv(header=False, index=False, line_terminator="\n"))
        file.close()

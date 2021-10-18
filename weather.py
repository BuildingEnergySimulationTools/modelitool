import requests
import json
import pandas as pd

# TODO
# Lots of hardcoded things, elevation, GMT+1
# Use of java is bad
# Use Lafrech devs instead


class OikolabWeatherData:
    def __init__(
        self,
        location_name,
        location_lat,
        location_long,
        start,
        end,
        time_zone,
        api_key
    ):
        self.location_name = location_name
        self.location_lat = location_lat
        self.location_long = location_long
        self.start = start
        self.end = end
        self.timezone = time_zone
        self.api_key = api_key
        self.data = None

    def _get_oikolab_json(self):
        return requests.get(
            'https://api.oikolab.com/weather',
            params={
                'param': [
                    # DryBulb {C}
                    'temperature',
                    # DewPoint {C}
                    'dewpoint_temperature',
                    # RelHum {%}
                    'relative_humidity',
                    # Atmos Pressure {Pa}
                    'surface_pressure',
                    # ExtHorzRad {Wh/m2}
                    # ExtDirRad {Wh/m2}
                    # HorzIRSky {Wh/m2}
                    # GloHorzRad {Wh/m2}
                    'surface_solar_radiation',
                    # DirNormRad {Wh/m2}
                    'direct_normal_solar_radiation',
                    # DifHorzRad {Wh/m2}
                    'surface_diffuse_solar_radiation',
                    # GloHorzIllum {lux}
                    # DirNormIllum {lux}
                    # DifHorzIllum {lux}
                    # ZenLum {Cd/m2}
                    # WindDir {deg}
                    'wind_direction',
                    # WindSpd {m/s}
                    'wind_speed',
                    # TotSkyCvr {.1}
                    'total_cloud_cover',
                    # OpaqSkyCvr {.1}
                    # Visibility {km}
                    # Ceiling Hgt {m}
                    # PresWeathObs
                    # PresWeathCodes
                    # Precip Wtr {mm}
                    # Aerosol Opt Depth {.001}
                    # SnowDepth {cm}
                    'snowfall',
                    # Rain {mm}
                    'total_precipitation',
                    # Days Last Snow Albedo {.01}
                    # Rain Quantity {hr}
                ],
                'start': self.start,
                'end': self.end,
                'lat': self.location_lat,
                'lon': self.location_long,
                'api-key': self.api_key
            }
        )

    def get_data(self):
        r = self._get_oikolab_json()
        weather_data = json.loads(r.json()['data'])
        self.data = pd.DataFrame(
            index=pd.to_datetime(weather_data['index'], unit='s'),
            data=weather_data['data'],
            columns=weather_data['columns']
        )

        self.data.index = self.data.index.tz_localize("UTC")
        self.data.index = self.data.index.tz_convert(self.timezone)

        self.data.drop(
            [
                'model (name)',
                'coordinates (lat,lon)',
                'model elevation (surface)',
                'utc_offset (hrs)'
            ],
            axis=1,
            inplace=True
        )

    def generate_epw(self, file_path):
        epw_df = pd.DataFrame(index=self.data.index.tz_convert("UTC"))
        epw_df["Year"] = epw_df.index.year
        epw_df["Month"] = epw_df.index.month
        epw_df["Day"] = epw_df.index.day
        epw_df["Hour"] = epw_df.index.hour
        epw_df["Hour"] = epw_df["Hour"] + 1
        epw_df["Minute"] = epw_df.index.minute
        epw_df["Minute"] = epw_df["Minute"] + 60
        epw_df["Data Source"] = ["?"]*epw_df.shape[0]
        epw_df["Dry Bulb Temperature"] = self.data['temperature (degC)']
        epw_df[
            "Dew Point Temperature"
        ] = self.data['dewpoint_temperature (degC)']
        epw_df["Relative Humidity"] = self.data['relative_humidity (0-1)']*100
        epw_df[
            "Atmospheric Station Pressure"
        ] = self.data['surface_pressure (Pa)']
        epw_df[
            "Extraterrestrial Horizontal Radiation"
        ] = [9999] * epw_df.shape[0]
        epw_df[
            "Extraterrestrial Direct Normal Radiation"
        ] = [9999] * epw_df.shape[0]
        epw_df[
            "Horizontal Infrared Radiation Intensity"
        ] = [9999] * epw_df.shape[0]
        epw_df[
            "Global Horizontal Radiation"
        ] = self.data['surface_solar_radiation (W/m^2)']
        epw_df[
            "Direct Normal Radiation"
        ] = self.data['direct_normal_solar_radiation (W/m^2)']
        epw_df[
            "Diffuse Horizontal Radiation"
        ] = self.data['surface_diffuse_solar_radiation (W/m^2)']
        epw_df["Global Horizontal Illuminance"] = [999999] * epw_df.shape[0]
        epw_df["Direct Normal Illuminance"] = [999999] * epw_df.shape[0]
        epw_df["Diffuse Horizontal Illuminance"] = [999999] * epw_df.shape[0]
        epw_df["Zenith Luminance"] = [9999] * epw_df.shape[0]
        epw_df["Wind Direction"] = self.data['wind_direction (deg)']
        epw_df["Wind Speed"] = self.data['wind_speed (m/s)']
        epw_df["Total Sky Cover"] = self.data['total_cloud_cover (0-1)']
        epw_df["Opaque Sky Cover"] = [99] * epw_df.shape[0]
        epw_df["Visibility"] = [9999] * epw_df.shape[0]
        epw_df["Ceiling Height"] = [99999] * epw_df.shape[0]
        epw_df["Present Weather Observation"] = [9] * epw_df.shape[0]
        epw_df["Present Weather Codes"] = ["'999999999"] * epw_df.shape[0]
        epw_df["Precipitable Water"] = [999] * epw_df.shape[0]
        epw_df["Aerosol Optical Depth"] = [.999] * epw_df.shape[0]
        epw_df["Snow Depth"] = [999] * epw_df.shape[0]
        epw_df["Days Since Last Snowfall"] = [99] * epw_df.shape[0]
        epw_df["Albedo"] = [999] * epw_df.shape[0]
        epw_df["Liquid Precipitation Quantity"] = [99] * epw_df.shape[0]

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
        mon_mean = epw_df[
            "Dry Bulb Temperature"].groupby(epw_df.index.month).mean()
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
        file.write(epw_df.to_csv(
            header=False,
            index=False,
            line_terminator='\n'
        ))
        file.close()

# Modelitool

Python tools for modelica.
Use only UTC in datetime index.

## modelitool.measures.MeasuredDats

<span style="background-color: #ffe600">***class modelitool.measures.***<span style="color: #ff0040; font-size:1em">**MeasuredDats**</span>***(data=None,   data_type=\{\},   corr_dict=\{\})***</span>


Handle time series of measured data with missing or incorrect values.

Modelica CombiTimeTable input file can be generated from raw data or corrected data.
 
 
**Parameters:**
>**data :** *DataFrame* index must be a UTC timezone aware datetime index.

>**data_type :** *Dictionary* keys are data_type name (eg. temperatures, power), item is a list of columns

>**corr_dict :** *Dictionary* keys are data_type name. Item is a dictionary holding correction methods properties


>**Available corrections :**
- <code>minmax</code> : Remove values beyond or below specific threshold. Arguments : <code>upper, lower</code>
- <code>derivative</code> : Remove values when absolute variation between to timstep is beyond defined threshold. Arguments : <code>lower_rate, upper_rate</code>
- <code>fill_nan</code> : Fill <code>nan</code> values. Argument is a list of method :
    - <code>bfill</code> : Back propagates the next not nan value
    - <code>ffill</code> : Front propagates the last not nan value
- <code>linear_interpolation</code>: Interpolate value with <code>linear</code> method.

**Attributes**
- **data :** copy of raw data
- **data_type_dict  :** dictionary of data_type
- **corrected_data :** corrected data. Copy of data if no correction is applied
- **corr_dict :** dictionary of corrections
- **correction_journal :** History of measures properties. Gaps and missing values

**Methods**
- **remove_anomalies**_()_ : apply minmax and derivative corrections described in <code>corr_dict</code>. Update correction journal with dict of missing values and measure gaps resume
- **fill_nan**_()_ : apply <code>fill_nan</code> corrections described in <code>corr_dict</code>. Update correction journal with dict of missing values and measure gaps resume
- **resample**_(timestep)_ : resample <code>corrected_data</code> to specify timestep. If no timestep is specified, the function uses the DataFrame mean difference of its time index.
- **generate_combitimetable_input**_(file_path, corrected_data=True)_ : generate a modelica CombiTimeTable input file based on <code>data</code> or <code>corrected_data</code>
- **auto_correct**_()_ : apply <code>remove_anomalies(), fill_nan(), resample()</code> in that order, using default arguments


**Example**

<pre><code>
my_measures = MeasuredDats(
	data = my_dirty_measure_df,
	data_type_dict = {
		"power": ["Power1", "Power2"],
		"temperatures": ["Temperature1", "Temperature2"]
	},
	corr_dict = {
		"power": {
			"minmax": {
				"upper": 40000,
				"lower": 0
			},
			"derivative": {
				"upper_rate": 5000,
				"lower_rate": 0,
			},
			"fill_nan": [
				"linear_interpolation",
				"bfill",
				"ffill"
			],
			"resample": np.mean,
		},
		"temperatures": {
			"minmax": {
				"upper": 50,
				"lower": -2
			},
			"derivative": {
				"upper_rate": 5,
				"lower_rate": 0,
			},
			"fill_nan": [
				"linear_interpolation",
				"bfill",
				"ffill"
			],
			"resample": np.sum,
		}
	}
)
</code></pre>

## modelitool.weather.OikolabWeatherData

<span style="background-color: #ffe600">***class modelitool.weather.***<span style="color: #ff0040; font-size:1em">**OikolabWeatherData**</span>***(location_name=None,   location_lat=None,   location_long=None,    start=None,   end=None,   api_key=None)***</span>

Get meteorological data from oikolab API. Transforms it to epw and TMY5 file
Available data are:
- Dry bulb temperatures [°C]
- Dew point temperatures [°C]
- Relative humidity [%]
- Atmospheric pressure [Pa]
- Global horizontal radiation [W/m²]
- Direct normal radiation [W/m²]
- Diffuse solar radiation [W/m²]
- Wind direction [deg]
- Wind speed [m/s]
- Total cloud cover [0-1]
- Snowfall [mm of water equivalent]
- Total precipitation [mm of water]

 
 
**Parameters:**
>**location_name :** *String* Name of the meteo station.

>**location_lat :** *Float* Meteo site location latitude

>**location_long :** *Float* Meteo site location longitude

>**start :** *String* starting point date time format : <code>"YYYY-MM-DDThh:mm:ss"</code> provite UTC

>**end :** *String* ending point date time format : <code>"YYYY-MM-DDThh:mm:ss"</code> provite UTC

>**api_key :** *String* Oikolab API key

**Attributes**
- **data :** DataFrame holding meteo data

**Methods**
- **get_data**_()_ : Fetch meteorological data from Oikolab API
- **generate_tmy5**_(file_path)_ : generate  a TMY5 *.mos file compatible with Modelica Building library reader
- **generate_epw**_(file_path)_ : generate a EPW *.epw file compatible with energyPlus 

**Example**
<pre><code>
my_weather = OikolabWeatherData(
    location_name="Anglet",
    location_lat=43.47414153237001,
    location_long=-1.5095114151032965,
    start='2017-01-01T00:00:00',
    end='2017-12-31T23:00:00',
    api_key="d414d453fb0f496fb6dfa1c3xxxxxxxx"
)
</code></pre>
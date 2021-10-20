# Modelitool

Python tools for modelica.
Use only UTC in datetime index.

## modelitool.measures.MeasuredDats

<pre><code>
my_measures = MeasuredDats(
    data = my_dirty_measure_df,
    data_type = {
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
                "rate": 5000
            },
            "interpolate": {
                "method": "linear"
            },
            "fillna": {
                "bfill": True,
                "ffill": True,
            }
        }
        "temperatures": {
            "minmax": {
                "upper": 50,
                "lower": -2
            },
            "derivative": {
                "rate": 5
            },
            "interpolate": {
                "method": "linear"
            },
            "fillna": {
                "bfill": True,
                "ffill": True,
            }        
		}
    }
)
</code></pre>

Designed to handle time series of measured data with missing or incorrect values.


A modelica <code>combitimetable</code> input file can be generated from raw data or corrected data.
 

Data must be provided as <code> Pandas DataFrame </code> objects. Index must be a **UTC timezone aware** datetime index.



<code>my_dirty_measure_df</code> is a DataFrame that holds 

<code> get_corrected </code> method apply correction to  copy of passed measured data.
Available corrections :
- <code>minmax</code> : Remove values beyond or below specific threshold
- <code>derivative</code> : Remove values when absolute variation between to timstep is beyond defined threshold
- <code>interpolate</code>: Interpolate <code>nan</code> value with specified method (default is <code>linear</code>)
- <code>ffill</code> : Fill  <code>nan</code> propagating the last value
- <code>bfill</code> : Back fill  <code>nan</code> propagating the following value


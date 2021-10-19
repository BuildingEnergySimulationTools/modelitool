# Modelitool

Python tools for modelica.
Use only UTC in datetime index.

## MeasuredDats

This object is designed to handle incomplete time series of measured data.
Data must be provided as <code> Pandas DataFrame </code> objects. Index must be a **UTC timezone aware** datetime index.

<code> get_corrected </code> method apply correction to  copy of passed measured data.
Available corrections :
- <code>minmax</code> : Remove values beyond or below specific threshold
- <code>derivative</code> : Remove values when absolute variation between to timstep is beyond defined threshold
- <code>interpolate</code>: Interpolate <code>nan</code> value with specified method (default is <code>linear</code>)
- <code>ffill</code> : Fill  <code>nan</code> propagating the last value
- <code>bfill</code> : Back fill  <code>nan</code> propagating the following value

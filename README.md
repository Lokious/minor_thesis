# minor_thesis

## steps:

### real-life data (start from DH line, platfrom data) preprocessing for ML and DL model:
	-use the p spline to get the smooth data, then the dataset only contains 0 at the start or end because the range of meaured date is slightly different
	-the nan in predicted value(from spline) will be change to 0 or the value of the last day
	-
### data simulation: use SDE model to generate plant growth cureves(4 types limitation and 4 types noise)
### classification: CNN + LSTM are used for plant growth limitation classify

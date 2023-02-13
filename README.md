# minor_thesis

## steps:

### data (start from DH line, platfrom data) preprocessing for ML and DL model:
	1) missing value in traits data (
		-deletion(×): 
		there are longitudinal data, only three rep of each genotype
		,and missing value are quite often at the end of days (while still some in the middle), 
		we can not delete the whole time serie with missing value
		-use the same method as the spline and mixed model does?
		
		-or make it simple, fill the NA based on average if there are in the middle, and use the cloest value if the missing value is at the start or at the end.

	2) normalization?

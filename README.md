# Ensemble bias correction using quantile mapping 

Bias correction method using quantile mapping. The aproach uses a cumulative distribution function-transform method of the entire ensemble to ensure the preservation of the internal variability of members. 

  **Article reference - under review**

## Overview of workflow


Inputs: 

	-(A) Model: ensamble members of air temperature (divided into batches) generated using the HadAM4 from the UK Met Office Hadley Centre.

	-(B) Observed: ERA5 (with the same grid/resolution)

Outputs: 

	-(1) Model with bias correction applied

	-(2) Histogram of data distribution before and after bias correction in comparison with ERA5




#### Install dependencies (use Python3)

```python

pip install -r requirements.txt

```

A detailed description of data and methods are included in the article. A summary of the workflow is described in the next figure. 

![CSV example](https://github.com/lizanafj/ensemble-bias-correction/blob/main/resources/bias_correction_workflow.JPG )






# Ensemble bias correction method using quantile mapping 

Bias correction method using quantile mapping. The aproach uses a cumulative distribution function-transform method of the entire ensemble to ensure the preservation of the internal variability of members. 

  ** Article**

## Overview of workflow


Inputs: 

	-(A)

	-(B) 

	-(C) 

	-(D)

Outputs: 

	-(1)

	-(2)

	-(3)

	-(4)

	-G(5)





#### Install dependencies (use Python3)

```python

pip install -r requirements.txt

```

## How to use these **diagnostic analytics**?

First, collect required data for the analysis: 

	-Indoor: temperature, CO2 concentrations

	-Outoor: temperature

Second, prepare the data: 

	-See example in folder /data. 

Third, run the three methods using the functions defined here: 

	-diagnostic indicators.py


### 1. Analysis of the overheating situation of building - Seasonal building overheating index (SBOI): 

In a well-designed building, SBOI should be closer to 0% (Fig. a).

In an overheated indoor environment, SBOI can show different scenarios (Fig. b): 

	- SBOI >10%: slightly overheated indoor environment.
	- SBOI >25%: overheated scenario.
	- SBOI >50%: extremely overheated indoor environment. 
	- SBOI ≈100%: tremendously overheated scenario, where the indoor temperature is always higher than outside. 

  
![CSV example](https://github.com/lizanafj/Indicators-to-assess-the-heat-resilience-of-buildings/blob/master/resources/1_SBOI.jpg )


A detailed description of data and methods are included in the article.






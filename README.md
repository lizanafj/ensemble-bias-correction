# bias correction 🌞

Diagnostic analytics to audit the passive building performance using real long-term monitoring data obtained through indoor and outdoor IoT sensors.

This approach is further detailed in the following scientific article: 

  **E. López-García, J. Lizana, et al. (2022) Monitoring and analytics to measure heat resilience of buildings and support retrofitting by passive cooling**

## Overview

These diagnostic analytics characterise the indoor overheating situation of buildings and identify passive cooling opportunities. The approach is based on three methods. 

First, the overheating situation in the indoor environment is characterised by a seasonal building overheating index (SBOI) ranging from 0% to 100%. 

Second, the indoor environment is diagnosed through a heat balance map that divides building performance into four thermal stages related to the positive or negative influence of total heat flux and the ventilation and infiltration load. 

Third, the air changes (ACH), associated with ventilation and infiltration per thermal stage, are calculated using the CO2-based decay method. 







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


### 2. Analysis of building thermal stages - Heat balance map

The passive thermal performance of the building is analysed through four thermal stages related to the positive or negative influence of total heat flux and the ventilation and infiltration load.
These stages can be labelled according to the three main action groups for the passive conditioning of buildings:

	- Stage 1. Heat modulation. This stage shows cooling periods due to building thermal mass (or sporadic AC operation)
	- Stage 2. Solar and heat gains 1/2. This stage illustrates temperature increasing as a result of solar and heat gains.
	- Stage 3. Solar and heat gains 2/2. This stage illustrates temperature increasing despite the lower outdoor temperature. In this stage, heat fluxes from the building surface and internal heat gains are predominant. 
	- Stage 4. Heat dissipation. This stage 4 is associated with cooling periods mainly due to ventilative cooling (or sporadic AC operation)


![CSV example](https://github.com/lizanafj/Indicators-to-assess-the-heat-resilience-of-buildings/blob/master/resources/2_Thermalbuildingstages.jpg )


### 3. Analysis of air change rates (ACH, h-1) through the CO2-based decay method

This method uses indoor CO2 concentrations in the indoor environment to calculate the air change rate (ACH, h-1) related to ventilation and air infiltration.

![CSV example](https://github.com/lizanafj/Indicators-to-assess-the-heat-resilience-of-buildings/blob/master/resources/3_ACHmethod.png )


### 4. Final outputs of the script 

As a result, the following diagram summarises all the indicators calculated using the three methods. 

![CSV example](https://github.com/lizanafj/Indicators-to-assess-the-heat-resilience-of-buildings/blob/master/resources/4_scriptresults.png )






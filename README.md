# bgev_octave
Octave implementation of the blended generalized extreme value (bGEV) distribution from Castro‐Camilo et al. 2022, including the extension to positive shape parameter from Krakauer 2024.

This repository contains the following Octave function files:

 - bgevcdf.m  bGEV cumulative distribution function.
 - bgevpdf.m  bGEV probability density function (derivative of CDF).
 - bgevinv.m  bGEV quantile function (inverse of CDF).
 - bgevnll.m  bGEV negative log likelihood (negative log of the PDF)
 - gevnll.m   GEV negative log likelihood

As well as
  
 - bgev_fit_demo.m  script with application of bGEV to temperature extremes data
 - bgev_test_data.mat  temperature data for bgev_fit_demo
 - hilo_sl_ann_max.txt  sea level data for bgev_fit_demo

In order to use these functions, the Octave Statistics package is required.

References:

- Castro‐Camilo, D.; Huser, R. & Rue, H. (2022), Practical strategies for generalized extreme value‐based regression models for extremes, Environmetrics, 33, doi: 10.1002/env.2742
- Krakauer, N.Y. (2024), Extending the blended generalized extreme value distribution, Discover Civil Engineering, 1, 97, doi: 10.1007/s44290-024-00102-x

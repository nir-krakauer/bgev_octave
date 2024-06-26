# bgev_octave
Octave implementation of the blended generalized extreme value (bGEV) distribution from Castro‐Camilo et al. 2022.

This repository contains the following Octave function files:

 - bgevcdf.m  bGEV cumulative distribution function.
 - bgevpdf.m  bGEV probability density function (derivative of CDF).
 - bgevinv.m  bGEV quantile function (inverse of CDF).

In order to use these functions, the Octave Statistics package is required.

Reference:
Castro‐Camilo, D.; Huser, R. & Rue, H. (2022), Practical strategies for generalized extreme value‐based regression models for extremes, Environmetrics, 33, doi: 10.1002/env.2742

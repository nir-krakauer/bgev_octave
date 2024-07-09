## Copyright (C) 2024 Nir Krakauer <nkrakauer@ccny.cuny.edu>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {@var{p} =} bgevcdf (@var{x}, @var{k}, @var{sigma}, @var{mu}, @var{p_a}, @var{p_b}, @var{s})
## @deftypefnx {@var{p} =} bgevcdf (@var{x}, @var{k}, @var{sigma}, @var{mu}, @var{p_a}, @var{p_b}, @var{s}, @qcode{"upper"})
##
## Blended generalized extreme value (bGEV) cumulative distribution function (CDF).
##
## For each element of @var{x}, compute the cumulative distribution function
## (CDF) of the bGEV distribution with positive shape parameter @var{k}, scale parameter
## @var{sigma}, location parameter @var{mu}, and blending parameters 
## starting quantile @var{p_a}, ending quantile @var{p_b}, and beta-distribution shape parameter @var{s}.  
##
## The size of @var{p} is the
## common size of @var{x}, @var{k}, @var{sigma}, @var{mu}, @var{p_a}, @var{p_b}, and @var{s}.  A scalar input
## functions as a constant matrix of the same size as the other inputs.
##
## @code{bgevcdf (@var{x}, @var{k}, @var{sigma}, @var{mu}, @var{p_a}, @var{p_b}, @var{s}, "upper")}
## computes the upper tail probability of the bGEV distribution with given parameters.
##
## If @var{p_a}, @var{p_b}, @var{s} are not given or empty, default values of 
## @var{p_a}=0.05, @var{p_b}=0.2 for positive @var{k} (@var{p_a}=0.95, @var{p_b}=0.8 
## for negative @var{k}), @var{s}=5 are used.
##
## The mean of the bGEV distribution is not finite when @qcode{@var{k} >= 1}, and
## the variance is not finite when @qcode{@var{k} >= 1/2}.  Unlike the GEV distribution
## with positive @var{k}, the bGEV distribution has positive probability density
## over all @var{x}.
##
## The CDF of the bGEV distribution is the same as that of the GEV distribution with the same parameters
## for quantiles above @var{p_b} of that GEV distribution. For quantiles below @var{p_a}, the CDF
## is the same as a Gumbel distribution with the same @var{p_a} and @var{p_b} quantiles. 
## Between @var{p_a} and @var{p_b} is a transition between the Gumbel and GEV distributions.
##
## Uses gevinv, gevcdf, gumbelcdf, betacdf from the Octave Statistics package.
##
## Reference: Castro‐Camilo, D.; Huser, R. & Rue, H. (2022), Practical strategies for generalized extreme value‐based regression models for extremes, Environmetrics, 33, doi: 10.1002/env.2742
##
## @seealso{bgevinv, bgevpdf, gevcdf}
## @end deftypefn

function p = bgevcdf (x, k, sigma, mu, p_a, p_b, s, uflag)

  ## Check for valid number of input arguments
  if (nargin < 4)
    error ("bgevcdf: function called with too few input arguments.");
  endif
  
  ## Set missing parameters to defaults
  if (nargin < 5) || isempty(p_a)
    p_a = 0.05 + (0.9 * (k < 0));
  endif
  if (nargin < 6) || isempty(p_b)
    p_b = 0.2 + (0.6 * (k < 0));
  endif  
  if (nargin < 7) || isempty(s)
    s = 5;
  endif    

  ## Check for valid "upper" flag
  if (nargin > 7)
    if (! strcmpi (uflag, "upper"))
      error ("bgevcdf: invalid argument for upper tail.");
    else
      uflag = true;
    endif
  else
    uflag = false;
  endif

  ## Check for common size of x, k, sigma, mu, p_a, p_b, s
  [retval, x, k, sigma, mu, p_a, p_b, s] = common_size (x, k, sigma, mu, p_a, p_b, s);
  if (retval > 0)
    error ("bgevcdf: X, K, SIGMA, MU, P_A, P_B, and S must be of common size or scalars.");
  endif

  ## Check for x, k, sigma, mu, p_a, p_b, s being reals
  if (iscomplex (x) || iscomplex (k) || iscomplex (sigma) || iscomplex (mu) || iscomplex(p_a) || iscomplex(p_b) || iscomplex(s))
    error ("bgevcdf: X, K, SIGMA, MU, P_A, P_B, and S must not be complex.");
  endif

  #Gumbel distribution parameters
  a = gevinv (p_a, k, sigma, mu);
  b = gevinv (p_b, k, sigma, mu);
  g_sigma = (b - a) ./ log(log(p_a) ./ log(p_b));
  g_mu = a + g_sigma .* log(-log(p_a));
  
  pr = betacdf ((x - a) ./ (b - a), s, s);
  if uflag
    p = (gevcdf(x, k, sigma, mu, "upper") .^ pr) .* (gumbelcdf(x, g_mu, g_sigma, "upper") .^ (1 - pr));
  else
    p = (gevcdf(x, k, sigma, mu) .^ pr) .* (gumbelcdf(x, g_mu, g_sigma) .^ (1 - pr));  
  endif


endfunction

%!demo
%! ## bGEV CDFs when the shape parameter is negative
%! x = -5:0.001:5; p_a = 0.99; p_b = 0.95; s = 5;
%! p1 = bgevcdf (x, 0, 1, 0, p_a, p_b, s);
%! p2 = bgevcdf (x, -0.5, 1, 0, p_a, p_b, s);
%! p3 = bgevcdf (x, -0.75, 1, 0, p_a, p_b, s);
%! p4 = bgevcdf (x, -1, 1, 0, p_a, p_b, s);
%! p5 = bgevcdf (x, -1.5, 1, 0, p_a, p_b, s);
%! plot (x, p1, "-b", x, p2, "-g", x, p3, "-r", ...
%!       x, p4, "-c", x, p5, "-m")
%! grid on
%! xlim ([-5, 5])
%! legend ({"k = 0", "k = -0.5", ...
%!          "k = -0.75", "k = -1", ...
%!          "k = -1.5"}, ...
%!         "location", "northwest")
%! title ("Blended generalized extreme value CDF")
%! xlabel ("values in x")
%! ylabel ("probability")

%!demo
%! ## Plot various CDFs from the blended generalized extreme value distribution
%! x = -1:0.001:10;
%! p1 = bgevcdf (x, 1, 1, 1);
%! p2 = bgevcdf (x, 0.5, 1, 1);
%! p3 = bgevcdf (x, 1, 1, 5);
%! p4 = bgevcdf (x, 1, 2, 5);
%! p5 = bgevcdf (x, 1, 5, 5);
%! p6 = bgevcdf (x, 1, 0.5, 5);
%! plot (x, p1, "-b", x, p2, "-g", x, p3, "-r", ...
%!       x, p4, "-c", x, p5, "-m", x, p6, "-k")
%! grid on
%! xlim ([-1, 10])
%! legend ({"k = 1, σ = 1, μ = 1", "k = 0.5, σ = 1, μ = 1", ...
%!          "k = 1, σ = 1, μ = 5", "k = 1, σ = 2, μ = 5", ...
%!          "k = 1, σ = 5, μ = 5", "k = 1, σ = 0.5, μ = 5"}, ...
%!         "location", "southeast")
%! title ("Blended generalized extreme value CDF")
%! xlabel ("values in x")
%! ylabel ("probability")

## Test output
%!test
%! x = 0:0.5:2.5;
%! sigma = 1:6;
%! k = 1;
%! mu = 0;
%! p = bgevcdf (x, k, sigma, mu);
%! expected_p = [0.36788, 0.44933, 0.47237, 0.48323, 0.48954, 0.49367];
%! assert (p, expected_p, 0.001);
%!test
%! x = -0.5:0.5:2.5;
%! sigma = 0.5;
%! k = 1;
%! mu = 0;
%! p = bgevcdf (x, k, sigma, mu);
%! expected_p = [2.1068e-03, 0.36788, 0.60653, 0.71653, 0.77880, 0.81873, 0.84648];
%! assert (p, expected_p, 0.001);
%! x = 1;
%! sigma = 0.5;
%! k = 0.01:0.01:0.03;
%! mu = 0;
%! p = bgevcdf (x, k, sigma, mu);
%! expected_p = [0.87107, 0.86874, 0.86643];
%! assert (p, expected_p, 0.001);



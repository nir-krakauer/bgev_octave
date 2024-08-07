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
## @deftypefn  {@var{y} =} bgevnll (@var{x}, @var{k}, @var{sigma}, @var{mu}, @var{p_a}, @var{p_b}, @var{s})
##
## Blended generalized extreme value (bGEV) negative log likelihoods.
##
## For each element of @var{x}, compute the negative log likelihood
## under the bGEV distribution with shape parameter @var{k}, scale parameter
## @var{sigma}, location parameter @var{mu}, and blending parameters starting quantile @var{p_a}, 
## ending quantile @var{p_b}, and beta-distribution shape parameter @var{s}.
##
## If @var{p_a}, @var{p_b}, @var{s} are not given or empty, default values of 
## @var{p_a}=0.05, @var{p_b}=0.2 for positive @var{k} (@var{p_a}=0.95, @var{p_b}=0.8 
## for negative @var{k}), @var{s}=5 are used.
##
## The size of @var{y} is the common size of the parameters.  A scalar input
## functions as a constant matrix of the same size as the other inputs.
##
## The result should be the same as @code{-log(bgevpdf(...))} with the same inputs, 
## up to numerical error.
##
## Uses gevinv, gevcdf, gevpdf, gumbelcdf, gumbelpdf, betapdf, betacdf from the Octave Statistics package.
##
## Reference: Castro‐Camilo, D.; Huser, R. & Rue, H. (2022), Practical strategies for generalized extreme value‐based regression models for extremes, Environmetrics, 33, doi: 10.1002/env.2742
##
## @seealso{bgevpdf}
## @end deftypefn

function y = bgevnll (x, k, sigma, mu, p_a, p_b, s)

  ## Check for valid number of input arguments
  if (nargin < 4)
    error ("bgevpdf: function called with too few input arguments.");
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

  ## Check for common size of x, k, sigma, mu, p_a, p_b, s
  [retval, x, k, sigma, mu, p_a, p_b, s] = common_size (x, k, sigma, mu, p_a, p_b, s);
  if (retval > 0)
    error ("bgevpdf: X, K, SIGMA, MU, P_A, P_B, and S must be of common size or scalars.");
  endif

  ## Check for X, K, SIGMA, and MU being reals
  if (iscomplex (x) || iscomplex (k) || iscomplex (sigma) || iscomplex (mu))
    error ("bgevpdf: X, K, SIGMA, and MU must not be complex.");
  endif

  a = gevinv (p_a, k, sigma, mu);
  b = gevinv (p_b, k, sigma, mu);
  #Gumbel distribution parameters  
  g_sigma = (b - a) ./ log(log(p_a) ./ log(p_b));
  g_mu = a + g_sigma .* log(-log(p_a));

  gumbel = ((x <= a) & (k > 0)) | ((x >= a) & (k < 0)); 
  gev = ((x >= b) & (k > 0)) | ((x <= b) & (k < 0));
  mixing = (x > min(a, b)) & (x < max(a, b));

  y = nan (size(x));
  
  if any(gumbel(:))  
    z = (x(gumbel) - g_mu(gumbel)) ./ g_sigma(gumbel);
    y(gumbel) = z + exp(-z) + log(g_sigma(gumbel));
  endif
  
  if any(gev(:))
    y(gev) = gevnll (x(gev), k(gev), sigma(gev), mu(gev));
  endif
  
  if any(mixing(:))
    x = x(mixing);
    a = a(mixing);
    b = b(mixing);
    g_mu = g_mu(mixing);
    g_sigma = g_sigma(mixing);
    s = s(mixing);  
    k = k(mixing);
    mu = mu(mixing);
    sigma = sigma(mixing);  
    pr = betacdf ((x - a) ./ (b - a), s, s);
    pr_d = betapdf((x - a) ./ (b - a), s, s) ./ (b - a);
    term1 = - pr_d .* (1 + k .* (x - mu) ./ sigma) .^ (-1 ./ k);
    term2 = pr ./ sigma .* (1 + k .* (x - mu) ./ sigma) .^ (-1 ./ k - 1);
    term3 = pr_d .* exp(- (x - g_mu) ./ g_sigma);
    term4 = (1 - pr) ./ g_sigma .* exp(- (x - g_mu) ./ g_sigma);
    term0 = pr .* log(gevcdf(x, k, sigma, mu)) + (1 - pr) .* log(gumbelcdf(x, g_mu, g_sigma));    
    y(mixing) = -(term0 + log(term1 + term2 + term3 + term4));
  endif

endfunction

%!demo
%! ## Plot various PDFs from the blended generalized extreme value distribution
%! x = -1:0.001:10;
%! y1 = bgevnll (x, 1, 1, 1);
%! y2 = bgevnll (x, 0.5, 1, 1);
%! y3 = bgevnll (x, 1, 1, 5);
%! y4 = bgevnll (x, -1, 1, 1);
%! y5 = bgevnll (x, -0.5, 1, 1);
%! y6 = bgevnll (x, -1, 1, 5);
%! plot (x, y1, "-b", x, y2, "-g", x, y3, "-r", ...
%!       x, y4, "-c", x, y5, "-m", x, y6, "-k")
%! grid on
%! xlim ([-1, 10])
%! ylim ([-10, 10])
%! legend ({"k = 1, σ = 1, μ = 1", "k = 0.5, σ = 1, μ = 1", ...
%!          "k = 1, σ = 1, μ = 5", "k = -1, σ = 1, μ = 1", ...
%!          "k = -0.5, σ = 1, μ = 1", "k = -1, σ = 1, μ = 5"}, ...
%!         "location", "southwest")
%! title ("Blended generalized extreme value NLL")
%! xlabel ("values in x")
%! ylabel ("negative log likelihood")

## Test output
%!test
%! x = 0:0.5:2.5;
%! sigma = 1:6;
%! k = 1;
%! mu = 0;
%! y = bgevnll (x, k, sigma, mu);
%! expected_y = -log( bgevpdf (x, k, sigma, mu));
%! assert (y, expected_y, 0.001);
%!test
%! x = -0.5:0.5:2.5;
%! sigma = 0.5;
%! k = 1;
%! mu = 0;
%! y = bgevnll (x, k, sigma, mu);
%! expected_y = -log( bgevpdf (x, k, sigma, mu));
%! assert (y, expected_y, 0.001);
%!test
%! x = 1;
%! sigma = 0.5;
%! k = 0.01:0.01:0.03;
%! mu = 0;
%! y = bgevnll (x, k, sigma, mu);
%! expected_y = -log( bgevpdf (x, k, sigma, mu));
%! assert (y, expected_y, 0.001);



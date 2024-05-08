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
## @deftypefn {@var{X} =} bgevinv (@var{P}, @var{k}, @var{sigma}, @var{mu}, @var{p_a}, @var{p_b}, @var{s})
## Compute a desired quantile (inverse CDF) of the blended generalized extreme value (bGEV) distribution.
##
## For each element of @var{P}, compute that quantile
## of the bGEV distribution with positive shape parameter @var{k}, scale parameter
## @var{sigma}, location parameter @var{mu}, and blending parameters starting quantile @var{p_a}, 
## ending quantile @var{p_b}, and beta-distribution shape parameter @var{s}.
##
## If @var{p_a}, @var{p_b}, @var{s} are not given or empty, default values of 
## @var{p_a}=0.05, @var{p_b}=0.2, @var{s}=5 are used.
##
## The size of @var{X} is the common size of the parameters.  A scalar input
## functions as a constant matrix of the same size as the other inputs.
##
## Uses gevinv, gevcdf, gumbelinv, gumbelcdf, betacdf from the Octave Statistics package.
##
## Reference: Castro‐Camilo, D.; Huser, R. & Rue, H. (2022), Practical strategies for generalized extreme value‐based regression models for extremes, Environmetrics, 33, doi: 10.1002/env.2742
##
## @seealso{bgevcdf, bgevpdf, gevinv}
## @end deftypefn

## Author: Nir Krakauer <nkrakauer@ccny.cuny.edu>
## Description: Inverse CDF of the generalized extreme value distribution

function [X] = bgevinv (p, k = 0, sigma = 1, mu = 0, p_a = 0.05, p_b = 0.2, s = 5)

  [retval, p, k, sigma, mu, p_a, p_b, s] = common_size (p, k, sigma, mu, p_a, p_b, s);
  if (retval > 0)
    error ("bgevinv: inputs must be of common size or scalars");
  endif

  gumbel = (p <= p_a);
  frechet = (p >= p_b);
  mixing = (p > p_a) & (p < p_b);

  X = nan (size(p));

  #Gumbel distribution parameters
  a = gevinv (p_a, k, sigma, mu);
  b = gevinv (p_b, k, sigma, mu);
  g_sigma = (b - a) ./ log(log(p_a) ./ log(p_b));
  g_mu = a + g_sigma .* log(-log(p_a));
  
  if any(gumbel(:))  
    X(gumbel) = gumbelinv (p(gumbel), g_mu(gumbel), g_sigma(gumbel));
  endif

  if any(frechet(:))
    X(frechet) = gevinv (p(frechet), k(frechet), sigma(frechet), mu(frechet));
  endif
    
  if any(mixing(:)) #no closed-form expression, but result is between the corresponding Frechet and Gumbel distributions
    p = p(mixing);
    a = a(mixing);
    b = b(mixing);
    g_mu = g_mu(mixing);
    g_sigma = g_sigma(mixing);    
    s = s(mixing);  
    k = k(mixing);
    mu = mu(mixing);
    sigma = sigma(mixing); 
    Xm = X(mixing); 
    
    x_gumbel = gumbelinv (p, g_mu, g_sigma);
    x_frechet = gevinv (p, k, sigma, mu);
     
    n = numel (p);
    for i = 1:n
      function result = p_fun(x, k, sigma, mu, p_a, p_b, s, a, b, g_mu, g_sigma)
        pr = betacdf ((x - a) ./ (b - a), s, s);
        result = (gevcdf(x, k, sigma, mu) .^ pr) .* (gumbelcdf(x, g_mu, g_sigma) .^ (1 - pr));
      endfunction
      fun2 = @(x)  p_fun(x, k(i), sigma(i), mu(i), p_a(i), p_b(i), s(i), a(i), b(i), g_mu(i), g_sigma(i)) - p(i);
      x_range = [x_gumbel(i) x_frechet(i)];
      Xm(i) = fzero (fun2, x_range);
    endfor
    X(mixing) = Xm;
  endif  

endfunction

%!demo
%! ## Plot various iCDFs from the generalized extreme value distribution
%! p = 0.001:0.001:0.999;
%! x1 = bgevinv (p, 1, 1, 1);
%! x2 = bgevinv (p, 0.5, 1, 1);
%! x3 = bgevinv (p, 1, 1, 5);
%! x4 = bgevinv (p, 1, 2, 5);
%! x5 = bgevinv (p, 1, 5, 5);
%! x6 = bgevinv (p, 1, 0.5, 5);
%! plot (p, x1, "-b", p, x2, "-g", p, x3, "-r", ...
%!       p, x4, "-c", p, x5, "-m", p, x6, "-k")
%! ylim ([-1, 20])
%! grid on
%! legend ({"k = 1, σ = 1, μ = 1", "k = 0.5, σ = 1, μ = 1", ...
%!          "k = 1, σ = 1, μ = 5", "k = 1, σ = 2, μ = 5", ...
%!          "k = 1, σ = 5, μ = 5", "k = 1, σ = 0.5, μ = 5"}, ...
%!         "location", "northwest")
%! title ("Blended generalized extreme value iCDF")
%! xlabel ("probability")
%! ylabel ("values in x")

%!test
%! p = 0.1:0.1:0.9;
%! k = 0.1;
%! sigma = 1;
%! mu = 0;
%! x = bgevinv (p, k, sigma, mu);
%! c = bgevcdf(x, k, sigma, mu);
%! assert (c, p, 0.001);

%!test
%! p = 0.1:0.1:0.9;
%! k = 1;
%! sigma = 1;
%! mu = 0;
%! x = bgevinv (p, k, sigma, mu);
%! c = bgevcdf(x, k, sigma, mu);
%! assert (c, p, 0.001);

%!test
%! p = 0.1:0.1:0.9;
%! k = 0.3;
%! sigma = 1;
%! mu = 0;
%! x = bgevinv (p, k, sigma, mu);
%! c = bgevcdf(x, k, sigma, mu);
%! assert (c, p, 0.001);



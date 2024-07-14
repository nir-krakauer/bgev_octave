#bgev_fit_demo

pkg load statistics

file_out = 'bgev_fits.mat'

load bgev_test_data.mat
t = t(:);

[nt, nl] = size (data);

#check ability of bGEV with negative shape parameter to fit annual maximum temperature data and make year-ahead forecasts

nt_start = 30;
nt_fcst = nt - nt_start + 1;
as = 0.975:-0.025:0.75;
bs = as - 0.01;
s = 5; #beta distribution shape parameters
nvals = numel(as); #number of different a, b values to try

fcst_q = fcst_nll = shape_params = nan (nt_fcst-1, nl, nvals+1);

warning('off', 'all', 'local')

for i = 1:(nt_fcst-1)
  x = t(1:(nt_start+i-1));
  xm = mean(x);
  x -= xm;
  xn = t(nt_start+i) - xm;
  disp(i)
  for j = 1:nl    
    y = data(1:(nt_start+i-1), j);
    yn = data(nt_start+i, j);
    #start with regular GEV fit, without and then with the time trend
    gev_const_params = gevfit_lmom (y);
    if !isfinite(sum(gevnll(y, gev_const_params(1), gev_const_params(2), gev_const_params(3))))
      #use a Gumbel distribution as the starting point
      pp = evfit (y);
      gev_const_params = [0, pp(2), pp(1)];
    endif
    gev_nll = @(params) sum (gevnll (y, params(1), exp(params(2)), params(3) + x*params(4)));
    gev_params_init = [gev_const_params(1) log(gev_const_params(2)) gev_const_params(3) 0];
    gev_params = fminunc (gev_nll, gev_params_init);
    params = gev_params;
    fcst_nll(i, j, 1) = gevnll (yn, params(1), exp(params(2)), params(3) + xn*params(4)); 
    fcst_q(i, j, 1) = gevcdf (yn, params(1), exp(params(2)), params(3) + xn*params(4));
    shape_params(i, j, 1) = gev_params(1);
    if gev_params(1) == 0
      fcst_nll(i, j, 2:end) = fcst_nll(i, j, 1); 
      fcst_q(i, j, 2:end) = fcst_q(i, j, 1);
    elseif gev_params(1) < 0
      for k = 1:nvals
        p_a = as(k);
        p_b = bs(k);  
        bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4), p_a, p_b, s));
        bgev_params = fminunc (bgev_nll, gev_params);
        params = bgev_params;
        fcst_nll(i, j, k+1) = bgevnll (yn, params(1), exp(params(2)), params(3) + xn*params(4), p_a, p_b, s); 
        fcst_q(i, j, k+1) = bgevcdf (yn, params(1), exp(params(2)), params(3) + xn*params(4), p_a, p_b, s);  
        shape_params(i, j, k+1) = bgev_params(1);
        if !isfinite(fcst_nll(i, j, k+1))
          disp('surprise, stopping')
          return
        endif
      endfor 
      elseif gev_params(1) > 0 #use default hyperparameters from bgevnll
       bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4)));
       bgev_params = fminunc (bgev_nll, gev_params);
       params = bgev_params;
       fcst_nll(i, j, 2:end) = bgevnll (yn, params(1), exp(params(2)), params(3) + xn*params(4)); 
       fcst_q(i, j, 2:end) = bgevcdf (yn, params(1), exp(params(2)), params(3) + xn*params(4));  
       shape_params(i, j, 2:end) = bgev_params(1);
       if !isfinite(fcst_nll(i, j, 2))
         disp('surprise, stopping') #this should not happen with the bGEV
         return
       endif  
     endif
  endfor
endfor

return

#save output, generate plots

save (file_out, 'fcst_q', 'fcst_nll', 'shape_params', 'nt_start', 'as', 'bs', 's')

plot_dir = 'plots/'


plot(as, sum(sum(fcst_nll, 1), 2)(:)(2:end), '-s')
xlabel('Blending quantile a')
ylabel('Forecast NLL')
print('-deps', [plot_dir 'a_NLL'])

ii = 50;
jj = find(!isfinite(fcst_nll(50, :, 1)));
y = data(1:(nt_start+ii-1), jj);
yn = data(nt_start+ii, jj);
x = t(1:(nt_start+ii-1));
xm = mean(x);
x -= xm;
xn = t(nt_start+ii) - xm;
gev_const_params = gevfit_lmom (y);
gev_nll = @(params) sum (gevnll (y, params(1), exp(params(2)), params(3) + x*params(4)));
gev_params_init = [gev_const_params(1) log(gev_const_params(2)) gev_const_params(3) 0];
gev_params = fminunc (gev_nll, gev_params_init);
a = 0.9; b = 0.89; s = 5;
bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4), a, b, s));
bgev_params = fminunc (bgev_nll, gev_params);

yy = linspace(min(y) - 1, yn + 1, 100);
p_gev = gevpdf (yy, gev_params(1), exp(gev_params(2)), gev_params(3) + xn*gev_params(4));
p_bgev = bgevpdf (yy, bgev_params(1), exp(bgev_params(2)), bgev_params(3) + xn*bgev_params(4), a, b, s);

[nn, xx] = hist(y);
w = xx(2) - xx(1);

set(0, 'DefaultLineLineWidth', 1);
set(0, 'DefaultAxesFontSize', 13);
set(0,'DefaultTextFontSize',13);
set(0,'DefaultAxesFontWeight','bold')
set(0,'DefaultTextFontWeight','bold')
hist(y, xx, 1/w)
hold on
plot(yy, p_gev, ':g', yy, p_bgev)
line ("xdata",[yn, yn], "ydata",[0, 0.5], "linewidth", 1, "color", "k")
xlim([min(yy) max(yy)])
legend('Past', 'GEV', 'bGEV', "location", "northwest")
ylabel("Probability density")
xlabel("Temperature (K)")
print('-depsc', [plot_dir 'example_fcst'])
hold off
#gev_params(3) + xn*gev_params(4) - exp(gev_params(2))/gev_params(1) # GEV upper limit




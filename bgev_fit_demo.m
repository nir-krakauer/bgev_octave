#bgev_fit_demo

#check ability of bGEV with negative shape parameter to fit annual maximum temperature (or sea level) data and make year-ahead forecasts

pkg load statistics




file_out = 'bgev_fits.mat'


load bgev_test_data.mat
t = t(:);
    
data_type = 'T'
switch data_type
  case 'T' #temperatures from ERA5
  
  case 'sl' #sea levels
    data = load ('hilo_sl_ann_max.txt');
    years = data(:, 1);
    t = t((years-1940)+1);
    data = data(:, 2);

endswitch 



[nt, nl] = size (data);



nt_start = 30;
nt_fcst = nt - nt_start + 1;
as = 0.975:-0.025:0.75;
bs = as - 0.01;
s = 5; #beta distribution shape parameters
nvals = numel(as); #number of different a, b values to try

fcst_q = fcst_nll = nan (nt_fcst-1, nl, nvals+2);
shape_params = nan (nt_fcst-1, nl, nvals+1);

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
    pp = evfit (y);
    if !isfinite(sum(gevnll(y, gev_const_params(1), gev_const_params(2), gev_const_params(3))))
      #use a Gumbel distribution as the starting point
      gev_const_params = [0, pp(2), pp(1)];
    endif
    gev_nll = @(params) sum (gevnll (y, params(1), exp(params(2)), params(3) + x*params(4)));
    gev_params_init = [gev_const_params(1) log(gev_const_params(2)) gev_const_params(3) 0];
    gev_params = fminunc (gev_nll, gev_params_init);
    params = gev_params;
    fcst_nll(i, j, 1) = gevnll (yn, params(1), exp(params(2)), params(3) + xn*params(4)); 
    fcst_q(i, j, 1) = gevcdf (yn, params(1), exp(params(2)), params(3) + xn*params(4));
    shape_params(i, j, 1) = gev_params(1);
    
    #bGEV fit
    if nvals > 0
      if gev_params(1) == 0
        fcst_nll(i, j, 2:(nvals+1)) = fcst_nll(i, j, 1); 
        fcst_q(i, j, 2:(nvals+1)) = fcst_q(i, j, 1);
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
         fcst_nll(i, j, 2:(nvals+1)) = bgevnll (yn, params(1), exp(params(2)), params(3) + xn*params(4)); 
         fcst_q(i, j, 2:(nvals+1)) = bgevcdf (yn, params(1), exp(params(2)), params(3) + xn*params(4));  
         shape_params(i, j, 2:(nvals+1)) = bgev_params(1);
         if !isfinite(fcst_nll(i, j, 2))
           disp('surprise, stopping') #this should not happen with the bGEV
            return
         endif  
       endif
      endif
      
     #Gumbel distribution fit for comparison
      gumbel_nll = @(params) sum (gevnll (y, 0, exp(params(1)), params(2) + x*params(3)));
      gumbel_params_init = [pp(2), pp(1), 0];
      gumbel_params = fminunc (gumbel_nll, gumbel_params_init);
      params = gumbel_params;
      fcst_nll(i, j, nvals+2) = gevnll (yn, 0, exp(params(1)), params(2) + xn*params(3)); 
      fcst_q(i, j, nvals+2) = gevcdf (yn,  0, exp(params(1)), params(2) + xn*params(3));          
  endfor
endfor

return

#save output, generate plots

save (file_out, 'fcst_q', 'fcst_nll', 'shape_params', 'nt_start', 'as', 'bs', 's')



#bGEV vs. GEV for simulated values (case A)
rand ("state", pi)
k = -0.2;
sigma = 1;
mu = 0;
n = 100; #values per replicate
m = 100; #number of replicates
x = gevrnd (k, sigma, mu, [n, m]);
params_gev = params_bgev = nan (3, m);
a = 0.91; b = 0.9; s = 5;
params_init = [k; log(sigma); mu];
for i = 1:m
  gev_nll = @(params) sum (gevnll (x(:, i), params(1), exp(params(2)), params(3)));
  params = fminunc (gev_nll, params_init);
  params(2) = exp (params(2));  
  params_gev(:, i) = params;
  bgev_nll = @(params) sum (bgevnll (x(:, i), params(1), exp(params(2)), params(3), a, b, s));
  params = fminunc (bgev_nll, params_init);
  params(2) = exp (params(2));
  params_bgev(:, i) = params;
endfor
gev_upper_limit = params_gev(3, :) - params_gev(2, :) ./ params_gev(1, :); #median: 4.6976 (compared to 5 actual)
gev_upper_limit(params_gev(1, :) >= 0) = Inf;
gev_upper_loss = gevcdf (gev_upper_limit, k, sigma, mu, "upper");
bgev_upper_prob = bgevcdf (median(gev_upper_limit), params_bgev(1, :), params_bgev(2, :), params_bgev(3, :), a, b, s, "upper"); #mean: 1.1556e-03, median: 7.6243e-04
correct_upper_prob = gevcdf (median(gev_upper_limit), k, sigma, mu, "upper"); #8.0948e-07

nb = 100;
[nn, xx] = hist(x(:), nb);
w = xx(2) - xx(1);
p_gev = p_bgev = zeros(nb, 1);
yy = linspace (xx(1), mu - sigma/k, nb);
for i = 1:m
  p_gev += gevpdf (yy', params_gev(1, i), params_gev(2, i), params_gev(3, i));  
  p_bgev += bgevpdf (yy', params_bgev(1, i), params_bgev(2, i), params_bgev(3, i), a, b, s);
endfor

hist(x(:), xx, 1/w)
hold on
plot(yy', p_gev/m, ':g', yy', p_bgev/m)
#xlim([min(yy) max(yy)])
legend('Samples', 'GEV', 'bGEV', "location", "northwest")
ylabel("Probability density")
#xlabel("Maximum sea level (m)")
hold off
print('-depsc', [plot_dir 'fit_caseA'])

return

#case with no hard upper bound (case B)
noise = 1;
x = gevrnd (k, sigma, mu, [n, m]) + noise*randn([n, m]);
params_gev = params_bgev = nan (3, m);
a = 0.91; b = 0.9; s = 5;
params_init = [k; log(sigma); mu];
for i = 1:m
  gev_nll = @(params) sum (gevnll (x(:, i), params(1), exp(params(2)), params(3)));
  params = fminunc (gev_nll, params_init);
  params(2) = exp (params(2));  
  params_gev(:, i) = params;
  bgev_nll = @(params) sum (bgevnll (x(:, i), params(1), exp(params(2)), params(3), a, b, s));
  params = fminunc (bgev_nll, params_init);
  params(2) = exp (params(2));
  params_bgev(:, i) = params;
endfor
gev_upper_limit = params_gev(3, :) - params_gev(2, :) ./ params_gev(1, :); #median: 5.0049
gev_upper_limit(params_gev(1, :) >= 0) = Inf;
bgev_upper_prob = bgevcdf (median(gev_upper_limit), params_bgev(1, :), params_bgev(2, :), params_bgev(3, :), a, b, s, "upper"); #mean: 2.9732e-03, median: 2.4488e-03 (range: 6.9198e-05 - 0.012406)
correct_upper_prob = mean (x(:) >= median(gev_upper_limit)); #1.5e-03

nb = 100;
[nn, xx] = hist(x(:), nb);
w = xx(2) - xx(1);
p_gev = p_bgev = zeros(nb, 1);
for i = 1:m
  p_gev += gevpdf (xx', params_gev(1, i), params_gev(2, i), params_gev(3, i));  
  p_bgev += bgevpdf (xx', params_bgev(1, i), params_bgev(2, i), params_bgev(3, i), a, b, s);
endfor

hist(x(:), xx, 1/w)
hold on
plot(xx', p_gev/m, ':g', xx', p_bgev/m)
#xlim([min(yy) max(yy)])
legend('Samples', 'GEV', 'bGEV', "location", "northwest")
ylabel("Probability density")
#xlabel("Maximum sea level (m)")
hold off
print('-depsc', [plot_dir 'fit_caseB'])


return

plot_dir = 'plots/'


plot(as, sum(sum(fcst_nll, 1), 2)(:)(2:nvals+1), '-s')
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


ii = 21;
jj = 1;
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
a = 0.75; b = 0.74; s = 5;
bgev_nll = @(params) sum (bgevnll (y, params(1), exp(params(2)), params(3) + x*params(4), a, b, s));
bgev_params = fminunc (bgev_nll, gev_params);
yy = linspace(min(y) - 0.1, yn + 0.1, 100);
gev_upper_limit = (gev_params(3) + xn*gev_params(4)) - exp(gev_params(2)) ./ gev_params(1);
p_gev = gevpdf (yy, gev_params(1), exp(gev_params(2)), gev_params(3) + xn*gev_params(4));
p_bgev = bgevpdf (yy, bgev_params(1), exp(bgev_params(2)), bgev_params(3) + xn*bgev_params(4), a, b, s);
set(0, 'DefaultLineLineWidth', 1);
set(0, 'DefaultAxesFontSize', 13);
set(0,'DefaultTextFontSize',13);
set(0,'DefaultAxesFontWeight','bold')
set(0,'DefaultTextFontWeight','bold')
[nn, xx] = hist(y);
w = xx(2) - xx(1);
hist(y, xx, 1/w)
hold on
plot(yy, p_gev, ':g', yy, p_bgev)
line ("xdata",[yn, yn], "ydata",[0, 10], "linewidth", 1, "color", "k")
xlim([min(yy) max(yy)])
legend('Past', 'GEV', 'bGEV', "location", "northwest")
ylabel("Probability density")
xlabel("Maximum sea level (m)")
hold off
print('-depsc', [plot_dir 'example_fcst_sl'])








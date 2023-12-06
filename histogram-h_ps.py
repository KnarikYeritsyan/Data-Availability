import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.utilities.lambdify import lambdify
from sympy.abc import x,mu,sigma
from sympy import sqrt,exp,pi
from matplotlib import ticker

# collecting all the h_ps values from the fits in one array
h_ps=[4979,4482,5229,4282,3219,3602,6433,4387,3060,3809,3373,4764,692,2034,3891,2313,2692,3369,2991,3509,2970,2570,1554,6557,6957,3659,3322,2926,3055,6298,5460,4394,1753]
print('h_ps mean: ',np.mean(h_ps))
print('h_ps std: ',np.std(h_ps,ddof=1))

bin_width=1000
hist_h, bins_h = np.histogram(h_ps,np.arange(500, 7500, bin_width),density=True)
avg_array = (bins_h[1::] + bins_h[:-1])/2
x_exp=avg_array
y_exp=hist_h

def gaussian(x, mu, sigma):
    return 1./(sqrt(2.*pi)*sigma)*exp(-((x - mu)/sigma)**2/2)

# stating initial values for parameters
mu_guess=4000
sigma_guess=1000

myfunc = lambdify((x,mu,sigma),gaussian(x,mu*mu_guess,sigma*sigma_guess),'numpy')

# fitting Gaussian to histogram data
pars, pcov = curve_fit(myfunc, x_exp, y_exp,p0=[1,1.5])
perr = np.sqrt(np.diag(pcov))
residuals = y_exp-myfunc(x_exp, *pars)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_exp-np.mean(y_exp))**2)
r_squared = 1 - (ss_res / ss_tot)
mu_error=100 * perr[0] / abs(pars[0])
sigma_error=100 * perr[1] / abs(pars[1])
print('\n','Results from Gaussian fit to histogram data')
print('\N{GREEK SMALL LETTER MU} = ', pars[0]*mu_guess, " (", mu_error, "%)")
print('\N{GREEK SMALL LETTER SIGMA} = ', pars[1]*sigma_guess, " (", sigma_error, "%)")
print('R\N{SUPERSCRIPT TWO} (coefficient of det.) = ', r_squared)

xmin, xmax = min(bins_h)-bin_width/2,max(bins_h)+bin_width/2
xopt = np.linspace(xmin, xmax, 100)
fit_y = myfunc(xopt, *pars)


fig, ax = plt.subplots()
ax.set_xlabel('$h_{ps}$ [ J/mol ]')
# plt.xlabel('$h_{ps}$ [ J/mol ]')
# plt.xlabel('$\sigma$')
ax.set_ylabel('Normalized Probability $\\times 10^3$')
ax.hist(h_ps, bins=bins_h, density=True, alpha=0.6)
ax.scatter(avg_array, hist_h,color='black')
ax.plot(xopt, fit_y, c='r', label='Gaussian fit')
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, pos:'%1.2f'%(y*1e3)))
plt.legend()
plt.tight_layout()
plt.show()
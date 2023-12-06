import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sympy.utilities.lambdify import lambdify
from sympy.abc import x,mu,sigma
from sympy import sqrt,exp,pi
from matplotlib import ticker
import matplotlib

# collecting all the h values from the fits in one array
h=[5153,4662,5518,4624,3837,4140,7215,5051,3668,4641,4324,4832,1350,2414,4605,3078,3507,4178,3849,3502,3076,2758,2200,6853,8078,3733,3420,3708,3127,6389,5681,4664,2154]
print('h mean: ',np.mean(h))
print('h std: ',np.std(h,ddof=1))

bin_width=1000
hist_h, bins_h = np.histogram(h,np.arange(1000, 9000, bin_width),density=True)
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
print('\N{GREEK SMALL LETTER MU} = ',pars[0]*mu_guess, " (", mu_error, "%)")
print('\N{GREEK SMALL LETTER SIGMA} = ', pars[1]*sigma_guess, " (", sigma_error, "%)")
print('R\N{SUPERSCRIPT TWO} (coefficient of det.) = ', r_squared)

xmin, xmax = min(bins_h)-bin_width/2,max(bins_h)+bin_width/2
xopt = np.linspace(xmin, xmax, 100)
fit_y = myfunc(xopt, *pars)

font = {'size'   : 15}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['axes.linewidth'] = 1.5

fig, ax = plt.subplots()
ax.set_xlabel('$h$ [ J/mol ]')
ax.set_ylabel('Normalized Probability $\\times 10^3$')
ax.hist(h, bins=bins_h, density=True, alpha=0.6)
ax.plot(xopt, fit_y, c='r', label='Gaussian')
ax.scatter(avg_array, hist_h,color='black',label='histogram',zorder=1)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, pos:'%1.2f'%(y*1e3)))
plt.legend()
plt.tight_layout()
plt.show()
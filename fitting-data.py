# loading python libraries
import numpy as np
from sympy.abc import s,sigma,h,Q,o,p,T,H,P,O	#T_0=O, H=Delta H_0, P=Delta C_p^0, o=t_0, p=h_ps
from sympy.utilities.lambdify import lambdify
from sympy import diff,sqrt,log,exp
import matplotlib.pyplot as plt
from pandas import read_csv
from scipy.optimize import curve_fit
import matplotlib
# loading fitting settings file
import cal_settings

# gas constant
R = 8.3

# 2-state formulas

def delta_H_nu(T,H,O,P):
    return H+P*(T-O)

def delta_G_nu(T,H,O,P):
    return H*(1-T/O)+P*(T-O)-T*P*log(T/O)

def K_nu(T,H,O,P):
    return exp(-delta_G_nu(T,H,O,P)/(R*T))

def theta_u(T,H,O,P):
    return K_nu(T,H,O,P)/(1+K_nu(T,H,O,P))

def Cp_func(T,H,O,P):
    return (delta_H_nu(T,H,O,P)*diff(theta_u(T,H,O,P),T)+P*theta_u(T,H,O,P))/cal_settings.N

# ZB in water formulas

def sigma_func(Q):
    return Q**(-1)

def s_func(T,o,h,p,Q,q):
    return Q**(-1)*((exp(-(h/R)/(T-o))+(exp((p/R)/(T-o))*exp((-h/R)/(T-o))-exp((-h/R)/(T-o)))/q)**(-2)-1)

def theta_func(sigma,s):
    return ((s+sigma)/(1+s+sqrt((1-s)**2+4*sigma*s)))*(1+(2*sigma-1+s)/(sqrt((1-s)**2+4*sigma*s)))

def theta(T,o,h,p,Q,q):
    return theta_func(sigma,s).subs([(sigma,sigma_func(Q)),(s,s_func(T,o,h,p,Q,q))])

def u_1(T,o,p,q):
    return exp(p/(R*T-R*o))/(exp(p/(R*T-R*o))+q-1)

def Cp_func_water(T,o,h,p,Q,q=16):
    return -2*h*diff(theta(T,o,h,p,Q,q),T)-2*p*diff(u_1(T, o, p, q),T)*(1-theta(T, o, h, p, Q, q))+2*p*u_1(T, o, p, q)*diff(theta(T, o, h, p, Q, q),T)

# loading data
data = read_csv(cal_settings.load_file, header=None).values
x_exp, y_exp = data.T

# stating parameters initial values for 2-state
O_guess=x_exp[y_exp == max(y_exp)][0]
H_guess=cal_settings.H_guess
P_guess=cal_settings.P_guess

# stating parameters initial values for ZB in water
o_guess=x_exp[x_exp == min(x_exp)][0]-1
h_guess=cal_settings.h_guess
p_guess=cal_settings.p_guess
Q_guess=cal_settings.Q_guess

xopt = np.arange(x_exp[x_exp == min(x_exp)][0]-2, x_exp[x_exp == max(x_exp)][0]+3, 0.2, dtype=np.float64)
myfunc = lambdify((T,H,O,P),Cp_func(T,H*H_guess,O*O_guess,P*P_guess),'numpy')
myfunc_water = lambdify((T,o,h,p,Q),Cp_func_water(T,o*o_guess,h*h_guess,p*p_guess,Q*Q_guess),'numpy')

# fitting 2-state model
print('\n','2-state model fitting data')
pars, pcov = curve_fit(myfunc, x_exp, y_exp, p0=[1.0,1.0,1.0],  bounds=[[0.01, 0.01, 0.01], [2.0, 2.0, 2.0]])
perr = np.sqrt(np.diag(pcov))
residuals = y_exp-myfunc(x_exp, *pars)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_exp-np.mean(y_exp))**2)
r_squared = 1 - (ss_res / ss_tot)
H_error=100 * perr[0] / abs(pars[0])
O_error=100 * perr[1] / abs(pars[1])
P_error=100 * perr[2] / abs(pars[2])
# printing fitting results
print('T\N{SUBSCRIPT ZERO} = ',pars[1]*O_guess, 'K', "(error = ", O_error, "%)")
print("\N{GREEK CAPITAL LETTER DELTA}H\N{SUBSCRIPT ZERO} = ",pars[0]*H_guess,'J/mol',"(error = ", H_error, "%)")
print("\N{GREEK CAPITAL LETTER DELTA}C\N{LATIN SUBSCRIPT SMALL LETTER P}\N{SUPERSCRIPT ZERO} = ",pars[2]*P_guess,'J/(mol K)',"(error = ", P_error, "%)")
print('R\N{SUPERSCRIPT TWO} (coefficient of det.) = ', r_squared)
yopt = myfunc(xopt,*pars)

# fitting ZB in water model
print('\n','ZB in water model fitting data')
if hasattr(cal_settings, 'p0'):
    pars_water, pcov_water = curve_fit(myfunc_water, x_exp, y_exp, p0=cal_settings.p0,  bounds=[[0.1, 0.01, 0.01, 0.01], [0.99, 2.0, 2.0, 2.0]])
else:
    pars_water, pcov_water = curve_fit(myfunc_water, x_exp, y_exp, p0=[0.75,1.0,1.0,1.0],  bounds=[[0.1, 0.01, 0.01, 0.01], [0.99, 2.0, 2.0, 2.0]])
perr_water = np.sqrt(np.diag(pcov_water))
residuals_water = y_exp-myfunc_water(x_exp, *pars_water)
ss_res_water = np.sum(residuals_water**2)
ss_tot_water = np.sum((y_exp-np.mean(y_exp))**2)
r_squared_water = 1 - (ss_res_water / ss_tot_water)
o_error=100 * perr_water[0] / abs(pars_water[0])
h_error=100 * perr_water[1] / abs(pars_water[1])
p_error=100 * perr_water[2] / abs(pars_water[2])
Q_error=100 * perr_water[3] / abs(pars_water[3])
# printing fitting results
print('t\N{SUBSCRIPT ZERO} = ',  pars_water[0]*o_guess,'K',"(error = ", o_error, "%)")
print("h = ", pars_water[1]*h_guess,'J/mol',"(error = ", h_error, "%)")
print("h\N{LATIN SUBSCRIPT SMALL LETTER P}\N{LATIN SUBSCRIPT SMALL LETTER S} = ", pars_water[2]*p_guess,'J/mol',"(error = ", p_error, "%)")
print("Q = ", pars_water[3]*Q_guess, "(error = ", Q_error, "%)")
print('\N{GREEK SMALL LETTER SIGMA} = ', 1/(pars_water[3]*Q_guess))
print('R\N{SUPERSCRIPT TWO} (coefficient of det.) = ', r_squared_water)
yopt_water = myfunc_water(xopt,*pars_water)

# plot settings
font = {'size'   : 15}
matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('ytick', labelsize=14)
matplotlib.rcParams['axes.linewidth'] = 1.5

# plotting graph
fig, ax = plt.subplots()
plt.xlabel('Temperature [K]',{'fontname':'Arial', 'size':'22'})
plt.ylabel('$ C_p \;[\:J\:mol^{-1}\:K^{-1}]$ per aa',{'fontname':'Arial', 'size':'22'})
plt.scatter(x_exp, y_exp, marker='o', facecolors='none', color='black', s=50)
plt.plot(xopt, yopt_water, 'r-', label='ZB in water', lw=2, zorder=3)
plt.plot(xopt, yopt, 'b--', label='2-state', lw=2)
plt.legend()
plt.tight_layout()
plt.show()
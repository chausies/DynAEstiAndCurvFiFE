#---------------------------------------#
# CurvFiFE polling example for writeup  #
#---------------------------------------#

from __future__ import division, print_function
import matplotlib.pylab as P
from curvfife import CurvFiFE, log, norm_ppf, norm_cdf
from scipy.interpolate import UnivariateSpline as US
from tqdm import trange
P.rc('text', usetex=True)
P.rc('font', family='serif')

P.seed(56513)
N = 100
Ny = 501

x = P.linspace(0, 1, N)

y = 0.28*P.cos(6.2*x + 1.4) + .25*P.sin(9.7*x) + .45
x = P.r_[:100]+1

e = (P.rand(N)<y).reshape(1, -1)

p = P.linspace(0, 1, Ny).reshape(-1, 1)

ldists = log(e*p + (1-e)*(1-p))

C = CurvFiFE()
h, LL = C.feed_data_CV(x, ldists, bounded=True, bar=True)

print("best h value: {}".format(h))
print("ll's:")
for h, ll in sorted(LL.items()):
  print("h: {0:.2f},\tll: {1:.5f}".format(h, ll))

y_hat = P.array(C.get_y_hat(x))
y_low, y_high = map(P.array, C.get_confidence_interval(x))

# P.title("Polling Example")
P.plot(x, y, linewidth=2, label=r"$y(t)$", color='blue')
P.plot(x, y_hat, linewidth=2, label=r"$\widehat{y}(t)$", color='red')
P.fill_between(x, y_low, y_high, label="70\% conf interval", color='purple', alpha=0.5)
P.ylabel(r"\% of voters who say Yes")
P.xlabel(r"Time (days)")

P.legend()

# P.savefig('../icdm_writeup/figs/curvfife_ex.pdf')

P.show()

#---------------------------------------------------#
# Runs DynAEsti on a Masters golf data with 3PL IRF #
#---------------------------------------------------#

# NEEDS TO BE RUN ON Python3

import matplotlib.pylab as P
import torch as T
from dynaesti import DynAEsti, IRF, to_tens
from curvfife import CurvFiFE
from scipy.misc import logsumexp as lse
import hickle
import json
from tqdm import tqdm # for progress bar
P.rc('text', usetex=True)
P.rc('font', family='serif')

seed = 250193

# Load golf_data.json
with open("golf_data.json") as f:
  G = json.load(f)
# Convert keys back to integers
for player in list(G.values()):
  for ys, yv in list(player.items()):
    player[int(ys)] = P.array([ yv[str(i)] for i in range(1, 5) ])
    del player[ys]
# Access data as
# G["Player Name"][year][round, hole]
# Round and hole indexed from 0. 4 rounds, 18 holes.

class Golf_GPCM_IRF(IRF):
  def __init__(self, params, strokes, par):
    super().__init__(params)
    self._strokes = sorted(strokes)
    self._par = par
    self._params0 = P.array(params)
  def log_p(self, theta, resp):
    strokes = self._strokes
    s = len(strokes)
    A = self._params[:(s-1)]
    B = self._params[(s-1):]
    make_tens = T.Tensor in map(type, [A, B, theta, resp])
    if make_tens:
      theta = to_tens(theta)
    d = { stroke: i for i, stroke in enumerate(strokes) }
    resp = P.clip(resp, strokes[0], strokes[-1])
    temp = P.vectorize(d.get)(resp)
    if make_tens:
      resp = to_tens(temp)
    else:
      resp = temp
    lo = [0.0*theta]*s # log odds vs par
    I = P.searchsorted(strokes, self._par)
    for i in range(I+1, s):
      lo[i] = lo[i-1] - A[i-1]*(theta - B[i-1])
    for i in range(I-1, -1, -1):
      lo[i] = lo[i+1] + A[i]*(theta - B[i])
    if make_tens:
      lo = T.stack(lo)
      lpp = lo - T.logsumexp(lo, dim=0, keepdim=True)
      lp = sum([ lpp[i]*to_tens(resp==i) for i in range(s) ])
    else:
      lo = P.stack(lo)
      lpp = lo - lse(lo, 0, keepdims=True)
      lp = sum([ lpp[i]*(resp==i) for i in range(s) ])
    return lp
  def fit_params_from_marginals(self, resps, theta, marginals, 
      **kwargs):
    s = len(self._strokes)
    bounds = (((.01, 4),)*(s-1) + ((None, None),)*(s-1))
    super().fit_params_from_marginals(resps, theta, marginals,
        **kwargs, bounds=bounds, params0=self._params0)

# The par for each of the 18 holes
pars = [4, 5, 4, 3, 4, 3, 4, 5, 4, 4, 4, 3, 5, 4, 5, 3, 4, 4]

strokes = [
    [   3, 4, 5, 6   ], # 1
    [   3, 4, 5, 6, 7], # 2
    [   3, 4, 5, 6   ], # 3
    [2, 3, 4, 5      ], # 4
    [   3, 4, 5, 6   ], # 5
    [2, 3, 4, 5      ], # 6
    [   3, 4, 5, 6   ], # 7
    [   3, 4, 5, 6   ], # 8
    [   3, 4, 5, 6   ], # 9
    [   3, 4, 5, 6   ], # 10
    [   3, 4, 5, 6   ], # 11
    [2, 3, 4, 5, 6   ], # 12
    [   3, 4, 5, 6, 7], # 13
    [   3, 4, 5, 6   ], # 14
    [   3, 4, 5, 6, 7], # 15
    [2, 3, 4, 5      ], # 16
    [   3, 4, 5, 6   ], # 17
    [   3, 4, 5, 6   ], # 18
    ]

if __name__ == "__main__":
  from datetime import datetime
  import os
  import errno

  NOW = datetime.now().strftime("%B_%d_%Y_at_%I%M%p")
  dirname = NOW + "/"
  if not os.path.exists(os.path.dirname(dirname)):
      try:
          os.makedirs(os.path.dirname(dirname))
      except OSError as exc: # Guard against race condition
          if exc.errno != errno.EEXIST:
              raise

  desc = \
  """
  Analyze golf data
  polytomous GPCM
  bounded fuzzy kriging + grafting
  5-fold CV with data being randomly shuffled
  CV based on items being removed, not times
  20000 monte samps
  temp = P.linspace(1, 20, 10)
  hh = 0.9 + (temp+temp**2)*25/420
  hh[-1] = P.inf
  Seed={}
  R = 100
  OOP version
  torch version
  tol=1e-7, eps=1e-3, max_iter=200
  1 runs
  """.format(seed)
  print(desc)

  with open(dirname + "desc.txt", "w") as f:
    f.write(desc)

  B = [[0.0]*(len(st)-1) for st in strokes]
  for ind,b in enumerate(B):
    p = P.searchsorted(strokes[ind], pars[ind])
    for i in range(p, len(b)):
      b[i] = -3.*(i - p + 1)
    for i in range(p-1, -1, -1):
      b[i] = 3.*(p - i)


  items = [ 
      Golf_GPCM_IRF([0.2]*len(B[i]) + B[i], strokes[i], pars[i])
      for i in range(18) 
      ]

  responses = []
  for name in tqdm(sorted(G.keys()), 
      ascii=True, desc="Preprocessing golf data"):
    records = G[name]
    resp = []
    for year, card in records.items():
      for r in range(4):
        for h in range(18):
          e = card[r, h]
          resp.append((year, h, e))
    responses.append(resp)

  # # Initialize from a previous run
  # Theta0_bundles, _ = \
  #     hickle.load("April_14_2019_at_0903PM/round_4.hkl")
  #
  # Theta0 = [ CurvFiFE() for _ in range(len(responses)) ]
  # for C, bundle in zip(Theta0, Theta0_bundles):
  #   C.restore_from_bundle(bundle)

  temp = P.linspace(1, 20, 10)
  hh = 0.9 + (temp+temp**2)*25/420
  hh[-1] = P.inf

  Theta_hat = DynAEsti(items, responses, bounded=False, R=100, k=5, 
      hh=hh,
      bar=True, save_each_round=True, monte_samps=20000, max_iter=200)

  N = 501
  theta = P.linspace(-6, 6, N)
  for h, item in enumerate(items):
    a, b, c = item.get_params()
    p = P.array(item(theta, 1))
    P.plot(theta, p, linewidth=2)
    P.ylabel("Prob. of making par")
    P.xlabel(r"$\theta$ (ability)")
    P.title(
        "Hole {h}. a={a:.2f}, b={b:.2f}, c={c:.2f}"\
        .format(h=h, a=a, b=b, c=c)
        )
    P.show()

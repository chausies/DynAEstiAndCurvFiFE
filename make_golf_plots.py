# NEEDS TO BE RUN ON Python3

import matplotlib.pylab as P
import json
import hickle
from curvfife import CurvFiFE
from analyze_golf_poly import G, pars, Golf_GPCM_IRF, strokes
import torch as T
from tqdm import tqdm # for progress bar
P.rc('text', usetex=True)
P.rc('font', family='serif')

N = 501

# Access data as
# G["Player Name"][year][round, hole]
# Round and hole indexed from 0. 4 rounds, 18 holes.

Theta_bundles, params_hat = \
    hickle.load("April_15_2019_at_1052PM/round_7.hkl")
    # hickle.load("April_15_2019_at_0425PM/round_4.hkl")
    # hickle.load("April_14_2019_at_1042PM/round_14.hkl")


best_names = [name for name in G.keys() if len(G[name])>1]
track = [
    "Seve Ballesteros",
    "Billy Casper",
    "Byron Nelson",
    "Tom Watson",
    "Phil Mickelson",
    "Gene Sarazen",
    "Gary Player",
    "Arnold Palmer",
    "Ben Hogan",
    "Sam Snead",
    "Walter Hagen",
    "Tiger Woods",
    "Jack Nicklaus"
    ]
names = sorted(G.keys())
I = P.searchsorted(names, best_names)
Theta_C = { name: CurvFiFE() for name in best_names }
for i in tqdm(I, ascii=True, desc="Restoring Ability distribution functions"):
  name = names[i]
  bundle = Theta_bundles[i]
  Theta_C[name].restore_from_bundle(bundle)

items = [ 
    Golf_GPCM_IRF(params_hat[i], strokes[i], pars[i])
    for i in range(18) 
    ]
winners = {}
for year in range(1937, 2019, 1):
  best = None
  for name in G.keys():
    if year in G[name]:
      score = G[name][year].sum()
      if (best is None) or (score < best):
        best = score
        winners[year] = name

P.figure(figsize=(5.5, 4))
theta = P.linspace(-6, 6, N)
t = P.linspace(1937-.5, 2018+.5, N)
i1 = 0
i2 = 0
temp = 0
for name, col in [
    ("Arnold Palmer", 'red'), 
    ("Jack Nicklaus", 'orange'), 
    ("Tiger Woods", 'green')
    ]:
  temp += 1
  years = sorted(G[name].keys())
  y1, y2 = years[0], years[-1]
  t = P.linspace(y1-.5, y2+.5, N)
  C = Theta_C[name]
  theta_hat = P.array(C.get_y_hat(t))
  P.plot(t, theta_hat, linewidth=2, color=col,
      label=name)
  label = "Attended Masters" if temp==1 else None
  P.scatter(years, P.array(C.get_y_hat(years)), color='blue', 
      label=label)
  label = "70\% conf interval" if temp==1 else None
  c_lo, c_hi = map(P.array, C.get_confidence_interval(t, 70))
  P.fill_between(t, c_lo, c_hi,
     facecolor='purple', alpha=0.5, label=label)
  P.legend()
# P.title("Ability Trajectories of 3 legendary players")
P.xlabel("Year")
P.ylabel("Ability")
P.ylim([-3, 3])
# P.savefig('../writeup/figs/legend_trajectories.pdf')
P.show()

abils = { y : [] for y in range(1937, 2019) }
for name in tqdm(best_names, desc="Plotting trajectories", ascii=True):
  years = sorted(G[name].keys())
  y1, y2 = years[0], years[-1]
  t = P.linspace(y1-.5, y2+.5, N)
  C = Theta_C[name]
  theta_hat = P.array(C.get_y_hat(t))
  theta_years = P.array(C.get_y_hat(years))
  for i, year in enumerate(years):
    abils[year].append(theta_years[i])
  if name in track:
    i1 += 1
    label = None if i1!=1 else "Top 15 players"
    P.plot(t, theta_hat, linewidth=2, color='black', label=label,
        zorder=10)
    if name in winners.values():
      i2 +=1
      label = None if i2!=1 else "Tournament win (top 15 only)"
      yy = [y for y in years if winners[y]==name]
      P.scatter(yy, P.array(C.get_y_hat(yy)), marker="*", 
          color="green", label=label, zorder=15)
  else:
    P.plot(t, theta_hat, linewidth=1, color='grey', 
        alpha=0.5, zorder=1)
years, med_theta = P.array([ 
    (year, P.median(abils[year]))
    for year in range(1937, 2019) if len(abils[year])>0
    ]).T
P.plot(years, med_theta, linewidth=2, 
    color='red', zorder=20, label="Median ability")
P.legend()
# P.title("Trajectories for abilities of Masters Tournament attendees")
P.xlabel("Year")
P.ylabel("Ability")
P.ylim([-3, 3])
# P.savefig('../writeup/figs/overall_golf_trajectories.pdf')
P.show()

name = "Arnold Palmer"
C = Theta_C[name]
ldists = []
X = []
for y, S in G[name].items():
  for r in range(4):
    for h in range(18):
      X.append(y)
      ldists.append(items[h].get_log_func(S[r, h]))
curr_h = float(C._h)

_, LL = C.feed_data_CV(X, ldists, runs=5, k=2,
    monte_samps=20000, bar=True, hh=[curr_h], force_CV=True)

C._h = curr_h

with open("new_golf_data.json") as f:
  Gnew = json.load(f)
# Convert keys back to integers
for player in list(Gnew.values()):
  for ys, yv in list(player.items()):
    player[int(ys)] = P.array([ yv[str(i)] for i in range(1, 5) ])
    del player[ys]
static_bundles, static_params = \
    hickle.load("April_26_2019_at_1019PM/round_8.hkl")
C = CurvFiFE()
i = P.searchsorted(sorted(Gnew.keys()), name)
C.restore_from_bundle(static_bundles[i])
static_items = [ 
    Golf_GPCM_IRF(static_params[i], strokes[i], pars[i])
    for i in range(18) 
    ]
ldists = []
X = []
for y, S in Gnew[name].items():
  for r in range(4):
    for h in range(18):
      X.append(y)
      ldists.append(static_items[h].get_log_func(S[r, h]))

curr_h = float(C._h)

_, LL_static = C.feed_data_CV(X, ldists, runs=5, k=2,
    monte_samps=20000, bar=True, hh=[curr_h], force_CV=True)

C._h = curr_h

for _ in range(3): print()
print("Comparison of bandwidths for Palmer")
for h, ll in sorted(LL.items()):
  print("h: {0:.2f},\tll: {1:.8f}".format(h, ll))
for h, ll in sorted(LL_static.items()):
  print("h: {0:.2f},\tll: {1:.8f}".format(h, ll))


for h, item in enumerate(items):
  if not (h+1) in [6, 13]:
    continue
  params = P.array(item.get_params())
  s1 = len(params)//2
  a, b = map(list, P.around([params[:s1], params[s1:]], 2))
  i = P.searchsorted(strokes[h], pars[h])
  a.insert(i, "N/A")
  b.insert(i, "N/A")
  p = P.array(item(theta.reshape(-1, 1), [strokes[h]]))
  P.plot(theta, p, linewidth=2)
  P.ylabel("Probability of getting stroke count")
  P.xlabel(r"$\theta$ (ability)")
  P.legend([ 
    "{s} strokes, a={aa}, b={bb}.".format(s=strokes[h][j], aa=a[j], bb=b[j])
    for j in range(len(strokes[h]))
    ])
  P.title(
      "Hole \#{h}, Par {p}"\
      .format(h=h+1, p=pars[h])
      )
  # P.savefig('../writeup/figs/hole_{0}_irf.eps'.format(h+1))
  P.show()

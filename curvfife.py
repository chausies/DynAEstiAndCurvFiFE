from __future__ import division, print_function
import torch as T
import matplotlib.pylab as P
from tqdm import trange, tqdm
from scipy.optimize import fmin_l_bfgs_b as BFGS_min
from scipy.interpolate import CubicSpline as CS
norm_cdf = lambda x : (1+T.erf(x/P.sqrt(2)))/2
norm_ppf = lambda x : P.sqrt(2)*T.erfinv(2*T.clamp(x, 1e-9, 1-1e-9)-1)

def to_tens(a, dims=None):
  """Converts to pytorch tensor with `dims` dimensions if not already"""
  if not (type(a) == T.Tensor):
    a = T.tensor(a, dtype=T.float64)
  if dims is None:
    d = T.tensor(0.0, dtype=T.float64)
  else:
    d = T.zeros((1,)*dims, dtype=T.float64)
  return a.double() + d

def psd_inv(mat):
  u = T.cholesky(mat, upper=False)
  return T.potri(u, upper=False)

def log(x):
  x = to_tens(x)
  return T.log(x + P.spacing(1))

def _kern(t):
  return T.exp(-t**2/2)

def get_percentile(x, dist, p):
  F = T.cumsum(dist, 0)
  F = F/F[-1]
  a, b = P.searchsorted(P.array(F), [min(p), max(p)])
  return to_tens(CS(F[a:b], x[a:b])(p))

class CurvFiFE(object):
  """Get the posterior distribution on a curve given probabilistic
  emissions produced from it.
  
  A curve `y` is unknown. At several points `x`, emissions are produced,
  with probability (density) `dist(yy)` given that the curve is equal to
  `yy` at point `x`. CurvFiFE allows one to find a posterior distribution
  on the curve `y` given all these emission distributions.
  """
  def __init__(self):
    pass

  def feed_data(self, X, ldists, h=0.19, bounded=False, 
    eps=1e-4, y_max=6, Ny=1001, tol=1e-7, dx_min=0.0001, max_iter=200, s=1.0,
    auto_increase_eps=False, eps_0_for_h_inf=True):
    """Finds the posterior distribution of a curve given emission
    distributions.

    Given that, at each `X[i]`, the log emission distribution is 
    `ldists[:, i]`, computes the distribution for the underlying curve
    `y(x)` producing the emissions.

    Parameters
    ----------
    X : length n 1D array-like
      The points at which the emissions were observed. Note that, if
      emissions were observed extremely close together, one might consider
      combining those emissions for increased speed/stability.
    ldists : Ny by n 2D array-like, or length-n list of functions
      `ldists[i,j]` is the LOG (ln) probability (density) of the emission
      given the curve `y` at `X[j]` is equal to `y_discretized[i]`. If
      `bounded` is True, `y_discretized` is `linspace(0, 1, Ny)`, else it
      is `linspace(-y_max, y_max, Ny)`. Else, if a list of functions,
      `ldist[j](yy)` should give the log probability density that a the
      curve `y` at `X[j]` is equal to `yy`. Each function should be able to
      handle pytorch tensor `yy`. NOTE: Make sure `Ny` is sufficiently
      large. For the interval [0,1], `Ny=1001` is a good rule of thumb.
    h: positive float
      Gives the "bandwidth", or roughly how close points have to be to
      start affecting each other significantly. Defaults to 0.19, which is
      a reasonable value if X is bound between 0 and 1. If `h` is infinity,
      then uses static IRT.
    bounded : bool
      Whether the curve `y` is bounded to the range 0 to 1, or if its range
      isn't bounded. In the unbounded case, the prior marginal distribution
      of `y(x)` is taken to be standard gaussian. In the bounded case, the
      marginal distribution is taken to be standard uniform.
    eps : small positive float
      The kriging matrix's diagonal is regularized by adding `eps` (which
      can help a lot with numerical instability). In theoretical terms, if
      there is a true smooth curve, then this says one is allowed to stray
      from it independently at any point with a variance of `eps`. Defaults
      to 1e-4.
    y_max : positive float (recommended >>1)
      In unbounded (gaussian) space, the range of the curve `y`
      will only be considered between `-y_max` and `y_max`. This should be
      large enough so that the curve is never expected to go beyond this.
      Assuming the curve has a prior marginal distribution that's standard
      normal (in unbounded space), 6 standard deviations is the default
      (and probably more than enough).
    Ny : Large positive int
      The discretization factor used for the `y`-axis. If `ldists` is not a
      list of functions, then this parameter is ignored.
    tol : small positive float
      Will keep iteratively grafting until the grafted means+std's change by
      less than `tol`.
    dx_min: small positive float
      Emissions whose x values are less than dx_min apart will be lumped
      together to help with speed and numerical stability. E.g. if multiple
      emissions are observed at the same point `x`, they will all be lumped
      together (with the emission distributions multipled).
    max_iter: positive int or None
      The max number of iterations of grafting before breaking. If None,
      will continue endlessly. 10 iterations for speed, 20 is usually
      enough, 50 is more than enough, 100-200 iterations is overkill, but
      could be useful if `h` is inordinately large (curve is relatively
      static).
    s: positive float
      Gives the "undulation size", or roughly how by how much the curve
      undulates up and down (in transformed [unbounded] space). Recommended
      to always be 1.
    auto_increase_eps: bool
      If `eps` is extremely small, then CurvFiFE might break due to
      numerical instability. If `auto_increase_eps` is True, then CurvFiFE
      will keep retrying with successively doubled `eps` until successful.
      If `eps` becomes greater than 1e-2, then throws the error. Defaults to
      False.
    eps_0_for_h_inf: bool
      If True (default), then `eps` will automatically be set to 1e-9
      (essentially 0) if `h` is infinity, since basically no numerical
      stability is required. Else, if False, `eps` will remain whatever
      value it is regardless of `h`.
    """
    if auto_increase_eps:
      try:
        self.feed_data(X, ldists, h, bounded, 
          eps, y_max, Ny, tol, dx_min, max_iter, 
          s, auto_increase_eps=False)
        return
      except RuntimeError as e:
        if eps>1e-2:
          raise e
        self.feed_data(X, ldists, h, bounded, 
          2*eps, y_max, Ny, tol, dx_min, max_iter, 
          s, auto_increase_eps=True)
        return
    X = to_tens(X, 1)
    assert X.dim() == 1, \
        "X must be 1D array-like. Instead, it is {}D.".format(X.dim())
    if callable(ldists[0]):
      func = True
      n_check = len(ldists)
    else:
      func = False
      ldists = to_tens(ldists, 2)
      assert ldists.dim() == 2, \
          (
              "ldists must be 2D array-like "
              "or a list of functions. Instead, it is {}D."
          )\
          .format(ldists.dim())
      Ny, n_check = ldists.shape
    n, = X.size()
    assert n_check == n, \
        "The size of X and the number of columns in ldists must match. {0}!={1}"\
        .format(n, n_check)
    yy = T.linspace(-y_max, y_max, Ny, dtype=T.float64)
    if func:
      tyy = norm_cdf(yy) if bounded else yy # transformed yy
      ldists = T.stack([ldist(tyy) for ldist in ldists]).t()
    if h == float("inf"): # Static case
      if eps_0_for_h_inf:
        eps = 1e-9
      lD = ldists.sum(1)
      if bounded and not func:
        yy_orig = T.linspace(0, 1, Ny, dtype=T.float64)
        # log of the dists, interpolated over norm_cdf(yy)
        lD = to_tens(
          CS(yy_orig, lD)(norm_cdf(yy))
          )
      D = T.exp(lD - T.logsumexp(lD, 0))
      X = to_tens(X.mean(), 1)
      mu = to_tens(T.matmul(yy, D), 1)
      v = to_tens(T.matmul(yy**2, D) - mu**2, 1)
    else:
      I = T.argsort(X)
      X = X[I]
      ldists = ldists[:, I].t()
      # combine close entries
      x_temp = X.clone()
      ldists_temp = ldists.clone()
      X = [X[0]]
      ldists = [ldists[0, :]]
      x_prev = X[0]
      for x, ldist in zip(x_temp[1:], ldists_temp[1:, :]):
        if abs(x - x_prev) <= dx_min:
          ldists[-1] = ldists[-1] + ldist
        else:
          X.append(x)
          ldists.append(ldist)
          x_prev = x
      X = to_tens(X)
      ldists = T.stack(ldists)
      n, = X.size()
      Ny = ldists.shape[1]
      if bounded and not func:
        yy_orig = T.linspace(0, 1, Ny, dtype=T.float64)
        # log of the dists, interpolated over norm_cdf(yy)
        ldists = to_tens(
          CS(yy_orig, ldists.t())(norm_cdf(yy))
          ).t()
      # Start grafting
      K = s**2 * _kern(
          (X.view(-1, 1) - X.view(1, -1)) / h
          ) + eps*T.eye(n).double()
      Kinv = psd_inv(K)
      mu = T.zeros(n, dtype=T.float64) # means of grafted gaussians
      v = T.ones(n, dtype=T.float64) # vars of grafted gaussians
      mu_last = T.tensor(float('inf'))
      v_last = T.tensor(float('inf'))
      curr_iter = 0
      if max_iter is None:
        max_iter = P.inf
      while (curr_iter < max_iter) and T.sqrt(T.mean(
        (mu_last - mu)**2 + (T.sqrt(v_last) - T.sqrt(v))**2
        )) > tol:
        curr_iter += 1
        mu_last = mu
        v_last = v
        Pi = Kinv + T.diag(1/v)
        # posterior covariance matrix
        S = psd_inv(Pi)
        mu_hat = S.matmul(mu/v) # posterior mean
        mS = S + T.diag(S).view(-1, 1)*S / \
            (v.view(-1, 1) - T.diag(S).view(-1, 1))
        # message variance (message from all other variables to the ith
        # variable)
        mv = T.diag(mS)
        M = T.ones((n, 1), dtype=T.float64)*(mu/v).view(1, -1)
        M[range(n), range(n)] = 0
        # message mean
        mmu = T.sum(mS*M, 1)
        lD = ldists - ((yy.view(1, -1) - mmu.view(-1, 1))
                /T.sqrt(mv.view(-1, 1))
                )**2/2
        lD = lD - T.logsumexp(lD, 1).view(-1, 1) # normalize
        D = T.exp(lD)
        # marginal means of each variable when grafting the other vars
        means = D.matmul(yy) 
        # marginal variances
        variances = T.sum(D*(yy.view(1, -1) - means.view(-1, 1))**2, 1)
        v = 1/(1/variances - 1/mv) # updated variances of grafted gaussians
        v[v<=0] = 1000000.0 # Deleting nonsense variances by setting them
                           # extremely high
        mu = v * (means/variances - mmu/mv) # updated means
    self._x = X
    self._mu = mu
    self._v = v
    self._h = h
    self._s = s
    self._eps = eps
    self._bounded = bounded
    self._helper()

  def feed_data_CV(self, X, ldists, 
      hh=None, k=5, shuffle=True, runs=1, monte_samps=10000, bar=False,
      default_h=None, force_CV=False, **kwargs):
    """The same as feed_data, but chooses bandwidth `h` via k-fold Cross
    Validation.

    The emissions are broken up into `k` sets, with each
    set being used in turn as a test set. The best `h` found is returned.

    Parameters
    ----------
    hh : array-like of positive floats
      An array containing each of the bandwidths `h` to be tested.  The
      default is `(max(X) - min(X)) * [0.17, 0.22, 0.28, 0.34, 0.475,
      infty]`, a decent choice.
    k : positive int
      `k`-fold validation is used. `k` between 5 to 10 is standard.
      Defaults to 5.
    shuffle : boolean
      Whether the `k` sets are chosen randomly, or uniformly interleaved.
      Defaults to True.
    runs : int
      If `shuffle` is True, then k-fold CV is repeated for `runs` runs, and
      the performance of each bandwidth is averaged over those runs. This
      can help to prevent a bad random shuffling from ruining the fitting.
      Defaults to 1. Use 2 or 3 for more consistency.
    monte_samps : large positive int
      How many samples are simulated to see how well the fit does on the
      `k` test sets. 1000 is good for speed, 10000 (the default) is better
      for quality.
    bar : boolean
      If True, then a progress bar will be displayed showing the time until
      completion.
    default_h : positive float
      If all of emissions happen at the same time, it's impossible to know
      how fast or slow the underlying curve `y` changes.  `default_h` gives
      the default bandwidth `h` used to fit these cases. Defaults to
      infinity (i.e. static ability, not dynamic).
    force_CV : boolean (False by default)
      Normally, if `len(hh)` is 1 (or `default_h` is used), then CV will be
      skipped because there's no point. However, if you want to force
      perform CV (e.g. to get the log liklihood performance of a single
      `h`), then set this to True.
    kwargs
      The rest of the key-word arguments one would use for feed_data. These
      can be `bounded`, `eps`, `y_max`, `Ny`, `tol`, `dx_min`, `max_iter`,
      `s`, `auto_increase_eps`, and `eps_0_for_h_inf`. 

    Returns
    -------
    best_h : positive float
      The best bandwidth `h` selected from `hh`
    LL : a dictionary with `len(hh)` entries, or None
      `LL[hh[i]] -> ll_i`, where `ll_i` is the average log-liklihood
      achieved by `hh[i]` in the cross validation. Useful to see the
      performance of different bandwidths. If `force_CV` is False, and if
      `len(hh)==1` or `default_h` is used, then CV is unnecessary, and LL
      will be None.  Each log-liklihood is normalized to be the log of the
      geometric mean probability assigned to a single emission. E.g. if 3
      left-out emissions are assigned probabilities 0.3, 0.4, and 0.6, the
      `ll` will be `(0.3 * 0.4 * 0.6)^(1/3)`.
    """
    if default_h is None:
      default_h = P.inf
    # convert hh to list, whether it's a tensor, numpy array, or list
    # already
    X = to_tens(X, 1)
    assert X.dim() == 1, \
        "X must be 1D array-like. Instead, it is {}D.".format(X.dim())
    if callable(ldists[0]):
      func = True
      n_check = len(ldists)
      Ny = 1001 if (not 'Ny' in kwargs) else kwargs['Ny']
    else:
      func = False
      ldists = to_tens(ldists, 2)
      assert ldists.dim() == 2, \
          (
              "ldists must be 2D array-like "
              "or a list of functions. Instead, it is {}D."
          )\
          .format(ldists.dim())
      Ny, n_check = ldists.shape
    n, = X.size()
    assert n_check == n, \
        "The size of X and the number of columns in ldists must match. {0}!={1}"\
        .format(n, n_check)
    dx_min = 0.0001 if (not 'dx_min' in kwargs) else kwargs['dx_min']
    bounded = False if (not 'bounded' in kwargs) else kwargs['bounded']
    y_max = 6 if (not 'y_max' in kwargs) else kwargs['y_max']
    if max(X) - min(X) <= dx_min:
      hh = [float(default_h)]
    if hh is None:
      hh = ((X.max() - X.min())*to_tens([0.17, 0.22, 0.28, 0.34, 0.475, P.inf]))
    elif len(hh)==1 and (not force_CV):
      self.feed_data(X, ldists, h=hh[0], **kwargs)
      return hh[0], None
    hh = to_tens(hh)
    hh = hh.tolist() 
    I = T.argsort(X)
    X = X[I]
    yy = T.linspace(-y_max, y_max, Ny, dtype=T.float64)
    if bounded:
      yy_orig = T.linspace(0, 1, Ny, dtype=T.float64)
    else:
      yy_orig = T.linspace(-y_max, y_max, Ny, dtype=T.float64)
    if func:
      ldists = P.array(ldists)[I]
      yyt = norm_cdf(yy) if bounded else yy
      lD = T.stack([
        ldist(yyt) for ldist in ldists
        ]).t()
    else:
      lD = ldists[:, I]
      ldists = [ 
          CS(yy_orig, lD[:, i]) for i in range(n)
          ]
      if bounded:
        # log of the dists, interpolated over norm_cdf(yy)
        lD = T.stack([
          to_tens(ldist(norm_cdf(yy))) for ldist in ldists
          ]).t()
    if not shuffle:
      runs = 1 # No point doing multiple runs unless you shuffle
    if k>=n: # Reduce `k` if not enough points `n`
      k = n
      runs = 1 # No point doing multiple runs if k=n
    if bar and runs>1:
      r_bar = trange(runs, ascii=True, desc="Avg-ing independent runs")
    else:
      r_bar = range(runs)
    LL = { h : 0.0 for h in hh }
    for r in r_bar:
      if shuffle:
        perm = T.randperm(n)
      else:
        perm = T.arange(n)
      if bar:
        hh = tqdm(hh, ascii=True, desc="Testing h's")
      kwargs["bounded"] = False
      for h in hh: 
        ll = 0.0 # log-liklihood
        rang = trange(k, ascii=True, desc="Calculating CV_loss") \
            if bar else range(k)
        tot = 0
        for i in rang:
          # training points
          train_inds = perm[[ j for j in range(n) if (j+i)%k!=0 ]]
          # validation points
          val_inds = perm[[ j for j in range(n) if (j+i)%k==0 ]]
          tot += len(val_inds)
          lD_slice = lD[:, train_inds]
          self.feed_data(
              X[train_inds], lD_slice, h=h, **kwargs
              )
          samples = self.get_samples(X[val_inds], 
              k=monte_samps, bounded=bounded)
          ll += T.logsumexp(T.stack([
            to_tens(ldists[j](samps_at_j))
            for j, samps_at_j in zip(val_inds, samples)
            ]).sum(0), 0) - log(monte_samps)
        LL[h] += ll/(tot*runs)
    best_h = max( (ll,h) for h,ll in LL.items() )[1]
    self.feed_data(
        X, lD, h=best_h, **kwargs
        )
    self._bounded = bounded
    return best_h, LL

  def restore_from_bundle(self, bundle):
    """
    Restore this CurvFiFE object from a previously saved `bundle`, without
    having to go through all the computation again.
    """
    self._x = to_tens(bundle['x'])
    self._mu = to_tens(bundle['mu'])
    self._v = to_tens(bundle['v'])
    self._h = bundle['h']
    self._s = bundle['s']
    self._eps = bundle['eps']
    self._bounded = bundle['bounded']
    self._helper()

  def export_to_bundle(self):
    """
    Returns `bundle`, a dictionary that can be stored. Then, one can
    restore this CurvFiFE object from `bundle` without having to go through
    all the computation again.
    """
    bundle = {
        'x': self._x.numpy(),
        'mu': self._mu.numpy(),
        'v': self._v.numpy(),
        'h': float(self._h),
        's': float(self._s),
        'eps': float(self._eps),
        'bounded': bool(self._bounded)
        }
    return bundle

  def get_dist(self, xx):
    """
    Returns the mean `mu` and covariance `S` of `y(xx)`, the curve at
    values `xx` (in unbounded Gaussian space). `mu[i]` is the mean/median
    value of `y(xx[i])`, and `S[i,j]` is the covariance between `y(xx[i])`
    and `y(xx[j])`.
    """
    x = self._x
    h = self._h
    s = self._s
    mu = self._mu
    v = self._v
    eps = self._eps
    w = self._w
    iK1 = self._iK1
    xx = to_tens(xx, 1)
    assert xx.dim() == 1, \
        "xx must be 1D array-like. Instead, it is {}D.".format(xx.dim())
    m, = xx.size()
    K2 = s**2 * _kern(
        (xx.view(-1, 1) - xx.view(1, -1)) / h
        ) + eps*T.eye(m).double()
    K12 = s**2 * _kern(
        (x.view(-1, 1) - xx.view(1, -1)) / h
        )
    W = K12.t().matmul(w)
    S_hat2 = K2 + (W - K12.t()).matmul(iK1).matmul(K12)
    mu_hat2 = W.matmul(mu/v)
    return mu_hat2, S_hat2

  def get_y_hat(self, xx, bounded=None):
    """
    Returns an estimate for the curve `y` at points in `xx`. Uses the
    median of the posterior marginal distribution on `y(xx[i])` as the
    estimate. If `bounded` is None, uses the default `bounded` set during
    initialization.
    """
    if bounded is None:
      bounded = self._bounded
    y_hat, _ = self.get_dist(xx)
    if not bounded:
      return y_hat
    return norm_cdf(y_hat)

  def get_ll(self, xx, yy, bounded=None):
    """
    Returns the log-liklihood that the curve goes through the points
    `(xx[i], yy[i])`. If `bounded` is None, uses the default `bounded` set
    during initialization.
    """
    if bounded is None:
      bounded = self._bounded
    yy = to_tens(yy, 1)
    assert yy.dim() == 1, \
        "yy must be 1D array-like. Instead, it is {}D.".format(yy.dim())
    if bounded:
      assert (yy>=0).all() and (yy<=1).all(), \
          "yy values must be between 0 and 1 if bounded==True"
      yy = norm_ppf(yy)
    y_hat, S_hat = self.get_dist(xx)
    Si = psd_inv(S_hat)
    # log probability in transformed (norm_ppf) space 
    l1 = -(
        (yy-y_hat).matmul(Si).matmul(yy-y_hat)
        + log(e).sum())/2
    # correction factor from transformation
    l2 = -(yy**2).sum()/2
    return l1 - l2

  def get_lmarginals(self, xx, yy, bounded=None):
    """
    Returns `lmarginals`. `lmarginals[i, j]` is the LOG (ln) of the
    marginal probability (density) that `y(xx[j])` equals `yy[i]`. If
    `bounded` is None, uses the default `bounded` set during
    initialization.
    """
    if bounded is None:
      bounded = self._bounded
    yy = to_tens(yy, 1)
    assert yy.dim() == 1, \
        "yy must be 1D array-like. Instead, it is {}D.".format(yy.dim())
    if bounded:
      assert (yy>=0).all() and (yy<=1).all(), \
          "yy values must be between 0 and 1 if bounded==True"
    y_hat, S_hat = self.get_dist(xx)
    v_hat = T.diag(S_hat).view(1, -1)
    y_hat = y_hat.view(1, -1)
    if bounded:
      yy = norm_ppf(yy).view(-1, 1)
      lmarginals = (1/2)*(
          -(yy - y_hat)**2/v_hat + yy**2 - log(v_hat)
          )
    else:
      lmarginals =  -(1/2)*(
          (yy.view(-1, 1)-y_hat)**2/v_hat +
          log(2*P.pi*v_hat)
          )
    return lmarginals

  def get_confidence_interval(self, xx, percent=70, bounded=None, 
      Ny=1001, y_max=6):
    """
    Returns `yy_lower` and `yy_upper` such that, `xx[i]` has a marginal
    `percent` chance to lie between `yy_lower[i]` and `yy_upper[i]`. The
    default 70% confidence interval corresponds to about a 1 standard
    deviation confidence interval. `Ny` is the factor by which the curve
    range is discretized when searching for the confidence interval. If
    this method errors, you should consider increasing Ny 10-fold. If
    `bounded` is None, uses the default `bounded` set during
    initialization. If `bounded` is False, `[-y_max, y_max]` is the range
    searched along.
    """
    if bounded is None:
      bounded = self._bounded
    if bounded:
      yy = T.linspace(0, 1, Ny, dtype=T.float64)
    else:
      yy = T.linspace(-y_max, y_max, Ny, dtype=T.float64)
    marginals = T.exp(self.get_lmarginals(xx, yy, bounded=bounded))
    b = (100 - percent)/200.
    C = T.stack([
        get_percentile(yy, marg, [b, 1-b])
        for marg in marginals.t()
        ])
    return C[:, 0], C[:, 1]

  def get_samples(self, xx, k, bounded=None):
    """
    Returns `samples`, a matrix with `k` columns, where each column is a
    curve sampled from the joint posterior distribution on `y(xx)`. If
    `bounded` is None, uses the default `bounded` set during
    initialization.
    """
    if bounded is None:
      bounded = self._bounded
    y_hat, S_hat = self.get_dist(xx)
    L = T.cholesky(S_hat, upper=False)
    samples = L.matmul(T.randn((len(xx), k), dtype=T.float64)) \
        + y_hat.reshape(-1, 1)
    if bounded:
      return norm_cdf(samples)
    return samples

  def _helper(self):
    """
    Given gaussian emissions (sensor readings) with mean `self._mu[i]` and
    variance `self._v[i]` at `self._x[i]`, precomputes the necessary
    matrices to give the gaussian distribution at any points `xx`.Uses a
    kriging prior with bandwidth `h` and undulation `s` and fuzz-factor
    `eps`.
    """
    x = self._x
    v = self._v
    h = self._h
    s = self._s
    eps = self._eps
    n, = x.size()
    K1 = s**2 * _kern(
        (x.view(-1, 1) - x.view(1, -1)) / h
        ) + eps*T.eye(n).double()
    iK1 = psd_inv(K1)
    S_hat1 = psd_inv(iK1 + T.diag(1/v))
    w = iK1.matmul(S_hat1)
    self._iK1 = iK1
    self._w = w

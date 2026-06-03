from __future__ import division, print_function
import torch as T
import matplotlib.pylab as P
from scipy.stats import pearsonr as corrcoef
from scipy.interpolate import CubicSpline as CS
from tqdm import trange, tqdm
norm_cdf = lambda x : (1+T.erf(x/P.sqrt(2)))/2
norm_ppf = lambda x : P.sqrt(2)*T.erfinv(2*T.clamp(x, 1e-9, 1-1e-9)-1)

device = T.device("cpu")#"cuda" if T.cuda.is_available() else "cpu")
# T.set_default_tensor_type(T.cuda.DoubleTensor)

def to_tens(a, dims=None):
  """Converts to pytorch tensor with `dims` dimensions if not already"""
  if not (type(a) == T.Tensor):
    a = T.tensor(a, dtype=T.double, device=device)
  if dims is None:
    d = T.tensor(0.0, dtype=T.double, device=device)
  else:
    d = T.zeros((1,)*dims, dtype=T.double, device=device)
  return a.to(device=device, dtype=T.double) + d

def slr(x):
  # Smooth Linear Ramp for converting (-inf, inf) to (0, inf). Increases
  # linearly for positive x, and decreases as (1/x) for small x. At x=0,
  # has a value and slope of 1
  out = T.zeros_like(x)
  g = x>=0
  l = x<0
  out[g] = x[g] + 1
  out[l] = 1/(1-x[l])
  return out

def slr_inv(x):
  out = T.zeros_like(x)
  l = x<1
  g = x>=1
  out[g] = x[g] - 1
  out[l] = -1/x[l] + 1
  return out

def h_poly_helper(tt):
  A = T.tensor([
      [1, 0, -3, 2],
      [0, 1, -2, 1],
      [0, 0, 3, -2],
      [0, 0, -1, 1]
      ], dtype=tt[-1].dtype, device=device)
  A = A.t()[(...,) + (None,)*(tt.dim()-1)]
  return (A*tt.unsqueeze(1)).sum(0)
  # return [
  #   sum( A[i, j]*tt[j] for j in range(4) )
  #   for i in range(4) ]

def h_poly(t):
  tt = T.zeros((4,) + t.shape, device=device)
  tt[0] = 1.0
  temp = 1.0
  for i in range(1, 4):
    temp = temp*t
    tt[i] = t
  return h_poly_helper(tt)

def dh_poly(t):
  tt = [0, 1, 2*t, 3*t**2]
  return h_poly_helper(tt)

def H_poly(t):
  tt = [ None for _ in range(4) ]
  tt[0] = t
  for i in range(1, 4):
    tt[i] = tt[i-1]*t*i/(i+1)
  return h_poly_helper(tt)

def interp_func(x, y):
  if y.shape[0]>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if y.shape[0]==1: # in the case of 1 point, treat as constant function
      return y[0] + T.zeros_like(xs)
    I = T.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = h_poly((xs-x[I])/dx)
    return hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
  return f

def interp(x, y, xs):
  return interp_func(x,y)(xs)

def deriv_func(x, y):
  if y.shape[0]>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):
    if y.shape[0]==1:
      return T.zeros_like(xs)
    I = T.searchsorted(x[1:], xs)
    dx = (x[I+1]-x[I])
    hh = dh_poly((xs-x[I])/dx)
    return (
        hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
        )/dx
  return f

def deriv(x, y, xs):
  return deriv_func(x,y)(xs)

def integ_func(x, y):
  if y.shape[0]>1:
    m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
    m = T.cat([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
    Y = T.zeros_like(y)
    Y[1:] = (x[1:]-x[:-1])*(
        (y[:-1]+y[1:])/2 + (m[:-1] - m[1:])*(x[1:]-x[:-1])/12
        )
    Y = Y.cumsum(0)
  def f(xs):
    if y.shape[0]==1:
      return y[0]*(xs - x[0])
    I = T.searchsorted(x[1:].detach(), xs)
    dx = (x[I+1]-x[I])
    hh = H_poly((xs-x[I])/dx)
    return Y[I] + dx*(
        hh[0]*y[I] + hh[1]*m[I]*dx + hh[2]*y[I+1] + hh[3]*m[I+1]*dx
        )
  return f

def integ(x, y, xs):
  return integ_func(x,y)(xs)

def gauss_interp(x, y, xx, eps=1e-2):
  K1 = _kern(
      x.view(-1, 1) - x.view(1, -1)
      ) + eps**2*T.eye(x.shape[0], device=device, dtype=T.double)
  iK1 = psd_inv(K1)
  K12 = _kern(
      x.view(-1, 1) - xx.view(1, -1)
      )
  yy = y.mean() + K12.t().matmul(iK1.matmul(y - y.mean()))
  return yy

def get_stopper(n=200, thresh=0.05, f_subsampling=20, restops=0):
  curr_iter = [0]
  history = P.zeros(n)
  ihist = P.zeros(n)
  rho = ["Calculating..."]
  stops = [0]
  hit_stop = [False]
  def stopQ(loss):
    history[curr_iter[0]%200] = float(loss)
    ihist[curr_iter[0]%200] = curr_iter[0]
    curr_iter[0] += 1
    if (curr_iter[0] >= n) and \
        (curr_iter[0]%f_subsampling==0):
      rho[0] = abs(corrcoef(history, ihist)[0])
      stop = rho[0]<thresh
    else:
      stop = False
    if stop and not hit_stop[0]:
      stops[0] += 1
    hit_stop[0] = stop
    if type(rho[0]) == str:
      prog = rho[0]
    else:
      prog = rho[0] + stops[0]
    return ((stops[0]>restops) and hit_stop[0]), prog
  return stopQ

def psd_inv(mat):
  u = T.linalg.cholesky(mat, upper=False)
  return T.cholesky_inverse(u, upper=False)

def log(x):
  x = to_tens(x)
  return T.log(x + P.spacing(1))

def _kern(t):
  return T.exp(-t**2/2)

def grouper(X, dx_min=None):
  # Yields groups of indices `I` such that `X[I[i]]` is within `dx_+min` of
  # `X[I[j]]` for some `j`.
  I = T.argsort(X).tolist()
  if dx_min is None:
    dx_min = 1e-4*(X.max() - X.min())
  prev = X[I[0]]
  group = []
  for i in I:
    x = X[i]
    if x - prev <= dx_min:
      group.append(i)
    else:
      yield group
      group = [i]
      prev = x
  if group:
    yield group

def get_cents_and_imap(X, dx_min):
  imap = T.zeros(X.shape[0], device=device, dtype=int)
  cents = []
  for i, group in enumerate(grouper(X, dx_min)):
    cents.append(X[group].mean())
    for j in group:
      imap[j] = i
  return to_tens(cents), imap

def get_percentile(x, dist, min_p, max_p):
  F = T.cumsum(dist, 0)
  F = F/F[-1]
  a, b = T.searchsorted(F, T.tensor([min_p, max_p], device=device))
  b = min(b+1, F.shape[0]-1)
  return x[T.stack([a, b])]
  # return to_tens(CS(F[a:b], x[a:b])(p))

class CurvFiFE(object):
  """Get the posterior distribution on a curve given probabilistic
  emissions produced from it.
  
  A curve `y` is unknown. At several points `x`, emissions are produced,
  with probability (density) `dist(yy)` given that the curve is equal to
  `yy` at point `x`. CurvFiFE allows one to find a posterior distribution
  on the curve `y` given all these emission distributions.
  """
  def __init__(self):
    self._optms = None
    self._lh_grad = None
    self.train = None
    self._mu = None
    self._lv = None
    self._lh = None
    self._vhx = None
    self._mean_curve = None 

  def start_training(self, X, h=[None], hx=None, bounded=False, 
    fit_h=None, eps=1e-2, B=512, dr_rate=0.0, h_dr_rate=0.5, tol=1e-7,
    dx_min=None, max_iter=float("inf"), auto_increase_eps=False,
    h_scale=None, h_concentration=1.0, h_dist="rayleigh",
    eps_0_for_h_inf=True, default_h=float("inf"), from_scratch=False,
    frozen=False, opt_state=None):
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
    h: array-like of positive floats or `None`s
      Gives the "bandwidth", or roughly how close points have to be to
      start affecting each other significantly. If `h` is `[float('inf')]`,
      then uses static IRT. If `h[i]` is `None`, then `h[i]` is fitted
      (starting from 1) using leave-1-out CV.
    hx: `None` or array-like of floats
      Gives the locations of the instantaneous bandwidths `h[i]`. If None
      (default), then locations are evenly spread across the range of `X`.
    bounded : bool
      Whether the curve `y` is bounded to the range 0 to 1, or if its range
      isn't bounded. In the unbounded case, the prior marginal distribution
      of `y(x)` is taken to be standard gaussian. In the bounded case, the
      marginal distribution is taken to be standard uniform.
    eps : small positive float
      The kriging matrix's diagonal is regularized by adding `eps` (which
      can help a lot with numerical instability). In theoretical terms, if
      there is a true smooth curve, then this says one is allowed to stray
      from it independently at any point with a variance of `eps**2`.
      Defaults to 1e-2.
    tol : small positive float
      Will keep iteratively grafting until the grafted means+std's change
      by less than `tol`.
    dx_min: small positive float or None
      Emissions whose x values are less than dx_min apart will be lumped
      together to help with speed and numerical stability. E.g. if multiple
      emissions are observed at the same point `x`, they will all be lumped
      together (with the emission distributions multipled). None (default)
      corresponds to 0.01% of the range of x values.
    max_iter: positive int or None
      The max number of iterations of grafting before breaking. If None,
      will continue endlessly. 10 iterations for speed, 20 is usually
      enough, 50 is more than enough, 100-200 iterations is overkill, but
      could be useful if `h` is inordinately large (curve is relatively
      static).
    auto_increase_eps: bool
      If `eps` is extremely small, then CurvFiFE might break due to
      numerical instability. If `auto_increase_eps` is True, then CurvFiFE
      will keep retrying with successively 1.3x larger `eps` until
      successful.  If `eps` becomes greater than 0.1, then throws the
      error. Defaults to False.
    eps_0_for_h_inf: bool
      If True (default), then `eps` will automatically be set to 1e-9
      (essentially 0) if `h` is infinity, since basically no numerical
      stability is required. Else, if False, `eps` will remain whatever
      value it is regardless of `h`.
    """
    if frozen:
      fit_h = False
    elif fit_h is None:
      fit_h = None in h
    X = to_tens(X, 1)
    assert X.dim() == 1, \
        "X must be 1D array-like. Instead, it is {}D.".format(X.dim())
    n = X.shape[0]
    if max_iter is None:
      max_iter = P.inf
    if h[0] == float("inf"): # Static case
      if eps_0_for_h_inf:
        eps = 1e-5
      dx_min = 2*(X.max() - X.min())
    # combine close entries
    X, self._imap = get_cents_and_imap(X, dx_min)
    n = X.shape[0]
    vhx = T.zeros(max(0, len(h)-2), dtype=T.double, device=device)
    if n==1:
      h=T.tensor([default_h], dtype=T.double, device=device) # dummy
    else:
      if hx is None:
        vhx = norm_ppf(T.linspace(0, 1, len(h), device=device)[1:-1])
      else:
        assert len(hx)==len(h), "h and hx need to be same length"
        vhx = norm_ppf(T.linspace(0, 1, len(h), device=device)[1:-1]).requires_grad_(not frozen)
    self._x = X
    if self._vhx is None:
      self._vhx = vhx
    elif vhx.requires_grad:
      self._vhx.requires_grad_(not frozen)
    self._eps = eps
    self._bounded = bounded
    self._h_scale = float(
        (self._x[-1] - self._x[0])
        if (h_scale in [None, 0, False])
        else h_scale
        )
    self._h_concentration = h_concentration
    self._h_dist = h_dist
    # means of grafted gaussians
    if from_scratch or (self._mu is None):
      self._mu = T.zeros(n, dtype=T.double, device=device, requires_grad=(not frozen))
    else:
      self._mu = to_tens(self._mu).requires_grad_(not frozen)
    # log of vars of grafted gaussians
    if from_scratch or (self._lv is None):
      self._lv = T.zeros(n, dtype=T.double, device=device, requires_grad=(not frozen))
    else:
      self._lv = to_tens(self._lv).requires_grad_(not frozen)
    mu_lv_optimizer = T.optim.Adadelta([self._mu, self._lv])
    if not (opt_state is None):
      mu_lv_optimizer.load_state_dict(opt_state[0])
    self._optms = [mu_lv_optimizer]
    if fit_h:
      # log of bandwidth
      if self._lh is None:
        self._lh = log(T.tensor([
          hh if hh is not None else self._h_scale
          for hh in h], dtype=T.double, device=device)).requires_grad_(not frozen)
      else:
        self._lh = to_tens(self._lh).requires_grad_(not frozen)
      to_opt = [self._lh]
      if self._vhx.requires_grad:
        to_opt.append(self._vhx)
      h_optimizer = T.optim.Adadelta(to_opt)
      if (not (opt_state is None)) and (len(opt_state)>1):
        h_optimizer.load_state_dict(opt_state[1])
      self._optms.append(h_optimizer)
    else:
      if self._lh is None:
        self._lh = log(h)
    def train(ldists, scaling=None):
      return self.train_step(ldists=ldists, B=B, dr_rate=dr_rate, 
          h_dr_rate=h_dr_rate, auto_increase_eps=auto_increase_eps,
          scaling=scaling)
    self.train = train

  def take_step(self):
    optms = self._optms
    fit_h = len(optms)==2
    if fit_h:
      fit_vhx = len(optms[1].param_groups[0]['params'])==2
    else:
      fit_vhx = False
    optms[0].step()
    optms[0].zero_grad()
    if fit_h:
      # # If _lh.grad doesn't include variational loss
      # self._lh.grad = self._lh_grad
      if not (self._lh_grad is None):
        self._lh.grad += self._lh_grad
      if fit_vhx:
        self._vhx.grad = self._vhx_grad
      optms[1].step()
      optms[1].zero_grad()
      self._lh_grad = None
      if fit_vhx:
        self._vhx_grad = None

  def finish_training(self):
    self._optms = None
    self._lh_grad = None
    self.train = None
    self._mu = self._mu.detach()
    self._lv = self._lv.detach()
    self._lh = self._lh.detach()
    self._vhx = self._vhx.detach()
    self._helper()

  def feed_data(self, X, ldists, h=[None], hx=None, bounded=False, 
    fit_h=None, eps=1e-2, B=512, dr_rate=0.2, h_dr_rate=0.5, tol=1e-7, dx_min=None, 
    max_iter=float("inf"), auto_increase_eps=False, h_scale=None,
    h_concentration=1.0, h_dist="rayleigh", eps_0_for_h_inf=True, bar=False, 
    default_h=float("inf"), restops=0, from_scratch=False):
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
    ldists : length-n list of functions
      `ldists[j](yy)` should give the log probability density that a the
      curve `y` at `X[j]` is equal to `yy`. Each function should be able to
      handle pytorch tensor `yy`.
    h: array-like of positive floats or `None`s
      Gives the "bandwidth", or roughly how close points have to be to
      start affecting each other significantly. If `h` is `[float('inf')]`,
      then uses static IRT. If `h[i]` is `None`, then `h[i]` is fitted
      (starting from 1) using leave-1-out CV.
    hx: `None` or array-like of floats
      Gives the locations of the instantaneous bandwidths `h[i]`. If None
      (default), then locations are evenly spread across the range of `X`.
    bounded : bool
      Whether the curve `y` is bounded to the range 0 to 1, or if its range
      isn't bounded. In the unbounded case, the prior marginal distribution
      of `y(x)` is taken to be standard gaussian. In the bounded case, the
      marginal distribution is taken to be standard uniform.
    eps : small positive float
      The kriging matrix's diagonal is regularized by adding `eps` (which
      can help a lot with numerical instability). In theoretical terms, if
      there is a true smooth curve, then this says one is allowed to stray
      from it independently at any point with a variance of `eps**2`.
      Defaults to 1e-2.
    tol : small positive float
      Will keep iteratively grafting until the grafted means+std's change
      by less than `tol`.
    dx_min: small positive float or None
      Emissions whose x values are less than dx_min apart will be lumped
      together to help with speed and numerical stability. E.g. if multiple
      emissions are observed at the same point `x`, they will all be lumped
      together (with the emission distributions multipled). None (default)
      corresponds to 0.01% of the range of x values.
    max_iter: positive int or None
      The max number of iterations of grafting before breaking. If None,
      will continue endlessly. 10 iterations for speed, 20 is usually
      enough, 50 is more than enough, 100-200 iterations is overkill, but
      could be useful if `h` is inordinately large (curve is relatively
      static).
    auto_increase_eps: bool
      If `eps` is extremely small, then CurvFiFE might break due to
      numerical instability. If `auto_increase_eps` is True, then CurvFiFE
      will keep retrying with successively 1.3x larger `eps` until
      successful.  If `eps` becomes greater than 0.1, then throws the
      error. Defaults to False.
    eps_0_for_h_inf: bool
      If True (default), then `eps` will automatically be set to 1e-9
      (essentially 0) if `h` is infinity, since basically no numerical
      stability is required. Else, if False, `eps` will remain whatever
      value it is regardless of `h`.
    """
    self.start_training(X, h=h, hx=hx, bounded=bounded, eps=eps, B=B,
        fit_h=fit_h, dr_rate=dr_rate, h_dr_rate=h_dr_rate, tol=tol,
        dx_min=dx_min, max_iter=max_iter,
        auto_increase_eps=auto_increase_eps, h_scale=h_scale,
        h_concentration=h_concentration, h_dist=h_dist,
        eps_0_for_h_inf=eps_0_for_h_inf, default_h=default_h,
        from_scratch=from_scratch)
    curr_iter = 0
    stopQ = get_stopper(restops=restops)
    if bar:
      mybar = tqdm(leave=False)
    while True:
      curr_iter += 1
      variational_loss = self.train(ldists)
      variational_loss.backward()
      self.take_step()
      if curr_iter>max_iter: break
      stp, rho = stopQ(variational_loss.detach())
      if bar:
        mybar.set_postfix(**{
          "Loss": "{:.5e}".format(float(variational_loss.detach())),
          "Progress Rate": rho
          })
        mybar.update()
      if stp: break
    if bar:
      mybar.close()
    self.finish_training()

  def train_step(self, ldists, B=512, dr_rate=0.20, h_dr_rate=0.5, 
      auto_increase_eps=False, scaling=None):
    if scaling is None:
      scaling = 1.0
    if auto_increase_eps:
      try:
        return self.train_step(ldists, B=B, dr_rate=dr_rate, 
            h_dr_rate=h_dr_rate, auto_increase_eps=False, scaling=scaling)
      except RuntimeError as e:
        if self._eps>0.1:
          raise e
        self._eps *= 1.3
        return self.train_step(ldists, B=B, dr_rate=dr_rate, 
            h_dr_rate=h_dr_rate, auto_increase_eps=True, scaling=scaling)
    loss = 0
    X = self._x
    n = len(X)
    fit_h = len(self._optms)==2
    if fit_h:
      fit_vhx = len(self._optms[1].param_groups[0]['params'])==2
    else:
      fit_vhx = False
    bounded = self._bounded
    eps = self._eps
    imap = self._imap
    mu = self._mu + 0
    v = T.zeros(n, device=device, dtype=T.double).fill_(P.inf)
    lh = self._lh + 0
    hx = T.zeros_like(lh)
    hx[0] = X[0]
    hx[-1] = X[-1]
    vhx = self._vhx + 1e-3*T.randn_like(self._vhx) # dither for stability
    hx[1:-1] = (hx[-1] - hx[0])*norm_cdf(vhx.sort()[0]) + hx[0]
    if h_dr_rate is None:
      h_dr_rate = dr_rate
    if n>5 and (not dr_rate==0):
      n_drop = max(1, int(n*dr_rate))
      drop_inds = T.randperm(n, device=device)[:n_drop]
      keep_inds = T.randperm(n, device=device)[n_drop:]
      mu[drop_inds] = 0
      v[keep_inds] = self._lv.exp()[keep_inds]
    else:
      v = self._lv.exp()
    if len(hx)>3 and (not h_dr_rate==0):
      n_drop = max(1, int(len(hx)*h_dr_rate))
      perm = T.randperm(len(hx), device=device)
      drop_inds = perm[:n_drop]
      keep_inds = perm[n_drop:]
      lh[drop_inds] = gauss_interp(hx[keep_inds], lh[keep_inds], 
          hx[drop_inds], eps=eps)
    l = (-lh).exp() # intensity l = 1/h
    m = len(imap)
    L = integ(hx, l, X)
    K = _kern(
        L.view(-1, 1) - L.view(1, -1)
        ) + eps**2*T.eye(n, device=device, dtype=T.double)
    Kinv = psd_inv(K)
    Pi = Kinv + T.diag(1/v)
    # posterior covariance matrix
    S = psd_inv(Pi)
    mu_hat = S.matmul(mu/v) # posterior mean
    # compute loss caused by mismatch with prior distribution
    kld = 0.5*(
        (Kinv.t()*S).sum()
        + mu_hat.matmul(Kinv).matmul(mu_hat) 
        - S.logdet()
        )
    if not (ldists is None):
      # compute loss caused by mismatch with conditional distributions
      z = T.randn(B, m, device=device, dtype=T.double)
      samps = S.diag().sqrt()[imap].view(1, -1)*z + mu_hat[imap].view(1, -1)
      if not (self._mean_curve is None):
        z = T.randn(B, m, device=device, dtype=T.double)
        mcmu, mcS = self._mean_curve.get_dist(X)
        samps = samps + mcS.diag().sqrt()[imap].view(1, -1)*z + mcmu[imap].view(1, -1)
      if bounded:
        samps = norm_cdf(samps)
      cross_ent = -ldists(samps).sum()/B
      variational_loss = cross_ent + kld
    if fit_h:
      # perm = T.randperm(m, device=device)
      # # I, J = imap[perm[:n//4]].unique(), imap[perm[n//4:n//2]]
      # I, J = imap[perm[:n//2]].unique(), imap[perm[n//2:]]
      # n1, n2 = map(len, [I, J])
      # K1 = _kern(
      #     L[I].view(-1, 1) - L[I].view(1, -1)
      #     ) + eps**2*T.eye(n1, device=device, dtype=T.double)
      # iK1 = psd_inv(K1)
      # S_hat1 = psd_inv(iK1 + T.diag(1/v[I]))
      # w = iK1.matmul(S_hat1)
      # K2 = _kern(
      #     L[J].view(-1, 1) - L[J].view(1, -1)
      #     ) + eps**2*T.eye(n2, device=device, dtype=T.double)
      # K12 = _kern(
      #     L[I].view(-1, 1) - L[J].view(1, -1)
      #     )
      # W = K12.t().matmul(w)
      # v_hat2 = K2.diag() + T.einsum('ij,jk,ki->i', W-K12.t(), iK1, K12)
      # mu_hat2 = W.matmul(mu[I]/v[I])
      # z = T.randn(B, len(J), device=device, dtype=T.double)
      # samps = v_hat2.sqrt().view(1, -1)*z + mu_hat2
      # if bounded:
      #   samps = norm_cdf(samps)
      # expanded_samps = T.zeros(B, m, device=device, dtype=T.double) + 0.5
      # expanded_samps[:, J] = samps
      # CV_ll = -ldists(expanded_samps)[:,J].sum()/B
      # if len(hx)==1:
      #   smooth_loss = 0
      # else:
      #   temp = (lh - lh.mean())
      #   hk = _kern(
      #         3*(hx.view(-1, 1)-hx.view(1, -1))/(max(hx)-min(hx))
      #         )+eps**2*T.eye(len(hx), device=device)
      #   ihk = psd_inv(hk)
      #   smooth_loss = (
      #       T.einsum("i,ij,j->", temp, ihk, temp) +
      #       0#hk.logdet()
      #       )/2/len(hx)
      # h_loss = CV_ll + smooth_loss
      # h_loss = variational_loss + (l*1.253)**2/2 - T.log(l)#CV_ll + smooth_loss

      # # need to recompute to put it on separate computation graph
      # l = (-self._lh).exp() 
      l = l * self._h_scale # scale the intensity to match the desired h_scale
      if self._h_dist == "rayleigh":
        l = slr(self._h_concentration*slr_inv(l))
        h_loss = (l*1.253).square()/2 - log(l) # + smooth_loss
      elif self._h_dist == "lognormal":
        h_loss = (log(l)*self._h_concentration).square()/2 + log(l) # + smooth_loss
      loss = loss + h_loss
      # h_loss = h_loss * scaling
      # wrt = [self._lh]
      # if fit_vhx:
      #   wrt.append(self._vhx)
      # grad = T.autograd.grad(h_loss, wrt, retain_graph=False)
      # self._lh_grad = grad[0]
      # if fit_vhx:
      #   self._vhx_grad = grad[1]
    if ldists is None:
      loss = loss + kld
    else:
      loss = loss + variational_loss
    return loss*scaling

  def restore_from_bundle(self, bundle):
    """
    Restore this CurvFiFE object from a previously saved `bundle`, without
    having to go through all the computation again.
    """
    self._x = to_tens(bundle['x'])
    self._mu = to_tens(bundle['mu'])
    self._lv = log(bundle['v'])
    self._lh = log(bundle['h'])
    if 'vhx' in bundle:
      self._vhx = to_tens(bundle['vhx'])
    else:
      self._vhx = T.linspace(0, 1, len(self._lh), dtype=T.double, device=device)[1:-1]
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
        'x': self._x.cpu().numpy(),
        'mu': self._mu.detach().cpu().numpy(),
        'v': self._lv.detach().exp().cpu().numpy(),
        'h': self._lh.detach().exp().cpu().numpy(),
        'vhx': self._vhx.detach().cpu().numpy(),
        'eps': float(self._eps),
        'bounded': bool(self._bounded)
        }
    return bundle

  def get_opt_state(self):
    if not (self._optms is None):
      return [ opt.state_dict() for opt in self._optms ]

  def get_dist(self, xx):
    """
    Returns the mean `mu` and covariance `S` of `y(xx)`, the curve at
    values `xx` (in unbounded Gaussian space). `mu[i]` is the mean/median
    value of `y(xx[i])`, and `S[i,j]` is the covariance between `y(xx[i])`
    and `y(xx[j])`.
    """
    x = self._x
    l = (-self._lh).exp()
    hx = T.zeros_like(l)
    hx[0] = x[0]
    hx[-1] = x[-1]
    if len(hx)>1:
      hx[1:-1] = (hx[-1] - hx[0])*norm_cdf(self._vhx.sort()[0]) + hx[0]
    mu = self._mu
    v = self._lv.exp()
    eps = self._eps
    w = self._w
    iK1 = self._iK1
    xx = to_tens(xx, 1)
    assert xx.dim() == 1, \
        "xx must be 1D array-like. Instead, it is {}D.".format(xx.dim())
    m = xx.shape[0]
    if len(hx)>1:
      L1 = integ(hx, l, x)
      L2 = integ(hx, l, xx)
    else:
      L1 = l*(x - hx)
      L2 = l*(xx - hx)
    K2 = _kern(
        L2.view(-1, 1) - L2.view(1, -1)
        ) + eps**2*T.eye(m, device=device, dtype=T.double)
    K12 = _kern(
        L1.view(-1, 1) - L2.view(1, -1)
        )
    W = K12.t().matmul(w)
    S_hat2 = K2 + (W - K12.t()).matmul(iK1).matmul(K12)
    mu_hat2 = W.matmul(mu/v)
    if not (self._mean_curve is None):
      mcmu, mcS = self._mean_curve.get_dist(xx)
      mu_hat2 = mu_hat2 + mcmu
      S_hat2 = S_hat2 + mcS
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
        + S_hat.logdet())/2
    # correction factor from transformation
    if bounded:
      l2 = -(yy**2).sum()/2
    else:
      l2 = P.log(2*P.pi)*yy.shape[0]/2
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
      lmarginals = -(1/2)*(
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
      yy = T.linspace(0, 1, Ny, dtype=T.double, device=device)
    else:
      yy = T.linspace(-y_max, y_max, Ny, dtype=T.double, device=device)
    marginals = T.exp(self.get_lmarginals(xx, yy, bounded=bounded))
    b = (100 - percent)/200.
    C = T.stack([
        get_percentile(yy, marg, b, 1-b)
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
    L = T.linalg.cholesky(S_hat, upper=False)
    samples = L.matmul(T.randn((len(xx), k), dtype=T.double, device=device)) \
        + y_hat.view(-1, 1)
    if bounded:
      return norm_cdf(samples)
    return samples

  def _helper(self):
    """
    Given gaussian emissions (sensor readings) with mean `self._mu[i]` and
    log variance `self._lv[i]` at `self._x[i]`, precomputes the necessary
    matrices to give the gaussian distribution at any points `xx`. Uses a
    kriging prior with bandwidth `h` and undulation `s` and fuzz-factor
    `eps`.
    """
    x = self._x
    v = self._lv.exp()
    l = (-self._lh).exp()
    hx = T.zeros_like(l)
    hx[0] = x[0]
    hx[-1] = x[-1]
    hx[1:-1] = (hx[-1] - hx[0])*norm_cdf(self._vhx.sort()[0]) + hx[0]
    eps = self._eps
    n = x.shape[0]
    L = integ(hx, l, x)
    K1 = _kern(
        L.view(-1, 1) - L.view(1, -1)
        ) + eps**2*T.eye(n, device=device, dtype=T.double)
    iK1 = psd_inv(K1)
    S_hat1 = psd_inv(iK1 + T.diag(1/v))
    w = iK1.matmul(S_hat1)
    self._iK1 = iK1
    self._w = w

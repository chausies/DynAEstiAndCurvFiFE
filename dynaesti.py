from __future__ import division, print_function
import matplotlib.pylab as P
import torch as T
from curvfife import CurvFiFE
from tqdm import trange, tqdm
import hickle
from scipy.optimize import fmin_l_bfgs_b as BFGS_min
norm_cdf = lambda x : (1+T.erf(x/P.sqrt(2)))/2
norm_ppf = lambda x : P.sqrt(2)*T.erfinv(2*T.clamp(x, 1e-9, 1-1e-9)-1)

from datetime import datetime
import os
import errno

def to_tens(a, dims=None):
  """Converts to pytorch tensor with `dims` dimensions if not already"""
  if not (type(a) == T.Tensor):
    a = T.tensor(a, dtype=T.float64)
  if dims is None:
    d = T.tensor(0.0, dtype=T.float64)
  else:
    d = T.zeros((1,)*dims, dtype=T.float64)
  return a.double() + d

def trapz(y, x):
  y_avg = (y[1:, :] + y[:-1, :])/2
  dx = x[1:] - x[:-1]
  return dx.matmul(y_avg)

def log(x):
  x = to_tens(x)
  return T.log(x + P.spacing(1))

class IRF(object):
  """Abstract class detailing the structure of an IRF"""

  def __init__(self, params):
    """Initialize the IRF with its params"""
    self._params = to_tens(params)

  def __call__(self, theta, resp):
    """Return the IRF evaluated at `theta` given item response `resp`.
    
    This returns the probability (density) `p` of observing a response of
    `resp` to this item given the responder has a latent trait of
    `theta`.

    `theta` and `resp` are both scalar or array-like, and broadcasting
    should be respected. E.g. if `theta.shape==(n, 1)` and
    `resp.shape==(1,m)`, then the return should have `p.shape==(n,m)`.
    """
    make_tens = (T.Tensor in map(type, [self._params, theta, resp]))
    if make_tens:
      return T.exp(self.log_p(theta, resp))

  def log_p(self, theta, resp):
    """Return the log of the IRF evaluated at `theta` given item response `resp`.
    
    This returns the LOG probability (density) `p` of observing a response of
    `resp` to this item given the responder has a latent trait of
    `theta`.

    `theta` and `resp` are both scalar or array-like, and broadcasting
    should be respected. E.g. if `theta.shape==(n, 1)` and
    `resp.shape==(1,m)`, then the return should have `p.shape==(n,m)`.
    """
    make_tens = (T.Tensor in map(type, [self._params, theta, resp]))
    if make_tens:
      return T.log(self(theta, resp))

  def get_func(self, resps):
    "Return a function `f` so that `f(theta) == self.__call__(theta, resps)`"
    def f(theta):
      return self(theta, resps)
    return f

  def get_log_func(self, resps):
    "Return a function `f` such that `f(theta) == self.log_p(theta, resps)`"
    def f(theta):
      return self.log_p(theta, resps)
    return f

  def fit_params_from_marginals(self, resps, theta, marginals, 
      bounds=None, params0=None):
    """Update IRF params given the latent traits of responders and their responses.

    Update the parameters given responses and the distributions on the
    latent traits of the responders that produced them. The update should
    be done to maximize the expected log probability (density) of
    responses, where expectation is taken over the distributions in the
    latent traits.

    Provided here is a base implementation.

    Parameters
    ----------
    resps : length n array-like
      The list of item responses by the `n` responders. `resps[i]` is the
      response of the `i`th responder.
    theta : length N array-like of floats
      The values at which the marginal distributions are evaluated at.
    marginals : N by n array-like of floats
      `marginals[j, i]` is the marginal probability (density) that the
      latent trait of responder `i` is equal to `theta[j]`.
    bounds : list of pairs, or else None
      Bounds on the values the parameters can take. `bounds[i]` is of the
      form `(min_value, max_value)`, dictating the min and max respectively
      of `params[i]`. If `min_value` or `max_value` are `None`, there will
      be no boundary in that direction. The default is `bounds` is None,
      which means all params are unbounded when being fit.
    params0 : same type as `params`, else None
      The initial point at which the optimization starts for the
      parameters. If None, uses old params as the starting point.
    """
    marginals = to_tens(marginals)
    theta = to_tens(theta)
    n = len(resps)
    N, n_check = marginals.shape
    assert n == n_check, \
        "resps and marginals aren't the same length. {0}!={1}"\
        .format(n, n_check)
    theta = theta.view(-1, 1)
    resps = to_tens(resps).view(1, -1)
    def loss(params):
      self._params = to_tens(params)
      self._params.requires_grad_(True)
      lp = self.log_p(theta, resps)
      neg_ll = -trapz(marginals*lp, theta.view(-1)).sum()
      neg_ll.backward()
      grad = self._params.grad
      self._params.requires_grad_(False)
      return neg_ll.detach().item(), P.array(grad)
    if params0 is None:
      params0 = self._params
    params_opt, loss_val, d = \
        BFGS_min(loss, params0, bounds=bounds, 
            maxls=100, pgtol=1e-08)
    self._params = to_tens(params_opt)
    return self._params

  def get_params(self):
    """Return the params of the IRF"""
    return self._params

class IRF_3PL(IRF):
  """3 Parameter Logistic IRF
  
  Example usage:
  >>> a = 1
  >>> b = 0.3
  >>> c = 0.1
  >>> irf = IRF_3PL((a, b, c))
  >>> theta = 0.2
  >>> response = 1
  >>> irf(theta, response) == c + (1-c)*1./(1 + P.exp(-a*(theta - b)))
  True
  """

  def __call__(self, theta, resp):
    """Return the 3PL IRF evaluated at `theta` given item response `resp`.
    
    Parameters
    ----------
    theta : scalar or array-like of floats
      The latent traits.
    resp : scalar or array-like of 0's and 1's
      Whether the response was correct or not.

    Returns
    -------
    p : scalar or array-like (following the shape of theta+resp)
      The conditional probability of `resp` given `theta`. Broadcasting is
      respected. E.g. if `theta.shape==(n, 1)` and `resp.shape==(1,m)`,
      then p will have `p.shape==(n,m)`.
    """
    a, b, c = self._params
    make_tens = T.Tensor in map(type, [a, b, c, theta, resp])
    if make_tens:
      theta = to_tens(theta)
    resp_temp = to_tens(resp)
    if make_tens:
      resp = to_tens(resp)
    if make_tens:
      p = c + (1-c)*1./(1 + T.exp(-a*(theta-b)))
    else:
      p = c + (1-c)*1./(1 + P.exp(-a*(theta-b)))
    return resp*p + (1-resp)*(1-p)

  def fit_params_from_marginals(self, resps, theta, marginals, 
      a_min=0.1, a_max=5, c_max=0.30, params0=None):
    """Update IRF params given the latent traits of responders and their responses.

    Update the parameters given responses and the distributions on the
    latent traits of the responders that produced them. The update should
    be done to maximize the expected log probability (density) of
    responses, where expectation is taken over the distributions in the
    latent traits.

    Parameters
    ----------
    resps : length n array-like
      The list of item responses by the `n` responders. `resps[i]` is the
      response of the `i`th responder.
    theta : length N array-like of floats
      The values at which the marginal distributions are evaluated at.
    marginals : N by n array-like of floats
      `marginals[j, i]` is the marginal probability (density) that the
      latent trait of responder `i` is equal to `theta[j]`.
    a_min, a_max, c_max
      The min and maxes for the `a` and `c` parameters. Won't fit anything
      beyond these limits.
    params0 : same type as `params`, else None
      The initial point at which the optimization starts for the
      parameters. If None, uses old params as the starting point.
    """
    bounds = ((a_min, a_max), (None, None), (0, c_max))
    return super().fit_params_from_marginals(resps, theta, marginals, 
        bounds=bounds, params0=params0)

def DynAEsti(items, responses, bounded=False, R=15, Nf=1001, bar=True, 
    default_h=None, save_each_round=False, Theta0=None, **kwargs):
  """Dynamic Ability Estimation 

  Given responders' `responses` to `items` and corresponding `times` that
  they responded, `DynAEsti` outputs a fitted `CurvFiFE` object for each
  responder's ability over time, as well as fits the IRFs for the items.

  Parameters
  ----------
  items : length m list of IRF objects
    The `m` items that responders can respond to. The parameters of each
    IRF should be initialized to their initial guesses. These parameters
    will by optimized by DynAEsti.
  responses : list of n lists of triplets
    The list of the `n` responders' responses to items. `responses[i]` is a
    list of triplets corresponding to responses by the `i`th responder. The
    triplets are of the form `(time, item_index, resp)`. `time` is the time
    they responded to the item.  `item_index` is the index of the item they
    responded to in `items`. `resp` is their response to the item.
  bounded : boolean
    Whether the latent trait should be bounded (between 0 and 1), or
    unbounded (-infty, infty).
  R : positive int
    Number of rounds of EM to run.
  Nf : int
    Controls the amount of discretization for the problem distributions.
  bar : boolean
    Whether you want a progress bar to display tracking progress of
    calculations (highly recommended).
  default_h : positive float
    If all of a responder's responses to items happen at the same time,
    it's impossible to know how fast or slow their latent trait changes.
    `default_h` gives the default bandwidth `h` used to fit these cases.
    If `None`, defaults to the maximum of 0.1, and `0.19 * ((the overall max
    response time) - (the overall min response time))`.
  save_each_round : boolean
    Whether to save the Thetas and IRF params each round to a file or not.
  Theta0 : None (default) or list of n fitted CurvFiFE objects
    If not `None`, then this is the initial guess for the distributions of
    the latent trait curves of the responders. DynAEsti will start with
    fitting the items using these.
  kwargs
    Extra key-word arguments to pass to CurvFiFE when using the
    feed_data_CV method. These can include `hh`, `k`, `monte_samps`,
    `shuffle`, `eps`, `y_max`, `Ny`, `tol`, `dx_min`, `max_iter`, `s`,
    `auto_increase_eps`, and `eps_0_for_h_inf`.  For speed, we recommend
    `k=5` and `monte_samps=1000`. For quality, `k=10` and
    monte_samps=10000`.

  Returns
  -------
  Thetas : length-n list of CurvFiFE objects
    List of CurvFiFE objects for the responders, containing the
    distribution of their latent traits over time. `Theta[i]` is the
    CurvFiFE object for the `i`th student. 
  """
  NOW = datetime.now().strftime("%B_%d_%Y_at_%I%M%p")
  dirname = NOW + "/"
  def savedata(darr, name):
    name = dirname + name + '.hkl'
    hickle.dump(darr, name, mode='w', compression='gzip')
  n = len(responses)
  m = len(items)
  if default_h is None:
    max_time = max( 
        max( trip[0] for trip in trip_list ) 
        for trip_list in responses
        )
    min_time = min( 
        min( trip[0] for trip in trip_list ) 
        for trip_list in responses
        )
    default_h = max(0.1, 0.19 * (max_time - min_time))
  Thetas = [ CurvFiFE() for _ in range(n) ] \
      if Theta0 is None else Theta0
  if not 'y_max' in kwargs:
    kwargs['y_max'] = 6
  if bounded:
    theta = T.linspace(0, 1, Nf)
  else:
    theta = T.linspace(-kwargs['y_max'], kwargs['y_max'], Nf)
  def n_loop_iter(i):
    """Learn the `i`th ability curve"""
    times = to_tens([ triplet[0] for triplet in responses[i] ])
    ldists = [ 
        items[triplet[1]].get_log_func(triplet[2]) 
        for triplet in responses[i]
        ]
    Thetas[i].feed_data_CV(times, ldists, bounded=bounded, bar=bar,
        default_h=default_h, **kwargs) 
  def m_loop_iter(j):
    """Learn the `j`th item parameters"""
    resps = []
    marginals = []
    # collect relevant marginals
    for i in range(n):
      for t, ind, r in responses[i]:
        if ind==j:
          resps.append(r)
          marginals.append(
              T.exp(
                Thetas[i].get_lmarginals(t, theta, bounded=bounded)[:, 0]
                )
              )
    items[j].fit_params_from_marginals(resps, theta, T.stack(marginals).t())
  R_rang = trange(R, ascii=True, desc="EM rounds") if bar else range(R)
  for round_num in R_rang:
    if (Theta0 is None) or (round_num!=0):
      n_rang = trange(n, ascii=True, desc="Fitting ability curves") \
          if bar else range(n)
      for i in n_rang:
        n_loop_iter(i)
    m_rang = trange(m, ascii=True, desc="Fitting Problems") if bar else range(m)
    for j in m_rang:
      m_loop_iter(j)
    if save_each_round:
      if not os.path.exists(os.path.dirname(dirname)):
          try:
              os.makedirs(os.path.dirname(dirname))
          except OSError as exc: # Guard against race condition
              if exc.errno != errno.EEXIST:
                  raise
    if save_each_round:
      Theta_bundles = [Thetas[i].export_to_bundle() for i in range(n)]
      params = [P.array(item.get_params()) for item in items]
      savedata([Theta_bundles, params], 'round_{0}'.format(round_num))
  return Thetas

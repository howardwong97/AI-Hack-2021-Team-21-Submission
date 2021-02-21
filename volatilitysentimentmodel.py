import numpy as np
from scipy import optimize
from scipy import stats

# Params: [c, phi, theta, omega, alpha, beta, gamma, delta]


class VolatilitySentimentModel:
    def __init__(self, log_returns, count, sentiment):
        self.log_returns = log_returns
        self.count = count
        self.sentiment = sentiment
        self.results = None
        self.params = None

    def fit(self):

        r = self.log_returns * 100.0  # scaling helps the optimizer to converge
        e = np.finfo(np.float64).eps

        bounds = [
            (-10 * np.abs(np.mean(r)), 10 * np.abs(np.mean(r))),  # c
            (-1.0 + e, 1 - e),  # phi
            (-1.0 + e, 1 - e),  # theta
            (e, 2 * np.var(r)),  # omega
            (e, 1.0 - e),  # alpha
            (e, 1.0 - e) , # beta
            (None, None),  # gamma
            (None, None)  # delta
        ]

        initial_params = [0.001 for _ in range(3)] + [0.001, 0.1, 0.8, 0.1, 0.1]

        result = optimize.fmin_slsqp(
            func=negative_loglikelihood,
            x0=initial_params,
            f_ieqcons=ineqcons_func,
            bounds=bounds,
            epsilon=1e-6,
            acc=1e-7,
            full_output=True,
            iprint=0,
            args=(r,self.count, self.sentiment),
        )
        self.results = result
        self.params = result[0]

    def predict(self, log_returns, count, sentiment):
        # One-step prediction for conditional variance
        c = self.params[0]
        phi = self.params[1]
        theta = self.params[2]
        omega = self.params[3]
        alpha = self.params[4]
        beta = self.params[5]
        gamma = self.params[6]
        delta = self.params[7]

        eps = get_epsilon(c, phi, theta, log_returns * 100.0)  # log returns scaled by 100 for convergence
        sigma2 = get_sigma2(count, sentiment, omega, alpha, beta, gamma, delta, eps)

        # Scale it back, but squared because variance
        return (omega + alpha * eps[-1] ** 2 + beta * sigma2[-1] + gamma * self.count[-1] + delta * self.sentiment[-1]) / 100.0 ** 2

    def print_summary_stats(self):
        step = 1e-5 * self.params
        T = len(self.log_returns)
        scores = np.zeros((T, len(self.params)))
        scaled_log_returns = self.log_returns * 100.0

        for i in range(len(self.params)):
            h = step[i]
            delta = np.zeros(len(self.params))
            delta[i] = h

            llh_neg = negative_loglikelihood(self.params - delta, scaled_log_returns, self.count, self.sentiment)
            llh_pos = negative_loglikelihood(self.params + delta, scaled_log_returns, self.count, self.sentiment)
            scores[:, i] = (llh_pos - llh_neg) / (2 * h)
            V = (scores.T @ scores) / T
            J = hessian_2sided(negative_loglikelihood, self.params, (scaled_log_returns, self.count, self.sentiment)) / T

        Jinv = np.mat(np.linalg.inv(J))

        asymptotic_variance = np.asarray(Jinv * np.mat(V) * Jinv / T)
        std_err = np.sqrt(np.diag(asymptotic_variance))
        tstats = np.abs(self.params / std_err)
        pvals = [stats.t.sf(np.abs(i), T - 1) * 2 for i in tstats]
        output = np.vstack((self.params, std_err, tstats, pvals)).T

        print('Parameter   Estimate       Std. Err.      T-stat     p-value')
        param = ['c', 'phi', 'theta', 'omega', 'alpha', 'gamma', 'delta']

        for i in range(len(param)):
            print('{0:<11} {1:>0.6f}        {2:0.6f}    {3: 0.5f}    {4: 0.5f}'.format(
                param[i], output[i, 0], output[i, 1], output[i, 2], output[i, 3])
            )


def get_epsilon(c, phi, theta, r):
    T = len(r)
    eps = np.zeros(T)
    for t in range(T):
        if t == 0:
            eps[t] = r[t] - np.mean(r)
        else:
            ar_component = phi * r[t - 1]
            ma_component = theta * eps[t - 1]
            eps[t] = r[t] - c - ar_component - ma_component
    return eps


def get_sigma2(counts, sentiments, omega, alpha, beta, gamma, delta, eps):
    T = len(eps)
    sigma2 = np.zeros(T)
    sigma2[0] = omega / (1 - alpha - beta)

    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1] + gamma * counts[t - 1] + delta * sentiments[t - 1]

    return sigma2


def negative_loglikelihood(params, r, counts, sentiments):
    c = params[0]
    phi = params[1]
    theta = params[2]
    omega = params[3]
    alpha = params[4]
    beta = params[5]
    gamma = params[6]
    delta = params[7]

    eps = get_epsilon(c, phi, theta, r)
    sigma2 = get_sigma2(counts, sentiments, omega, alpha, beta, gamma, delta, eps)

    llh = - 0.5 * (np.log(2 * np.pi) + np.log(sigma2) + eps ** 2 / sigma2)
    neg_llh = -llh  # minimize negative log likelihood
    total_llh = np.sum(neg_llh)

    return total_llh


def ineqcons_func(params, r, counts, sentiments):
    phi = params[1]
    theta = params[2]
    alpha = params[4]
    beta = params[5]
    return [1 - phi, 1 - theta, 1.0 - alpha - beta]


def fit_model(r, counts, sentiments):
    e = np.finfo(np.float64).eps
    bounds = [
        (-10 * np.abs(np.mean(r)), 10 * np.abs(np.mean(r))),  # c
        (-0.9999999999, 0.9999999999),  # phi
        (-0.9999999999, 0.9999999999),  # theta
        (e, 2 * np.var(r)),  # omega
        (e, 1.0 - e),  # alpha
        (e, 1.0 - e),  # beta
        (-1.0, 1.0),  # gamma
        (-1.0, 1.0),  # delta
    ]

    initial_params = [0.001 for _ in range(3)] + [0.001, 0.1, 0.8, 0.1, 0.1]

    result = optimize.fmin_slsqp(
        func=negative_loglikelihood,
        x0=initial_params,
        f_ieqcons=ineqcons_func,
        bounds=bounds,
        epsilon=1e-6,
        acc=1e-7,
        full_output=True,
        iprint=0,
        args=(r, counts, sentiments),

    )

    return result


def one_step_prediction(params, r, counts, sentiments):
    c, phi, theta, omega, alpha, beta, gamma, delta = params
    eps = get_epsilon(c, phi, theta, r)
    sigma2 = get_sigma2(counts, sentiments, omega, alpha, beta, gamma, delta, eps)
    sigma2_pred = omega + alpha * eps[- 1] ** 2 + beta * sigma2[-1] + gamma * counts[-1] + delta * sentiments[-1]

    return sigma2_pred * 0.01**2


def hessian_2sided(fun, theta, args):
    f = fun(theta, *args)
    h = 1e-5 * np.abs(theta)
    thetah = theta + h
    h = thetah - theta
    K = np.size(theta, 0)
    h = np.diag(h)

    fp = np.zeros(K)
    fm = np.zeros(K)
    for i in range(K):
        fp[i] = fun(theta + h[i], *args)
        fm[i] = fun(theta - h[i], *args)

    fpp = np.zeros((K, K))
    fmm = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            fpp[i, j] = fun(theta + h[i] + h[j], *args)
            fpp[j, i] = fpp[i, j]
            fmm[i, j] = fun(theta - h[i] - h[j], *args)
            fmm[j, i] = fmm[i, j]

    hh = (np.diag(h))
    hh = hh.reshape((K, 1))
    hh = hh @ hh.T

    H = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            H[i, j] = (fpp[i, j] - fp[i] - fp[j] + f
                       + f - fm[i] - fm[j] + fmm[i, j]) / hh[i, j] / 2
            H[j, i] = H[i, j]

    return H



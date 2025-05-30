import numpy as np


def gaussian_pdf(x, mu, sigma2):
    coef = 1.0 / np.sqrt(2 * np.pi * sigma2)
    exp_term = np.exp(-0.5 * (x - mu) ** 2 / sigma2)
    return coef * exp_term


def fit_gmm_em(X, K=5, max_iters=100, tol=1e-6):
    N = X.shape[0]

    # Initialize parameters
    np.random.seed(0)
    mu = np.random.choice(X, K)                      # initial estimation of representants
    sigma2 = np.ones(K) * np.var(X)                  # initial estimation of variances.
    pi = np.ones(K) / K                              # initial estimation of wieghts
    gamma = np.zeros((N, K))                         # initial estimation of prob_k(sample)
    #prob_k(sample) is the prob of sample being of class k

    log_likelihood_old = -np.inf

    for _ in range(max_iters):
        # --- E-step ---
        #For each class, estimate the probability of each sample being of class k
        # Basically is the prob of a sample being of class k given our mixture of gaussians.
        for k in range(K):
            gamma[:, k] = pi[k] * gaussian_pdf(X, mu[k], sigma2[k])
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        # --- M-step ---
        #gamma_i_k for sample i, is the probability of sample i of being class k.
        #N_k is the expected value of #samples belonging to class k.
        N_k = np.sum(gamma, axis=0) #number of expected values of #samples for each class k
        pi = N_k / N #By dividing by N, here we compute an estimation of the proportion
                    #of the classes in the data set.
        #Estimation of the mean per class. Computed as the expected value of the samples
        #given the estimation of gamma.
        mu = np.sum(gamma * X[:, np.newaxis], axis=0) / N_k
        #Estimation of the variance.
        sigma2 = np.sum(gamma * (X[:, np.newaxis] - mu)**2, axis=0) / N_k

        # Compute log-likelyhood. Used to determin convergence
        log_likelihood = np.sum(np.log(np.sum([
            pi[k] * gaussian_pdf(X, mu[k], sigma2[k])
            for k in range(K)
        ], axis=0)))

        if np.abs(log_likelihood - log_likelihood_old) < tol:
            break
        log_likelihood_old = log_likelihood

    return mu, sigma2, pi, gamma


class GaussianMixtureKnn:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.mu = None
        self.sigma2 = None
        self.pi = None

    def fit(self, X: np.ndarray) -> "GaussianMixtureKnn":
        self.mu, self.sigma2, self.pi, gamma = fit_gmm_em(X, K= self.k)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        gamma = np.zeros((N, self.k)) 

        for k in range(self.k):
            gamma[:, k] = self.pi[k] * gaussian_pdf(X, self.mu[k], self.sigma2[k])
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return np.argmax(gamma, axis=1)


    def transform(self, X: np.ndarray) -> np.ndarray:
        idcs = self.predict(X)
        return np.array([self.mu[idx] for idx in idcs])

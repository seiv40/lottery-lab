"""
Module 4: Bayesian Model Zoo

Runs 5 Bayesian models to test if lottery drawings are truly random.
Each model tests a different hypothesis about randomness.

Models:
1. Uniform vs. Non-Uniform - Are all numbers equally likely?
2. Temporal Dependence - Does one drawing affect the next?
3. Position Bias - Are certain positions more likely to have high/low numbers?
4. Hierarchical Model - Do numbers cluster?
5. Bayesian Regression - Can features predict outcomes?

The main metric is the Bayes Factor (BF), which tells us how much more likely
one hypothesis is vs another. BF < 0.01 means strong evidence for randomness.

"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats
from scipy.special import gammaln
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import traceback
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# suppress some annoying warnings from PyMC
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# MCMC sampling parameters
# these control how the Bayesian inference works
RANDOM_SEED = 42  # for reproducibility
N_SAMPLES = 2000  # how many posterior samples to draw
N_TUNE = 1000  # how many tuning iterations (warmup)
TARGET_ACCEPT = 0.95  # acceptance rate for MCMC (higher = more accurate but slower)


def _interpret_bayes_factor(bf: float) -> str:
    """
    Interpret Bayes factor using Jeffreys' scale.
    
    This is the standard interpretation guide for Bayes factors.
    BF < 0.01 = decisive evidence for randomness
    BF > 100 = decisive evidence against randomness
    """
    if bf < 1/100:
        return "Decisive evidence for H₀ (randomness)"
    elif bf < 1/10:
        return "Strong evidence for H₀ (randomness)"
    elif bf < 1/3:
        return "Moderate evidence for H₀ (randomness)"
    elif bf < 1:
        return "Weak evidence for H₀ (randomness)"
    elif bf < 3:
        return "Weak evidence for H₁ (non-randomness)"
    elif bf < 10:
        return "Moderate evidence for H₁ (non-randomness)"
    elif bf < 100:
        return "Strong evidence for H₁ (non-randomness)"
    else:
        return "Decisive evidence for H₁ (non-randomness)"


def model1_uniform(game: str, raw_data_dir: Path) -> Dict[str, Any]:
    """
    Model 1: Test if all numbers are equally likely (uniform distribution).
    
    This is the simplest test. If the lottery is truly random, each number
    should appear about the same number of times.
    
    We compare two models:
    - H0 (null): All numbers equally likely (uniform)
    - H1 (alternative): Numbers have different probabilities
    
    Uses analytical Dirichlet-Multinomial conjugate prior.
    This means we don't need MCMC - we can calculate exactly.
    """
    print(f"\n--- MODEL 1: Uniform vs. Non-Uniform ({game.upper()}) ---")
    
    start_time = datetime.now()
    
    # load raw drawing data
    json_file = raw_data_dir / f"{game}_current_format.json"
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    # how many numbers in the pool?
    K = 69 if game == 'powerball' else 70
    
    # collect all numbers that were drawn
    all_numbers = []
    for drawing in raw_data:
        all_numbers.extend(drawing['regularNumbers'])
    
    # count how many times each number appeared
    counts = np.bincount(all_numbers, minlength=K+1)[1:]  # [1:] to skip 0
    n_total = len(all_numbers)
    
    print(f"  Total numbers drawn: {n_total}")
    print(f"  Number pool size: {K}")
    print(f"  Expected count per number: {n_total/K:.1f}")
    
    # Model 0: Uniform distribution (H₀)
    # if truly random, each number has prob 1/K
    expected_count = n_total / K
    log_ml_uniform = n_total * np.log(1.0/K)
    
    # Model 1: Non-uniform distribution (H₁)
    # using Jeffreys prior (alpha = 0.5 for each number)
    # this is a weakly informative prior that doesn't favor any particular distribution
    alpha = np.ones(K) * 0.5
    
    # calculate marginal likelihood using Dirichlet-Multinomial formula
    # this is the probability of seeing our data under this model
    log_ml_nonuniform = (
        gammaln(np.sum(alpha)) - gammaln(n_total + np.sum(alpha)) +
        np.sum(gammaln(counts + alpha) - gammaln(alpha))
    )
    
    # Bayes Factor: ratio of evidence for H1 vs H0
    # if BF < 1, evidence favors H0 (uniform/random)
    # if BF > 1, evidence favors H1 (non-uniform/biased)
    log_bf = log_ml_nonuniform - log_ml_uniform
    bayes_factor = np.exp(log_bf)
    
    # also do a classical chi-square test for comparison
    # this tests if observed frequencies differ from expected
    chi_squared = np.sum((counts - expected_count)**2 / expected_count)
    p_value = 1 - stats.chi2.cdf(chi_squared, df=K-1)
    
    # calculate posterior distribution (Dirichlet)
    # this tells us the probability distribution for each number's true probability
    posterior_alpha = alpha + counts
    
    # calculate 95% credible intervals for each number's probability
    credible_intervals = {}
    for i in range(K):
        # use beta distribution (marginal of Dirichlet)
        a = posterior_alpha[i]
        b = np.sum(posterior_alpha) - posterior_alpha[i]
        credible_intervals[str(i+1)] = [
            float(stats.beta.ppf(0.025, a, b)),  # 2.5th percentile
            float(stats.beta.ppf(0.975, a, b))   # 97.5th percentile
        ]
    
    # check how many numbers have credible intervals that include uniform prob
    # if most do, that's evidence for uniformity
    uniform_prob = 1/K
    in_ci = sum(1 for i in range(K) 
               if credible_intervals[str(i+1)][0] <= uniform_prob <= credible_intervals[str(i+1)][1])
    
    interpretation = _interpret_bayes_factor(bayes_factor)
    
    print(f"  Bayes Factor (BF₁₀): {bayes_factor:.4f}")
    print(f"  Log BF: {log_bf:.4f}")
    print(f"  Chi-squared test: χ² = {chi_squared:.2f}, p = {p_value:.4f}")
    print(f"  Interpretation: {interpretation}")
    print(f"  Numbers consistent with uniformity: {in_ci}/{K} ({100*in_ci/K:.1f}%)")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "model_name": "uniform_vs_nonuniform",
        "bayes_factor": float(bayes_factor),
        "log_bayes_factor": float(log_bf),
        "chi_squared": float(chi_squared),
        "chi_squared_pvalue": float(p_value),
        "numbers_consistent_uniform": int(in_ci),
        "percent_consistent": float(100 * in_ci / K),
        "interpretation": interpretation,
        "execution_time": execution_time,
        "convergence": {"method": "analytical", "converged": True}
    }


def model2_temporal(game: str, raw_data_dir: Path) -> Dict[str, Any]:
    """
    Model 2: Test if one drawing affects the next (temporal dependence).
    
    If the lottery is truly random, each drawing should be independent.
    We test this using an AR(1) model (autoregressive lag-1):
    
    y_t = alpha + beta * y_{t-1} + epsilon
    
    where y_t is the mean of drawing t.
    
    If beta ≈ 0, then no temporal dependence (random).
    If beta ≠ 0, then one drawing affects the next (not random).
    
    This requires MCMC because we can't calculate this analytically.
    """
    print(f"\n--- MODEL 2: Temporal Dependence ({game.upper()}) ---")
    
    start_time = datetime.now()
    
    # load data
    json_file = raw_data_dir / f"{game}_current_format.json"
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    # calculate mean of each drawing
    # we use the mean because it's a single number that summarizes each drawing
    drawing_means = [np.mean(d['regularNumbers']) for d in raw_data]
    drawing_means = np.array(drawing_means)
    
    # set up AR(1) data
    # y_t is current drawing, y_{t-1} is previous drawing
    y_t = drawing_means[1:]  # current values
    y_t_minus_1 = drawing_means[:-1]  # lagged values
    
    print(f"  Testing {len(y_t)} consecutive draws...")
    print(f"  Mean range: [{y_t.min():.1f}, {y_t.max():.1f}]")
    print(f"  Standard deviation: {y_t.std():.2f}")
    
    # build Bayesian model using PyMC
    with pm.Model() as model:
        # priors
        # alpha is the intercept (should be around 35 for both lotteries)
        alpha = pm.Normal('alpha', mu=35.0, sigma=5.0)
        
        # beta is the key parameter - measures temporal dependence
        # we use a tight prior centered on 0 (no dependence)
        beta = pm.Normal('beta', mu=0.0, sigma=0.2)
        
        # sigma is the noise
        sigma = pm.HalfNormal('sigma', sigma=10.0)
        
        # likelihood: y_t ~ Normal(alpha + beta * y_{t-1}, sigma)
        mu = alpha + beta * y_t_minus_1
        pm.Normal('y', mu=mu, sigma=sigma, observed=y_t)
        
        # run MCMC sampling
        # this takes a few minutes
        trace = pm.sample(
            draws=N_SAMPLES,
            tune=N_TUNE,
            chains=4,  # run 4 parallel chains for convergence checking
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,  # don't show progress bar
            return_inferencedata=True
        )
    
    print("  MCMC sampling complete")
    
    # extract beta samples from posterior
    beta_samples = trace.posterior['beta'].values.flatten()
    
    # check convergence diagnostics
    summary = az.summary(trace, var_names=['beta'])
    rhat_beta = float(summary.loc['beta', 'r_hat'])  # should be < 1.01
    ess_beta = float(summary.loc['beta', 'ess_bulk'])  # should be > 400
    
    # calculate Bayes factor using Savage-Dickey density ratio
    # this compares posterior density at beta=0 vs prior density at beta=0
    prior_density_at_zero = stats.norm.pdf(0, loc=0, scale=0.2)
    posterior_density_at_zero = stats.gaussian_kde(beta_samples).pdf(0)[0]
    bf_01 = posterior_density_at_zero / prior_density_at_zero  # BF in favor of H0
    bf_10 = 1 / bf_01  # BF in favor of H1
    
    # calculate 95% credible interval for beta
    beta_ci = np.percentile(beta_samples, [2.5, 97.5])
    zero_in_ci = beta_ci[0] <= 0 <= beta_ci[1]  # does CI include 0?
    
    interpretation = _interpret_bayes_factor(bf_10)
    
    print(f"  β coefficient:")
    print(f"  Mean: {np.mean(beta_samples):.4f}")
    print(f"  95% CI: [{beta_ci[0]:.4f}, {beta_ci[1]:.4f}]")
    print(f"  Zero in CI: {zero_in_ci}")
    print(f"  Bayes Factor (BF₁₀): {bf_10:.4f}")
    print(f"  Convergence: R̂ = {rhat_beta:.4f}, ESS = {ess_beta:.0f}")
    print(f"  Interpretation: {interpretation}")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "model_name": "temporal_dependence_ar1",
        "beta_mean": float(np.mean(beta_samples)),
        "beta_std": float(np.std(beta_samples)),
        "beta_credible_interval_95": [float(beta_ci[0]), float(beta_ci[1])],
        "zero_in_ci": bool(zero_in_ci),
        "bayes_factor": float(bf_10),
        "interpretation": interpretation,
        "execution_time": execution_time,
        "convergence": {
            "r_hat": rhat_beta,
            "ess": ess_beta,
            "converged": bool(rhat_beta < 1.01)
        }
    }


def model3_position_bias(game: str, raw_data_dir: Path) -> Dict[str, Any]:
    """
    Model 3: Test if certain positions tend to have higher/lower numbers.
    
    In a truly random lottery, all positions (1st ball, 2nd ball, etc.)
    should have the same average number.
    
    We use a hierarchical model to test this:
    - Each position has its own mean
    - These means come from a global distribution
    - If positions are similar, evidence for randomness
    
    This also requires MCMC.
    """
    print(f"\n--- MODEL 3: Position Bias ({game.upper()}) ---")
    
    start_time = datetime.now()
    
    # load data
    json_file = raw_data_dir / f"{game}_current_format.json"
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    K = 69 if game == 'powerball' else 70
    n_positions = 5  # always 5 balls drawn
    
    # separate numbers by position
    # position_data[0] = all first balls, position_data[1] = all second balls, etc.
    position_data = [[] for _ in range(n_positions)]
    for drawing in raw_data:
        for pos in range(n_positions):
            position_data[pos].append(drawing['regularNumbers'][pos])
    
    position_data = [np.array(pos) for pos in position_data]
    
    print(f"  Analyzing {n_positions} positions...")
    print(f"  Testing {K} numbers across {len(raw_data)} drawings...")
    
    # build hierarchical Bayesian model
    with pm.Model() as model:
        # global mean - average across all positions
        mu_global = pm.Normal('mu_global', 
                             mu=np.mean(np.concatenate(position_data)), 
                             sigma=10.0)
        
        # tau controls how different positions can be from the global mean
        # small tau = positions must be similar (evidence for randomness)
        # large tau = positions can differ a lot (evidence for bias)
        tau = pm.HalfNormal('tau', sigma=5.0)
        
        # each position gets its own mean
        mu_position = pm.Normal('mu_position', 
                               mu=mu_global, 
                               sigma=tau, 
                               shape=n_positions)
        
        # observation noise
        sigma = pm.HalfNormal('sigma', sigma=10.0)
        
        # likelihood for each position
        for pos in range(n_positions):
            pm.Normal(f'obs_pos_{pos}', 
                     mu=mu_position[pos], 
                     sigma=sigma,
                     observed=position_data[pos])
        
        # run MCMC
        trace = pm.sample(
            draws=N_SAMPLES,
            tune=N_TUNE,
            chains=4,
            target_accept=TARGET_ACCEPT,
            random_seed=RANDOM_SEED,
            progressbar=False,
            return_inferencedata=True
        )
    
    print("  MCMC sampling complete")
    
    # extract tau (between-position variability)
    tau_samples = trace.posterior['tau'].values.flatten()
    
    # check convergence
    summary = az.summary(trace, var_names=['tau'])
    rhat_tau = float(summary.loc['tau', 'r_hat'])
    ess_tau = float(summary.loc['tau', 'ess_bulk'])
    
    # calculate Bayes factor using Savage-Dickey
    # we're testing if tau is close to 0 (positions are similar)
    prior_density_at_zero = stats.halfnorm.pdf(0, scale=5.0)
    posterior_density_at_zero = stats.gaussian_kde(tau_samples).pdf(0)[0]
    bf_01 = posterior_density_at_zero / prior_density_at_zero
    bf_10 = 1 / bf_01
    
    # calculate credible interval
    tau_ci = np.percentile(tau_samples, [2.5, 97.5])
    zero_in_ci = tau_ci[0] <= 0 <= tau_ci[1]
    
    # get position means
    mu_position_samples = trace.posterior['mu_position'].values
    position_means = []
    for pos in range(n_positions):
        position_means.append(float(np.mean(mu_position_samples[:, :, pos])))
    
    interpretation = _interpret_bayes_factor(bf_10)
    
    # IMPORTANT NOTE: High BF here is EXPECTED!
    # Lottery balls are drawn in sorted order (smallest to largest)
    # So position 1 will always be lower than position 5
    # This tests the drawing mechanism, not lottery randomness
    if bf_10 > 100:
        interpretation = f"{interpretation} - NOTE: This is expected! Balls are drawn in sorted order."
    
    print(f"  Between-position variability (τ):")
    print(f"  Mean: {np.mean(tau_samples):.4f}")
    print(f"  95% CI: [{tau_ci[0]:.4f}, {tau_ci[1]:.4f}]")
    print(f"  Position means: {[f'{m:.1f}' for m in position_means]}")
    print(f"  Bayes Factor (BF₁₀): {bf_10:.4f}")
    print(f"  Convergence: R̂ = {rhat_tau:.4f}, ESS = {ess_tau:.0f}")
    print(f"  Interpretation: {interpretation}")
    print(f"\n  NOTE: Positions have different means because balls are sorted!")
    print(f"  This is the expected behavior of the drawing mechanism.")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "model_name": "position_bias",
        "tau_mean": float(np.mean(tau_samples)),
        "tau_std": float(np.std(tau_samples)),
        "tau_credible_interval_95": [float(tau_ci[0]), float(tau_ci[1])],
        "position_means": position_means,
        "bayes_factor": float(bf_10),
        "interpretation": interpretation,
        "execution_time": execution_time,
        "convergence": {
            "r_hat": rhat_tau,
            "ess": ess_tau,
            "converged": bool(rhat_tau < 1.01)
        }
    }


def model4_hierarchical(game: str, raw_data_dir: Path) -> Dict[str, Any]:
    """
    Model 4: Test if numbers form clusters (hierarchical structure).
    
    NOTE: This model has implementation issues and doesn't properly test
    randomness. Kept for completeness but results should be ignored.
    
    The correct way to test clustering would be to use actual clustering
    algorithms on the co-occurrence patterns, which is done in Module 9.
    """
    print(f"\n--- MODEL 4: Hierarchical Clustering ({game.upper()}) ---")
    print("  NOTE: This model implementation is flawed - results not reliable")
    
    start_time = datetime.now()
    
    # load data
    json_file = raw_data_dir / f"{game}_current_format.json"
    with open(json_file, 'r') as f:
        raw_data = json.load(f)
    
    K = 69 if game == 'powerball' else 70
    
    # count occurrences
    all_numbers = []
    for drawing in raw_data:
        all_numbers.extend(drawing['regularNumbers'])
    
    counts = np.bincount(all_numbers, minlength=K+1)[1:]
    n_total = len(all_numbers)
    
    print(f"  Total numbers: {n_total}")
    print(f"  Number pool: {K}")
    
    # just use uniform model (same as Model 1)
    # this model doesn't actually test hierarchical structure properly
    alpha_flat = np.ones(K) * 0.5
    log_ml_flat = (
        gammaln(np.sum(alpha_flat)) - gammaln(n_total + np.sum(alpha_flat)) +
        np.sum(gammaln(counts + alpha_flat) - gammaln(alpha_flat))
    )
    
    # set BF to neutral (1.0) since this test isn't valid
    bayes_factor = 1.0
    log_bf = 0.0
    
    interpretation = "Model implementation flawed - use Module 9 for clustering tests"
    
    print(f"  Bayes Factor (BF₁₀): {bayes_factor:.4f} (not meaningful)")
    print(f"  Interpretation: {interpretation}")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "model_name": "hierarchical_dirichlet",
        "bayes_factor": float(bayes_factor),
        "log_bayes_factor": float(log_bf),
        "note": "Implementation flawed - results not reliable",
        "interpretation": interpretation,
        "execution_time": execution_time,
        "convergence": {"method": "analytical", "converged": True}
    }


def model5_regression(game: str, features_dir: Path) -> Dict[str, Any]:
    """
    Model 5: Test if engineered features can predict outcomes.
    
    I have 90+ features from Module 2. If lottery is random,
    these features shouldn't help predict the next drawing.
    
    I use Bayesian ridge regression to test this.
    If coefficients are all ~ 0, evidence for randomness.
    
    Note: To avoid overfitting with 90+ features, I first cluster
    correlated features and pick the one most correlated with target
    from each cluster.
    """
    print(f"\n--- MODEL 5: Bayesian Regression ({game.upper()}) ---")
    
    start_time = datetime.now()
    
    # load features from Module 2
    features_file = features_dir / f"features_{game}.parquet"
    df = pd.read_parquet(features_file)
    
    print(f"  Loaded {len(df)} drawings")
    print(f"  Total features: {len(df.columns)}")
    
    # select numeric features only
    # drop metadata columns and the target itself
    exclude_cols = ['game', 'drawing_index', 'date', 'dateISO', 'year', 'month', 
                   'day', 'day_of_week', 'jackpot_cash', 'mean', 'median', 'std', 'sum']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # target: mean of next drawing
    # I shift by 1 to predict next drawing from current features
    y = df['mean'].shift(-1).values[:-1]  # remove last row (no next drawing)
    X = df[feature_cols].iloc[:-1].values
    
    # remove any rows with NaN or inf values
    valid_idx = (~np.isnan(y) & 
                 ~np.any(np.isnan(X), axis=1) & 
                 ~np.any(np.isinf(X), axis=1))
    y = y[valid_idx]
    X = X[valid_idx]
    
    print(f"  Using {len(feature_cols)} numeric features")
    print(f"  Training samples: {len(y)}")
    
    # standardize features
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # avoid division by zero
    X_scaled = (X - np.mean(X, axis=0)) / X_std
    
    # standardize target
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_scaled = (y - y_mean) / y_std
    
    # Feature selection via clustering
    # cluster correlated features and pick one from each cluster
    # this reduces overfitting
    print("  Clustering features...")
    
    # calculate feature correlation matrix
    corr_matrix = np.corrcoef(X_scaled.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)  # replace NaNs
    
    # create distance matrix: distance = 1 - |correlation|
    # highly correlated features have small distance
    distance_matrix = 1 - np.abs(corr_matrix)
    
    # force symmetry to avoid floating point errors
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # diagonal must be exactly zero (distance to self = 0)
    np.fill_diagonal(distance_matrix, 0)
    
    # convert square distance matrix to condensed form
    # squareform() converts NxN matrix to N*(N-1)/2 vector
    condensed_distance = squareform(distance_matrix)
    
    # hierarchical clustering on the condensed distance vector
    linkage_matrix = linkage(condensed_distance, method='average')
    
    # form clusters using distance threshold
    # features with distance < 0.5 go in same cluster
    clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
    
    # from each cluster, pick the feature most correlated with target
    selected_indices = []
    for cluster_id in np.unique(clusters):
        cluster_features = np.where(clusters == cluster_id)[0]
        
        # calculate correlation of each feature in cluster with target
        correlations = []
        for i in cluster_features:
            corr = np.corrcoef(X_scaled[:, i], y_scaled)[0, 1]
            correlations.append(np.abs(corr))
        
        correlations = np.nan_to_num(correlations, nan=0.0)
        best_feature = cluster_features[np.argmax(correlations)]
        selected_indices.append(best_feature)
    
    # select the chosen features
    X_selected = X_scaled[:, selected_indices]
    selected_features = [feature_cols[i] for i in selected_indices]
    n_features = len(selected_features)
    
    print(f"  Reduced to {n_features} representative features")
    
    # build Bayesian ridge regression model with hierarchical shrinkage
    with pm.Model() as model:
        # global shrinkage parameter
        # controls how much all coefficients are pulled toward zero
        tau = pm.HalfNormal('tau', sigma=0.5)
        
        # intercept
        alpha = pm.Normal('alpha', mu=0, sigma=0.5)
        
        # coefficients with hierarchical prior
        # beta_raw ~ Normal(0, 1), then scaled by tau
        # this is "horseshoe-like" shrinkage
        beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=n_features)
        beta = pm.Deterministic('beta', beta_raw * tau)
        
        # noise
        sigma = pm.HalfNormal('sigma', sigma=1.0)
        
        # likelihood
        mu = alpha + pm.math.dot(X_selected, beta)
        pm.Normal('y', mu=mu, sigma=sigma, observed=y_scaled)
        
        # run MCMC
        # using fewer samples (1000) because we have more features
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=4,
            target_accept=0.90,  # slightly lower for speed
            random_seed=RANDOM_SEED,
            progressbar=False,
            return_inferencedata=True
        )
    
    print("  MCMC sampling complete")
    
    # extract samples
    beta_samples = trace.posterior['beta'].values
    tau_samples = trace.posterior['tau'].values.flatten()
    
    # check convergence
    summary = az.summary(trace, var_names=['tau', 'beta'])
    max_rhat = summary['r_hat'].max()
    
    # check which features have credible intervals excluding zero
    # if CI doesn't include 0, the feature might be predictive
    significant_features = []
    for i, feature in enumerate(selected_features):
        ci = np.percentile(beta_samples[:, :, i], [2.5, 97.5])
        if not (ci[0] <= 0 <= ci[1]):
            significant_features.append(feature)
    
    # calculate R² to measure predictive power
    # R² ≈ 0 means features can't predict
    n_chains, n_draws = trace.posterior['alpha'].shape
    total_samples = n_chains * n_draws
    alpha_samples_flat = trace.posterior['alpha'].values.flatten()
    beta_samples_flat = trace.posterior['beta'].values.reshape(total_samples, n_features)
    
    # sample 1000 posterior draws to calculate R²
    n_r2_samples = min(1000, total_samples)
    sample_indices = np.random.choice(total_samples, n_r2_samples, replace=False)
    
    r2_samples = []
    for idx in sample_indices:
        alpha_i = alpha_samples_flat[idx]
        beta_i = beta_samples_flat[idx, :]
        
        # predict with this set of parameters
        y_pred = alpha_i + X_selected @ beta_i
        
        # calculate R²
        ss_res = np.sum((y_scaled - y_pred)**2)
        ss_tot = np.sum((y_scaled - np.mean(y_scaled))**2)
        r2 = max(0, 1 - ss_res / ss_tot)  # clamp at 0
        r2_samples.append(r2)
    
    r2_mean = np.mean(r2_samples)
    r2_ci = np.percentile(r2_samples, [2.5, 97.5])
    
    # interpret based on R²
    # R² < 0.05 means features have negligible predictive power
    if r2_mean < 0.05:
        interpretation = f"Features have negligible predictive power (R² = {r2_mean:.3f}) - evidence for randomness"
    elif r2_mean < 0.15:
        interpretation = f"Features show weak predictive power (R² = {r2_mean:.3f}) - mostly random"
    else:
        interpretation = f"Features show some predictive power (R² = {r2_mean:.3f}) - may have patterns"
    
    print(f"  Global shrinkage (τ): {np.mean(tau_samples):.4f}")
    print(f"  Significant features: {len(significant_features)}/{n_features}")
    print(f"  R²: {r2_mean:.4f} [{r2_ci[0]:.4f}, {r2_ci[1]:.4f}]")
    print(f"  Convergence: Max R̂ = {max_rhat:.4f}")
    print(f"  Interpretation: {interpretation}")
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    return {
        "model_name": "bayesian_ridge_regression",
        "n_features_selected": n_features,
        "n_significant": len(significant_features),
        "significant_features": significant_features,
        "tau_mean": float(np.mean(tau_samples)),
        "r2_mean": float(r2_mean),
        "r2_ci": [float(r2_ci[0]), float(r2_ci[1])],
        "interpretation": interpretation,
        "execution_time": execution_time,
        "convergence": {
            "r_hat_max": float(max_rhat),
            "converged": bool(max_rhat < 1.01)
        }
    }


def run_all_models(raw_data_dir: Path, features_dir: Path, output_dir: Path):
    """
    Run all 5 Bayesian models for both games.
    
    This is the main entry point.

    """
    print("\n--- BAYESIAN MODEL ZOO - COMPLETE ANALYSIS ---")
    print(f"\nRaw data directory: {raw_data_dir}")
    print(f"Features directory: {features_dir}")
    print(f"Output directory: {output_dir}")
    
    # make sure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "description": "Bayesian analysis of lottery randomness",
            "n_samples": N_SAMPLES,
            "n_tune": N_TUNE,
            "random_seed": RANDOM_SEED
        }
    }
    
    # run models for each game
    for game in ['powerball', 'megamillions']:
        print(f"\n--- PROCESSING: {game.upper()} ---")
        
        game_results = {}
        
        try:
            # Model 1: Uniform test (fast - analytical)
            game_results['model1'] = model1_uniform(game, raw_data_dir)
            
            # Model 2: Temporal dependence (slow - MCMC)
            game_results['model2'] = model2_temporal(game, raw_data_dir)
            
            # Model 3: Position bias (slow - MCMC)
            game_results['model3'] = model3_position_bias(game, raw_data_dir)
            
            # Model 4: Hierarchical (fast - analytical)
            game_results['model4'] = model4_hierarchical(game, raw_data_dir)
            
            # Model 5: Regression (slow - MCMC)
            game_results['model5'] = model5_regression(game, features_dir)
            
            results[game] = game_results
            
        except Exception as e:
            # if one game fails, still try the other
            print(f"\n Error processing {game}: {str(e)}")
            traceback.print_exc()
            results[game] = {"error": str(e)}
    
    # save results
    output_file = output_dir / "bayesian_results_complete.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n--- ANALYSIS COMPLETE ---")
    print(f"\nResults saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    
    # print summary
    print("\n--- SUMMARY OF RESULTS ---")
    
    for game in ['powerball', 'megamillions']:
        if game in results and 'error' not in results[game]:
            print(f"\n{game.upper()}:")
            for model_key, model_data in results[game].items():
                if 'bayes_factor' in model_data:
                    bf = model_data['bayes_factor']
                    interpretation = model_data['interpretation']
                    print(f"  {model_data['model_name']:30s} BF = {bf:.4f} ({interpretation})")


def main():
    """Main execution"""
    # paths - update these for your setup
    base_dir = Path(r"C:\jackpotmath\lottery-lab")
    raw_data_dir = base_dir / "data" / "processed"
    features_dir = raw_data_dir / "features"
    output_dir = base_dir / "outputs" / "bayesian"
    
    # tried using relative paths but absolute is clearer
    # base_dir = Path(__file__).parent.parent.parent
    
    # run all models
    run_all_models(raw_data_dir, features_dir, output_dir)
    
    print("\n--- MODULE 4 COMPLETE ---")
    print("\nNext steps:")
    print("1. Review bayesian_results_complete.json")
    print("2. Run visualization script: python bayesian_visualization.py")
    print("3. Interpret Bayes factors using Jeffreys' scale")


if __name__ == "__main__":
    main()

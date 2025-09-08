"""
Tech Singularity Simulator
--------------------------
This script models compute growth under different scenarios and runs
Monte Carlo simulations to estimate the probability of crossing a
"singularity threshold" by given years.

Outputs:
- Log-scale plot of growth scenarios
- Probability table for selected years
- CSV with Monte Carlo results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------

years = np.arange(2012, 2061)  # simulation range
n_years = len(years)

baseline_year = 2012
baseline_compute = 1.0  # relative units

# Example "singularity threshold" (change if needed)
threshold = 1e6  # = 1 million × 2012 baseline compute

# Deterministic scenarios (annual multiplicative growth factor)
scenarios = {
    "Optimistic (≈11.5x/yr)": 11.5,
    "Moderate (≈5x/yr)": 5.0,
    "Conservative (≈2x/yr)": 2.0
}

# ----------------------------
# Deterministic trajectories
# ----------------------------
det_traj = {}
for name, factor in scenarios.items():
    years_from_baseline = years - baseline_year
    det_traj[name] = baseline_compute * (factor ** years_from_baseline)

# ----------------------------
# Monte Carlo simulation
# ----------------------------
mc_samples = 5000
log_mu = np.log(5.0)  # center around 5x/year
log_sigma = 0.8       # wide uncertainty

rng = np.random.default_rng(42)

# Each year: multiplicative factor per sample
mc_trajs = rng.lognormal(mean=log_mu, sigma=log_sigma, size=(mc_samples, n_years))
mc_compute = np.cumprod(mc_trajs, axis=1) * baseline_compute

# Probability of crossing threshold each year
crossed = (mc_compute >= threshold)
prob_by_year = crossed.mean(axis=0)

# ----------------------------
# Summarize by selected years
# ----------------------------
summary_rows = []
for y in [2025, 2030, 2040, 2045, 2050, 2060]:
    idx = int(y - baseline_year)
    if 0 <= idx < n_years:
        prob = prob_by_year[idx]
    else:
        prob = None
    summary_rows.append({"Year": y, "P(cross threshold by year)": prob})

summary_df = pd.DataFrame(summary_rows)

# ----------------------------
# Plot results
# ----------------------------
plt.figure(figsize=(10, 6))

# Deterministic scenarios
for name, vals in det_traj.items():
    plt.plot(years, vals, label=name, linewidth=1.6)

# Monte Carlo quantiles
q10 = np.quantile(mc_compute, 0.10, axis=0)
q50 = np.quantile(mc_compute, 0.50, axis=0)
q90 = np.quantile(mc_compute, 0.90, axis=0)

plt.plot(years, q50, linestyle='--', linewidth=2, label='MC median')
plt.fill_between(years, q10, q90, alpha=0.25, label='MC 10–90% range')

# Threshold line
plt.axhline(threshold, linestyle=':', linewidth=1.5, color='red')
plt.text(years[0]+1, threshold*1.1, f"Threshold = {threshold:.0e}", fontsize=9)

plt.yscale('log')
plt.xlabel('Year')
plt.ylabel('Relative compute (log scale, baseline 2012=1)')
plt.title('Tech Singularity: compute-growth scenarios & Monte Carlo uncertainty')
plt.grid(True, which="both", ls="--", linewidth=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------
# Save outputs
# ----------------------------
# Save summary probabilities
summary_df['P(cross threshold by year)'] = summary_df['P(cross threshold by year)'].apply(
    lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
)
print("\nSummary probabilities:\n", summary_df, "\n")

# Save full Monte Carlo results to CSV
out_df = pd.DataFrame({
    "Year": years,
    "MC_median": q50,
    "MC_p10": q10,
    "MC_p90": q90,
    "Prob_cross_threshold": prob_by_year
})
out_df.to_csv("singularity_simulation_results.csv", index=False)
print("Results saved to: singularity_simulation_results.csv")

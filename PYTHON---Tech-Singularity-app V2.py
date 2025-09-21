import numpy as np import pandas as pd import matplotlib.pyplot as plt

----------------------------

Functions

----------------------------

def compute_deterministic(years, baseline_year, baseline_compute, scenarios): """Compute deterministic trajectories for given scenarios.""" det_traj = {} for name, factor in scenarios.items(): years_from_baseline = years - baseline_year det_traj[name] = baseline_compute * (factor ** years_from_baseline) return det_traj

def run_monte_carlo(years, baseline_compute, mc_samples=5000, log_mu=np.log(5.0), log_sigma=0.8, seed=42): """Run Monte Carlo simulation of compute growth.""" rng = np.random.default_rng(seed) n_years = len(years) mc_trajs = rng.lognormal(mean=log_mu, sigma=log_sigma, size=(mc_samples, n_years)) mc_compute = np.cumprod(mc_trajs, axis=1) * baseline_compute return mc_compute

def compute_threshold_probs(mc_compute, threshold): """Compute probability of crossing threshold by year.""" crossed = (mc_compute >= threshold) prob_by_year = crossed.mean(axis=0) return prob_by_year

def summarize_probs(years, prob_by_year, baseline_year, selected_years): """Summarize crossing probabilities for selected years.""" summary_rows = [] n_years = len(years) for y in selected_years: idx = int(y - baseline_year) prob = prob_by_year[idx] if 0 <= idx < n_years else None summary_rows.append({"Year": y, "P(cross threshold by year)": prob}) return pd.DataFrame(summary_rows)

def plot_results(years, det_traj, mc_compute, threshold): """Plot deterministic and Monte Carlo results.""" plt.figure(figsize=(10, 6))

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

def test_custom_growth(years, baseline_year, baseline_compute, factor, threshold, until_year=2050): """Quick test for a custom growth factor (deterministic).""" years_from_baseline = years - baseline_year traj = baseline_compute * (factor ** years_from_baseline) idx = int(until_year - baseline_year) value = traj[idx] if 0 <= idx < len(years) else None crossed = value >= threshold if value is not None else False return {"Year": until_year, "Compute": value, "Crossed?": crossed}

----------------------------

Example usage

----------------------------

if name == "main": years = np.arange(2012, 2061) baseline_year = 2012 baseline_compute = 1.0 threshold = 1e6

scenarios = {
    "Optimistic (≈11.5x/yr)": 11.5,
    "Moderate (≈5x/yr)": 5.0,
    "Conservative (≈2x/yr)": 2.0
}

# Deterministic
det_traj = compute_deterministic(years, baseline_year, baseline_compute, scenarios)

# Monte Carlo
mc_compute = run_monte_carlo(years, baseline_compute)
prob_by_year = compute_threshold_probs(mc_compute, threshold)

# Summary
summary_df = summarize_probs(years, prob_by_year, baseline_year, [2025, 2030, 2040, 2045, 2050, 2060])
summary_df['P(cross threshold by year)'] = summary_df['P(cross threshold by year)'].apply(
    lambda x: f"{x:.1%}" if pd.notnull(x) else "N/A"
)
print("\nSummary probabilities:\n", summary_df, "\n")

# Save full results
q10 = np.quantile(mc_compute, 0.10, axis=0)
q50 = np.quantile(mc_compute, 0.50, axis=0)
q90 = np.quantile(mc_compute, 0.90, axis=0)
out_df = pd.DataFrame({
    "Year": years,
    "MC_median": q50,
    "MC_p10": q10,
    "MC_p90": q90,
    "Prob_cross_threshold": prob_by_year
})
out_df.to_csv("singularity_simulation_results.csv", index=False)
print("Results saved to: singularity_simulation_results.csv")

# Plot
plot_results(years, det_traj, mc_compute, threshold)

# Custom growth test example
print("Custom test (3x growth until 2050):", test_custom_growth(years, baseline_year, baseline_compute, 3.0, threshold, until_year=2050))


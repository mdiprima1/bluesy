import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def monte_carlo_simulation(start_price, time_period, step_simulation, vol, drift, num_paths=100):
    """
    Monte Carlo Simulation of Equity Prices.

    Parameters:
    - start_price: Initial price of the equity
    - time_period: Time period for the simulation (in years)
    - step_simulation: Frequency of steps ('daily' or 'monthly')
    - vol: Volatility of the equity (e.g., 0.2 for 20%)
    - drift: Drift rate (risk-free rate, e.g., 0.05 for 5%)
    - num_paths: Number of simulation paths (default: 100)

    Returns:
    - A DataFrame containing simulated price paths.
    """
    if step_simulation == "daily":
        dt = 1 / 252  # Daily steps
        steps = int(time_period * 252)
    elif step_simulation == "monthly":
        dt = 1 / 12  # Monthly steps
        steps = int(time_period * 12)
    else:
        raise ValueError("step_simulation must be 'daily' or 'monthly'")

    price_matrix = np.zeros((steps + 1, num_paths))
    price_matrix[0] = start_price

    for t in range(1, steps + 1):
        z = np.random.normal(size=num_paths)
        price_matrix[t] = price_matrix[t - 1] * np.exp((drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z)

    return pd.DataFrame(price_matrix)

def calculate_percentiles(data, steps):
    """
    Calculates the 2.5th, 25th, 50th, 75th, and 97.5th percentiles for the given steps.

    Parameters:
    - data: DataFrame containing simulation paths.
    - steps: List of steps to calculate percentiles for.

    Returns:
    - A DataFrame with percentiles at specified steps.
    """
    percentiles = {step: data.iloc[step].quantile([0.025, 0.25, 0.50, 0.75, 0.975]) for step in steps}
    return pd.DataFrame(percentiles).T

def plot_percentiles_with_lines(percentiles, risk_floor, blue_sky):
    """
    Plots the percentiles for one asset over time and adds horizontal lines for risk floor and blue sky values.

    Parameters:
    - percentiles: DataFrame containing percentiles for the asset.
    - risk_floor: The 2.5% worst-case value.
    - blue_sky: The 97.5% best-case value.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot percentiles
    ax.plot(percentiles.index, percentiles[0.025], label="Risk Floor (2.5% of Worst Outcomes)", linestyle="-", color="red")
    ax.plot(percentiles.index, percentiles[0.975], label="Blue Sky (2.5% of the Best Outcomes)", linestyle="-", color="blue")

    # Add horizontal lines for risk floor and blue sky values
    ax.axhline(y=risk_floor, color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=blue_sky, color="blue", linestyle="--", alpha=0.7)

    # Annotate horizontal lines
    ax.text(0, risk_floor, f"Risk Floor: {risk_floor:,.0f}", color="red", fontsize=12, fontweight="bold", verticalalignment='bottom')
    ax.text(0, blue_sky, f"Blue Sky: {blue_sky:,.0f}", color="blue", fontsize=12, fontweight="bold", verticalalignment='bottom')

    ax.set_title("Monte Carlo Simulation Percentiles")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0, top=1.1 * blue_sky)  # Ensure the Y-axis always includes zero
    st.pyplot(fig)

# Streamlit App
st.title("Monte Carlo Simulation")

# User input for volatility
vol = st.slider("Volatility", 0.1, 1.0, 0.2, 0.01)

# Fixed Parameters for Simulation
start_price = 10000
time_period = 1
step_simulation = "daily"
drift = 0.10
num_paths = 10000

# Simulate Asset
simulation_data = monte_carlo_simulation(
    start_price=start_price,
    time_period=time_period,
    step_simulation=step_simulation,
    vol=vol,
    drift=drift,
    num_paths=num_paths
)

# Calculate percentiles
steps_to_check = list(range(0, simulation_data.shape[0], 10))
percentiles = calculate_percentiles(simulation_data, steps_to_check)

# Get the 2.5% and 97.5% percentile values at the final step
final_step = steps_to_check[-1]
risk_floor = percentiles.loc[final_step, 0.025]
blue_sky = percentiles.loc[final_step, 0.975]

# Plot percentiles with horizontal lines
plot_percentiles_with_lines(percentiles, risk_floor, blue_sky)

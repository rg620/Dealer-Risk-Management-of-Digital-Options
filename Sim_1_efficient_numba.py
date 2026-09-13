import math
import numpy as np
import pandas as pd
import os
from numba import njit

K = 1.1000 #strike
sigma = 10/100 #volatility
T = 7/365 #Starting time to maturity
min_to_expiry = 7*24*60 #Starting minutes to expiry
one_min_in_years = 1 / (365 * 24 * 60) #one minute expressed in years
one_pip = 1/10000 #one pip
bend = 30 * one_pip
digi_shift = 30 * one_pip
parameter_grid_pips = (0, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300)
#seed = 20260810 # shard 1 #seed for the random number generator - makes the random sequence reproducible
#seed = 20260811 #shard 2
#seed = 20260812 #shard 3
seed = 20260813 #shard 4
hedge_minutes = (1, 5, 15, 30, 60)
test_payout = 50000000
S0 = 1.08442839935
hedge_delays = (1, 5, 15, 30, 60)
base_payout = 1000000
payouts = (5 * base_payout, 10 * base_payout, 25 * base_payout, 50 * base_payout)
results = []
p_hour = (0.00, 0.01, 0.025, 0.05, 0.10, 0.20)
critical_digi_threshold = 0.25
n_mc_paths = 5000

@njit
def normal_cdf(x):
    result = (1 + math.erf(x / math.sqrt(2))) / 2
    return result

@njit
def normal_pdf(x):
    result = math.exp((-x ** 2) / 2) / math.sqrt(2 * math.pi)
    return result

@njit
def d_2(S, kappa, sigma, tau):
    result = (math.log(S / kappa) - (1 / 2) * (sigma ** 2) * tau) / (sigma * math.sqrt(tau))
    return result

@njit
def d_1(S, kappa, sigma, tau):
    result = d_2(S, kappa, sigma, tau) + sigma * math.sqrt(tau)
    return result

@njit
def digital_value(S, kappa, sigma, tau):
    if tau > 0:
        result = normal_cdf(d_2(S, kappa, sigma, tau))
        return result
    else:
        if tau <= 0:
            if S >= kappa:
                return 1
            else:
                return 0

@njit
def digital_delta(S, kappa, sigma, tau):
    result = normal_pdf(d_2(S, kappa, sigma, tau)) / (S * sigma * math.sqrt(tau))
    return result

@njit
def call_delta(S, kappa, sigma, tau):
    result = normal_cdf(d_1(S, kappa, sigma, tau))
    return result

@njit
def call_spread_delta(S, kappa, sigma, tau, bend):
    if bend == 0:
        result = digital_delta(S, kappa, sigma, tau)
    else:
        lower_strike = kappa - bend / 2
        upper_strike = kappa + bend /2
        result = (call_delta(S, lower_strike, sigma, tau) - call_delta(S, upper_strike, sigma, tau)) / bend
    return result

@njit
def shifted_digital_delta(S, kappa, sigma, tau, shift):
    result = digital_delta(S, kappa+shift, sigma, tau)
    return result

def simulate_spot_path(starting_spot, mins_grid, one_min_in_years, vol, rand):
    spot_path_result = np.empty(len(mins_grid))
    mins_to_expiry = mins_grid[-1]
    spot_path_result[0] = starting_spot
    for i in range(mins_to_expiry):
        spot_path_result[i+1] = spot_path_result[i] * math.exp(
            (-(vol ** 2) * one_min_in_years / 2)
            + vol * math.sqrt(one_min_in_years) * rand[i]
        )
    return spot_path_result

def simulate_digital_value_path(spot_path_input, strike, vol, time_grid, mins_grid):
    digital_value_path_result = np.empty(len(mins_grid))
    for i in range(len(mins_grid)):
        digital_value_path_result[i] = digital_value(
            spot_path_input[i], 
            strike, 
            vol,
            time_grid[i]
            )
    return digital_value_path_result

def recover_critical_path(digi_value_path, critical_threshold):
    critical_path_result = digi_value_path >= critical_threshold
    return critical_path_result

def calculate_eligible_hours_path(critical_path):
    unwind_eligible_hours_path_result = np.empty(len(critical_path))
    for j in range(len(critical_path)):
        if j < len(critical_path) - 1 and critical_path[j]:
            if j == 0:
                unwind_eligible_hours_path_result[j] = 1 / 60
            else:
                unwind_eligible_hours_path_result[j] = (
                unwind_eligible_hours_path_result[j-1] 
                + 1 / 60
                )
        else:
            if j == 0:
                unwind_eligible_hours_path_result[j] = 0
            else:
                unwind_eligible_hours_path_result[j] = unwind_eligible_hours_path_result[j-1]
    return unwind_eligible_hours_path_result

def calculate_unwind_indices(eligible_hours_path, probabilities, client_latent, mins_grid):
    unwind_index_result = []
    for prob in probabilities:
        accumulated_hazard_path = (
            hazard_per_hour(prob) * eligible_hours_path
            )
        # find observations where random threshold has been reached
        crossing_indices = np.where(
            accumulated_hazard_path >= client_latent
            )[0]
        if len(crossing_indices) == 0:
            temp_unwind_index = {
                "p_hour": prob,
                "index": len(mins_grid) - 1
            }
        else:
            temp_unwind_index = {
                "p_hour": prob,
                "index": int(crossing_indices[0])
            }
        unwind_index_result.append(temp_unwind_index)
    return unwind_index_result

def simulate_one_path(
        rand,
        client_latent,
        starting_spot,
        mins_grid,
        time_grid,
        one_min_in_years,
        strike,
        vol,
        critical_threshold,
        probabilities
):
    spot_path = simulate_spot_path(
        starting_spot, 
        mins_grid,
        one_min_in_years,
        vol,
        rand
    )
    digital_value_path = simulate_digital_value_path(
        spot_path,
        strike,
        vol,
        time_grid,
        mins_grid
    )
    critical_path = recover_critical_path(digital_value_path, critical_threshold)
    unwind_eligible_hours_path = calculate_eligible_hours_path(critical_path)
    unwind_index = calculate_unwind_indices(
        unwind_eligible_hours_path,
        probabilities,
        client_latent,
        mins_grid
        )
    return(
        spot_path,
        digital_value_path,
        critical_path,
        unwind_eligible_hours_path,
        unwind_index
    )

# unwind probabilities

def hazard_per_hour(p_hour):
    return -math.log(1 - p_hour)


def hedge_management(
        delta_function,
        representation_parameter,
        hedge_delay,
        spot_path,
        tau_grid,
        critical_path,
        minute_grid,
        strike,
        vol,
        unwind_index=None
):
    hedge_observation_indices = minute_grid[minute_grid % hedge_delay == 0]

    if unwind_index is None:
        unwind_index = minute_grid[-1]

    hedge_observation_indices = hedge_observation_indices[
        hedge_observation_indices <= unwind_index
    ]

    if hedge_observation_indices[-1] != unwind_index:
        hedge_observation_indices = np.append(
            hedge_observation_indices,
            unwind_index
        )

    hedge_start_indices = hedge_observation_indices[:-1]
    hedge_end_indices = hedge_observation_indices[1:]

    critical_at_hedge_start = critical_path[hedge_start_indices]
    critical_at_hedge_end = critical_path[hedge_end_indices]

    included_hedge_interval = (
        critical_at_hedge_start | critical_at_hedge_end
    )

    hedge_start_spot = spot_path[hedge_start_indices]
    hedge_start_tau = tau_grid[hedge_start_indices]

    hedge_spot_change = (
        spot_path[hedge_end_indices] - spot_path[hedge_start_indices]
    )

    hedge_delta = np.empty(len(hedge_start_indices))

    for j in range(len(hedge_start_indices)):
        hedge_delta[j] = delta_function(
            hedge_start_spot[j],
            strike,
            vol,
            hedge_start_tau[j],
            representation_parameter
        )

    interval_hedge_pnl = hedge_delta * hedge_spot_change
    included_hedge_pnl = interval_hedge_pnl[included_hedge_interval]

    rebalance_clip = np.append(
        np.diff(hedge_delta),
        -hedge_delta[-1]
    )

    included_rebalance_clip = rebalance_clip[included_hedge_interval]

    hedge_pnl_per_unit = np.sum(included_hedge_pnl)

    absolute_clip_per_unit = np.sum(
        np.abs(included_rebalance_clip)
    )

    squared_clip_per_unit = np.sum(
        included_rebalance_clip ** 2
    )

    return (
        hedge_pnl_per_unit,
        absolute_clip_per_unit,
        squared_clip_per_unit
    )


def scale_hedge_results(
        payout,
        hedge_pnl_per_unit,
        absolute_clip_per_unit,
        squared_clip_per_unit
):
    hedge_pnl_pounds = payout * hedge_pnl_per_unit

    execution_cost_pounds = (
        2 * one_pip * payout * absolute_clip_per_unit
        + one_pip / 100000000
        * payout ** 2
        * squared_clip_per_unit
    )

    management_loss_pounds = (
        -hedge_pnl_pounds + execution_cost_pounds
    )

    return (
        hedge_pnl_pounds,
        execution_cost_pounds,
        management_loss_pounds
    )

@njit
def hedge_management_all_unwinds(
        delta_function,
        representation_parameter,
        hedge_delay,
        spot_path,
        tau_grid,
        critical_path,
        minute_grid,
        strike,
        vol,
        unwind_indices
):
    # Calculate the scheduled hedge history once.
    hedge_observation_indices = minute_grid[
        minute_grid % hedge_delay == 0
    ]

    hedge_start_indices = hedge_observation_indices[:-1]
    hedge_end_indices = hedge_observation_indices[1:]

    hedge_delta = np.empty(len(hedge_start_indices))

    for j in range(len(hedge_start_indices)):
        hedge_delta[j] = delta_function(
            spot_path[hedge_start_indices[j]],
            strike,
            vol,
            tau_grid[hedge_start_indices[j]],
            representation_parameter
        )

    hedge_spot_change = (
        spot_path[hedge_end_indices]
        - spot_path[hedge_start_indices]
    )

    included_hedge_interval = (
        critical_path[hedge_start_indices]
        | critical_path[hedge_end_indices]
    )

    interval_hedge_pnl = hedge_delta * hedge_spot_change

    # Ordinary scheduled rebalances. The last clip is the
    # expiry flattening.
    rebalance_clip = np.append(
        np.diff(hedge_delta),
        -hedge_delta[-1]
    )

    cumulative_pnl = np.cumsum(
        np.where(included_hedge_interval, interval_hedge_pnl, 0.0)
    )

    cumulative_abs_clip = np.cumsum(
        np.where(included_hedge_interval, np.abs(rebalance_clip), 0.0)
    )

    cumulative_squared_clip = np.cumsum(
        np.where(included_hedge_interval, rebalance_clip ** 2, 0.0)
    )

    results_by_unwind = []

    for unwind_index in unwind_indices:

        # Last scheduled observation at or before the unwind.
        last_observation = unwind_index // hedge_delay
        last_observation_index = last_observation * hedge_delay

        # Take only completed intervals BEFORE the final interval.
        if last_observation == 0:
            hedge_pnl_per_unit = 0.0
            absolute_clip_per_unit = 0.0
            squared_clip_per_unit = 0.0
        else:
            end_position = last_observation - 1

            hedge_pnl_per_unit = cumulative_pnl[end_position]
            absolute_clip_per_unit = cumulative_abs_clip[end_position]
            squared_clip_per_unit = cumulative_squared_clip[end_position]

        if unwind_index == minute_grid[-1]:
            # At expiry, the cumulative history already includes
            # the final interval and its flattening.
            pass

        elif unwind_index == last_observation_index:
            # Scheduled unwind before expiry:
            # replace the ordinary rebalance with a flatten.
            final_position = last_observation - 1
            old_delta = hedge_delta[final_position]

            include_final_interval = included_hedge_interval[final_position]

            if include_final_interval:
                old_clip = rebalance_clip[final_position]
                flatten_clip = -old_delta

                absolute_clip_per_unit += (
                    abs(flatten_clip) - abs(old_clip)
                )

                squared_clip_per_unit += (
                    flatten_clip ** 2 - old_clip ** 2
                )

        else:
            # Off-schedule unwind: add the partial interval using
            # the hedge held since the last scheduled observation.
            current_delta = hedge_delta[last_observation]

            include_final_interval = (
                critical_path[last_observation_index]
                or critical_path[unwind_index]
            )

            if include_final_interval:
                final_spot_change = (
                    spot_path[unwind_index]
                    - spot_path[last_observation_index]
                )

                hedge_pnl_per_unit += (
                    current_delta * final_spot_change
                )

                absolute_clip_per_unit += abs(current_delta)
                squared_clip_per_unit += current_delta ** 2

        results_by_unwind.append((
            hedge_pnl_per_unit,
            absolute_clip_per_unit,
            squared_clip_per_unit
        ))

    return results_by_unwind

#
''' 
def hedge_management(
        delta_function,
        representation_parameter,
        hedge_delay,
        payout,
        spot_path,
        tau_grid,
        critical_path,
        minute_grid,
        strike,
        vol,
        unwind_index=None
):
    hedge_observation_indices = minute_grid[minute_grid % hedge_delay == 0]
    if unwind_index is None:
        unwind_index = minute_grid[-1]
    hedge_observation_indices = hedge_observation_indices[
        hedge_observation_indices <= unwind_index
        ]
    if hedge_observation_indices[-1] != unwind_index:
        hedge_observation_indices = np.append(
            hedge_observation_indices,
            unwind_index
        )
    hedge_start_indices = hedge_observation_indices[:-1]
    hedge_end_indices = hedge_observation_indices[1:]
    critical_at_hedge_start = critical_path[hedge_start_indices]
    critical_at_hedge_end = critical_path[hedge_end_indices]
    included_hedge_interval = critical_at_hedge_start | critical_at_hedge_end
    hedge_start_spot = spot_path[hedge_start_indices]
    hedge_start_tau = tau_grid[hedge_start_indices]
    hedge_spot_change = spot_path[hedge_end_indices] - spot_path[hedge_start_indices]
    hedge_delta = np.empty(len(hedge_start_indices))
    for j in range(len(hedge_start_indices)):
        hedge_delta[j] = delta_function(
            hedge_start_spot[j], 
            strike, 
            vol, 
            hedge_start_tau[j], 
            representation_parameter
            )
    interval_hedge_pnl = hedge_delta * hedge_spot_change
    included_hedge_pnl = interval_hedge_pnl[included_hedge_interval]
    rebalance_clip = np.append(
        np.diff(hedge_delta), -hedge_delta[-1]
    )
    included_rebalance_clip = rebalance_clip[included_hedge_interval]
    actual_clip = payout * included_rebalance_clip
    effective_spread_pips = (
        2 + np.abs(actual_clip) / 100000000
    )
    transaction_cost = (
        np.abs(actual_clip)
        * effective_spread_pips
        * one_pip
    )
    hedge_pnl_pounds = payout * np.sum(included_hedge_pnl)
    execution_cost_pounds = np.sum(transaction_cost)
    management_loss_pounds = (
        -hedge_pnl_pounds + execution_cost_pounds
    )
    return(
        hedge_pnl_pounds,
        execution_cost_pounds,
        management_loss_pounds
    )
'''


minute_grid = np.arange(0, min_to_expiry+1)
tau_grid = (min_to_expiry - minute_grid) * one_min_in_years
rng = np.random.default_rng(seed)
#z = rng.standard_normal(min_to_expiry)

client_rng = np.random.default_rng(seed)
#client_latent = client_rng.exponential(1.0)

shard_file = "exercise1_shard_4_numba_5000.csv"
if os.path.exists(shard_file):
    raise FileExistsError(
        f"{shard_file} already exists. Rename or move it before starting a fresh run."
    )
for path_id in range(n_mc_paths):
    z = rng.standard_normal(min_to_expiry)
    client_latent = client_rng.exponential(1.0)
    spot_path, digital_value_path, critical_path, unwind_eligible_hours_path, unwind_index = simulate_one_path(
        z,
        client_latent,
        S0,
        minute_grid,
        tau_grid,
        one_min_in_years,
        K,
        sigma,
        critical_digi_threshold,
        p_hour
    )
    critical_indices = np.where(critical_path)[0]
    #where returns a tuple
    #where with just the condition returns the indices

    critical_int = critical_path.astype(int)
    critical_changes = np.diff(critical_int)
    entry_indices = np.where(critical_changes == 1)[0] + 1 #shift indices by 1
    exit_indices = np.where(critical_changes == -1)[0] + 1 #shift indices by 1

    included_interval = critical_path[:-1] | critical_path[1:]

    for hedge_delay in hedge_delays:
        for pips in parameter_grid_pips:

            bend = pips * one_pip
            shift = pips * one_pip

            unwind_times = [
                item["index"] for item in unwind_index
            ]

            cs_results = hedge_management_all_unwinds(
                call_spread_delta,
                bend,
                hedge_delay,
                spot_path,
                tau_grid,
                critical_path,
                minute_grid,
                K,
                sigma,
                unwind_times
            )

            sd_results = hedge_management_all_unwinds(
                shifted_digital_delta,
                shift,
                hedge_delay,
                spot_path,
                tau_grid,
                critical_path,
                minute_grid,
                K,
                sigma,
                unwind_times
            )

            for j in range(len(p_hour)):

                cs_pnl_unit, cs_abs_clip, cs_squared_clip = cs_results[j]
                sd_pnl_unit, sd_abs_clip, sd_squared_clip = sd_results[j]

                for payout in payouts:

                    cs_hedge_pnl, cs_cost, cs_loss = scale_hedge_results(
                        payout,
                        cs_pnl_unit,
                        cs_abs_clip,
                        cs_squared_clip
                    )

                    sd_hedge_pnl, sd_cost, sd_loss = scale_hedge_results(
                        payout,
                        sd_pnl_unit,
                        sd_abs_clip,
                        sd_squared_clip
                    )

                    result_dict = {
                        "payout": payout,
                        "hedge_delay": hedge_delay,
                        "pips": pips,
                        "p_hour": unwind_index[j]["p_hour"],
                        "path_id": path_id,
                        "cs_pnl": cs_hedge_pnl,
                        "sd_pnl": sd_hedge_pnl,
                        "cs_cost": cs_cost,
                        "sd_cost": sd_cost,
                        "cs_loss": cs_loss,
                        "sd_loss": sd_loss,
                        "sd_cs_loss_diff": sd_loss - cs_loss,
                        "sd_cs_pnl_diff": sd_hedge_pnl - cs_hedge_pnl,
                        "sd_cs_cost_diff": sd_cost - cs_cost
                    }

                    results.append(result_dict)

    '''
    for hedge_delay in hedge_delays:
        for pips in parameter_grid_pips:

            bend = pips * one_pip
            shift = pips * one_pip

            for j in range(len(p_hour)):

                cs_pnl_unit, cs_abs_clip, cs_squared_clip = hedge_management(
                    call_spread_delta,
                    bend,
                    hedge_delay,
                    spot_path,
                    tau_grid,
                    critical_path,
                    minute_grid,
                    K,
                    sigma,
                    unwind_index[j]["index"]
                )

                sd_pnl_unit, sd_abs_clip, sd_squared_clip = hedge_management(
                    shifted_digital_delta,
                    shift,
                    hedge_delay,
                    spot_path,
                    tau_grid,
                    critical_path,
                    minute_grid,
                    K,
                    sigma,
                    unwind_index[j]["index"]
                )

                for payout in payouts:

                    cs_hedge_pnl, cs_cost, cs_loss = scale_hedge_results(
                        payout,
                        cs_pnl_unit,
                        cs_abs_clip,
                        cs_squared_clip
                    )

                    sd_hedge_pnl, sd_cost, sd_loss = scale_hedge_results(
                        payout,
                        sd_pnl_unit,
                        sd_abs_clip,
                        sd_squared_clip
                    )

                    result_dict = {
                        "payout": payout,
                        "hedge_delay": hedge_delay,
                        "pips": pips,
                        "p_hour": unwind_index[j]["p_hour"],
                        "path_id": path_id,
                        "cs_pnl": cs_hedge_pnl,
                        "sd_pnl": sd_hedge_pnl,
                        "cs_cost": cs_cost,
                        "sd_cost": sd_cost,
                        "cs_loss": cs_loss,
                        "sd_loss": sd_loss,
                        "sd_cs_loss_diff": sd_loss - cs_loss,
                        "sd_cs_pnl_diff": sd_hedge_pnl - cs_hedge_pnl,
                        "sd_cs_cost_diff": sd_cost - cs_cost
                    }

                    results.append(result_dict)
    '''
    '''
    for payout in payouts:
        for hedge_delay in hedge_delays:
            for pips in parameter_grid_pips:
                bend = pips * one_pip
                shift = pips * one_pip
                for j in range(len(p_hour)):
                    cs_hedge_pnl, cs_cost, cs_loss = hedge_management(
                        call_spread_delta,
                        bend,
                        hedge_delay,
                        payout,
                        spot_path,
                        tau_grid,
                        critical_path,
                        minute_grid,
                        K,
                        sigma,
                        unwind_index[j]["index"]
                        )
                    sd_hedge_pnl, sd_cost, sd_loss = hedge_management(
                        shifted_digital_delta,
                        shift,
                        hedge_delay,
                        payout,
                        spot_path,
                        tau_grid,
                        critical_path,
                        minute_grid,
                        K,
                        sigma,
                        unwind_index[j]["index"]
                        )
                    result_dict = {
                        "payout": payout,
                        "hedge_delay": hedge_delay,
                        "pips": pips,
                        "p_hour": unwind_index[j]["p_hour"],
                        "path_id": path_id,
                        "cs_pnl": cs_hedge_pnl,
                        "sd_pnl": sd_hedge_pnl,
                        "cs_cost": cs_cost,
                        "sd_cost": sd_cost,
                        "cs_loss": cs_loss,
                        "sd_loss": sd_loss,
                        "sd_cs_loss_diff": sd_loss - cs_loss,
                        "sd_cs_pnl_diff": sd_hedge_pnl - cs_hedge_pnl,
                        "sd_cs_cost_diff": sd_cost - cs_cost
                    }
                    results.append(result_dict)
    '''


    if (path_id + 1) % 100 == 0 or path_id == n_mc_paths - 1:
        batch_df = pd.DataFrame(results)
        batch_df.to_csv(
            shard_file,
            mode = "a",
            header = (path_id < 100),
            index = False
        )
        results = []
        print(
            f"Completed {path_id + 1:,} / {n_mc_paths:,} paths",
            flush=True
        )


#results_df = pd.DataFrame(results)
#results_df.to_csv("exercise1_shard_1.csv", index=False)

#results_df = pd.read_csv(shard_file)

#print(results_df.shape)

'''
mean_results = results_df.groupby(
    ["payout", "hedge_delay", "pips", "p_hour"]
)[["cs_loss", 
   "sd_loss", 
   "sd_cs_loss_diff", 
   "cs_pnl", 
   "sd_pnl", 
   "cs_cost", 
   "sd_cost", 
   "sd_cs_pnl_diff", 
   "sd_cs_cost_diff"
   ]].agg([lambda x: x.count(),
          lambda x: x.mean(),
          lambda x: x.std(),
          lambda x: x.quantile(0.01),
          lambda x: x.quantile(0.05),
          lambda x: x.quantile(0.50),
          lambda x: x.quantile(0.95),
          lambda x: x.quantile(0.99),
          lambda x: x.std() / math.sqrt(x.count())
          ])


convergence_results = []

for n_paths in (25, 50, 75, 100):
    prefix_df = results_df[results_df["path_id"] < n_paths]
    prefix_summary = prefix_df.groupby(
        ["payout", "hedge_delay", "pips", "p_hour"]
    )[["cs_loss",
       "sd_loss",
       "sd_cs_loss_diff",
       "cs_pnl",
       "sd_pnl",
       "cs_cost",
       "sd_cost",
       "sd_cs_pnl_diff",
       "sd_cs_cost_diff"
       ]].agg([
          lambda x: x.count(),
          lambda x: x.mean(),
          lambda x: x.std(),
          lambda x: x.quantile(0.01),
          lambda x: x.quantile(0.05),
          lambda x: x.quantile(0.50),
          lambda x: x.quantile(0.95),
          lambda x: x.quantile(0.99),
          lambda x: x.std() / math.sqrt(x.count())
       ])
    convergence_results.append(prefix_summary)

convergence_df = pd.concat(
    convergence_results,
    keys=(25, 50, 75, 100),
    names=["n_paths"]
)
convergence_df.to_csv("exercise1_shard_1_convergence.csv")
print(mean_results)
print(mean_results.shape)
'''
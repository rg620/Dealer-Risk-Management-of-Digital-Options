import math
import numpy as np
import pandas as pd
import os
from statistics import NormalDist
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
#seed = 20260810 # shard 1 seed for the random number generator - makes the random sequence reproducible
seed = 20260811 # shard 2
hedge_minutes = (1, 5, 15, 30, 60)
test_payout = 50000000
#S0 = 1.08442839935
hedge_delays = (1, 5, 15, 30, 60)
base_payout = 1000000
payouts = (5 * base_payout, 10 * base_payout, 25 * base_payout, 50 * base_payout)
results = []
p_hour = (0.00, 0.01, 0.025, 0.05, 0.10, 0.20)
critical_digi_threshold = 0.25
n_mc_paths = 5000

jump_week_probs = (0.01, 0.025, 0.05, 0.10)
jump_sigmas = (0.005, 0.01, 0.02, 0.03)
jump_panels = ("overlay", "variance_matched")

jump_panel = "overlay"
jump_week_prob = 0.05
jump_sigma = 0.02

S0 = K * math.exp(
    NormalDist().inv_cdf(0.15) * sigma * math.sqrt(T)
    + 0.5 * sigma ** 2 * T
)

# Keep n_mc_paths = 100 for the first test.
# Use a different seed for each later 5,000-path shard.

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


def simulate_spot_path(
        starting_spot,
        mins_grid,
        one_min_in_years,
        vol,
        rand,
        jump_events,
        sigma_j,
        lambda_annual
):
    spot_path_result = np.empty(len(mins_grid))
    mins_to_expiry = mins_grid[-1]
    spot_path_result[0] = starting_spot

    drift = (
        -0.5 * vol ** 2
        - lambda_annual * jump_kappa(sigma_j)
    ) * one_min_in_years

    jump_number = 0
    n_jumps = len(jump_events)

    for i in range(mins_to_expiry):
        minute = i + 1
        log_jump = 0.0

        while (
            jump_number < n_jumps
            and jump_events[jump_number][0] == minute
        ):
            log_jump += sigma_j * jump_events[jump_number][1]
            jump_number += 1

        spot_path_result[i+1] = spot_path_result[i] * math.exp(
            drift
            + vol * math.sqrt(one_min_in_years) * rand[i]
            + log_jump
        )

    return spot_path_result

'''
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
'''

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
        probabilities,
        sigma_diff,
        lambda_annual,
        sigma_j,
        jump_events
):
    spot_path = simulate_spot_path(
        starting_spot,
        mins_grid,
        one_min_in_years,
        sigma_diff,
        rand,
        jump_events,
        sigma_j,
        lambda_annual
    )

    # Valuation remains Black at the original 10% volatility.
    digital_value_path = simulate_digital_value_path(
        spot_path,
        strike,
        vol,
        time_grid,
        mins_grid
    )

    critical_path = recover_critical_path(
        digital_value_path,
        critical_threshold
    )

    unwind_eligible_hours_path = calculate_eligible_hours_path(
        critical_path
    )

    unwind_index = calculate_unwind_indices(
        unwind_eligible_hours_path,
        probabilities,
        client_latent,
        mins_grid
    )

    return (
        spot_path,
        digital_value_path,
        critical_path,
        unwind_eligible_hours_path,
        unwind_index
    )

'''
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
'''

# unwind probabilities

def hazard_per_hour(p_hour):
    return -math.log(1 - p_hour)


def annual_jump_intensity(p_week):
    return -math.log(1 - p_week) / T

def jump_kappa(sigma_j):
    return math.exp(0.5 * sigma_j ** 2) - 1

def realised_diffusion_sigma(panel, p_week, sigma_j):
    if panel == "overlay":
        return sigma

    if panel == "variance_matched":
        lam = annual_jump_intensity(p_week)
        variance = sigma ** 2 - lam * sigma_j ** 2

        if variance <= 0:
            raise ValueError("Non-positive diffusion variance")

        return math.sqrt(variance)

    raise ValueError("Unknown jump panel")

def draw_jump_skeleton(seed, n_paths):
    max_week_mean = -math.log(1 - max(jump_week_probs))
    jump_rng = np.random.Generator(
        np.random.PCG64(seed + 10000000)
    )

    jump_counts = jump_rng.poisson(max_week_mean, size=n_paths)
    total_jumps = int(jump_counts.sum())

    jump_path_ids = np.repeat(
        np.arange(n_paths, dtype=np.int32),
        jump_counts.astype(np.int64)
    )

    jump_minutes = jump_rng.integers(
        1,
        min_to_expiry + 1,
        size=total_jumps,
        dtype=np.int32
    )

    thinning_marks = jump_rng.random(total_jumps)
    jump_z = jump_rng.standard_normal(total_jumps)

    return (
        jump_path_ids,
        jump_minutes,
        thinning_marks,
        jump_z
    )


def select_jump_events(
        jump_path_ids,
        jump_minutes,
        thinning_marks,
        jump_z,
        p_week,
        n_paths
):
    max_week_mean = -math.log(1 - max(jump_week_probs))
    target_week_mean = -math.log(1 - p_week)

    keep = thinning_marks <= (
        target_week_mean / max_week_mean + 1e-15
    )

    selected_paths = jump_path_ids[keep]
    selected_minutes = jump_minutes[keep]
    selected_z = jump_z[keep]

    order = np.lexsort((selected_minutes, selected_paths))

    selected_paths = selected_paths[order]
    selected_minutes = selected_minutes[order]
    selected_z = selected_z[order]

    events_by_path = [[] for _ in range(n_paths)]

    for j in range(len(selected_paths)):
        path_id = int(selected_paths[j])
        events_by_path[path_id].append((
            int(selected_minutes[j]),
            selected_z[j]
        ))

    return events_by_path


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


minute_grid = np.arange(0, min_to_expiry + 1)
tau_grid = (min_to_expiry - minute_grid) * one_min_in_years

# Generate the full production-shard random stream.
# The pilot uses the first 100 paths.
random_stream_paths = 5000

if n_mc_paths > random_stream_paths:
    raise ValueError("n_mc_paths exceeds random_stream_paths")

rng = np.random.Generator(np.random.PCG64(seed))

client_latents = rng.exponential(
    1.0,
    size=random_stream_paths
)

z_all = rng.standard_normal(
    (random_stream_paths, min_to_expiry)
)

# Generate the jump skeleton once.
jump_path_ids, jump_minutes, thinning_marks, jump_z = (
    draw_jump_skeleton(seed, random_stream_paths)
)

# Select the events for each weekly jump probability once.
# The same events are reused across jump sizes and panels.
events_for_probability = {}

for prob in jump_week_probs:
    events_for_probability[prob] = select_jump_events(
        jump_path_ids,
        jump_minutes,
        thinning_marks,
        jump_z,
        prob,
        random_stream_paths
    )

shard_file = "exercise2A_shard_2_numba_5000.csv"

if os.path.exists(shard_file):
    raise FileExistsError(
        f"{shard_file} already exists. Rename or move it before starting a fresh run."
    )

total_environments = (
    len(jump_panels)
    * len(jump_week_probs)
    * len(jump_sigmas)
)

environment_number = 0

for jump_panel in jump_panels:
    for jump_week_prob in jump_week_probs:
        for jump_sigma in jump_sigmas:

            environment_number += 1

            sigma_diff = realised_diffusion_sigma(
                jump_panel,
                jump_week_prob,
                jump_sigma
            )

            lambda_annual = annual_jump_intensity(jump_week_prob)

            events_by_path = events_for_probability[jump_week_prob]

            results = []

            for path_id in range(n_mc_paths):

                z = z_all[path_id]
                client_latent = client_latents[path_id]
                jump_events = events_by_path[path_id]

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
                    p_hour,
                    sigma_diff,
                    lambda_annual,
                    jump_sigma,
                    jump_events
                )

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
                                    "panel": jump_panel,
                                    "jump_week_probability": jump_week_prob,
                                    "jump_sigma_log": jump_sigma,
                                    "realised_diffusion_sigma": sigma_diff,
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

                # Write every 100 paths, including the final partial batch.
                if (path_id + 1) % 100 == 0 or path_id == n_mc_paths - 1:

                    batch_df = pd.DataFrame(results)

                    batch_df.to_csv(
                        shard_file,
                        mode="a",
                        header=not os.path.exists(shard_file),
                        index=False
                    )

                    results = []

            print(
                f"Completed environment {environment_number}/{total_environments}: "
                f"{jump_panel}, p_week={jump_week_prob}, "
                f"jump_sigma={jump_sigma}"
            )
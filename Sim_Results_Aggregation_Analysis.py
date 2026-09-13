import gc
import pandas as pd
import numpy as np
import math


'''
BLACK DIFFUSION RESULTS

'''


shard_files = [
    "exercise1_shard_1_numba_5000.csv",
    "exercise1_shard_2_numba_5000.csv",
    "exercise1_shard_3_numba_5000.csv",
    "exercise1_shard_4_numba_5000.csv",
]

key_cols = [
    "path_id",
    "payout",
    "hedge_delay",
    "pips",
    "p_hour",
]

'''
for filename in shard_files:

    print("\n========================================")
    print(filename)
    print("========================================")


    df = pd.read_csv(
        filename,
        usecols=key_cols
    )

    path_counts = df.groupby("path_id").size()

    print("Rows:", len(df))
    print("Number of path_ids:", df["path_id"].nunique())
    print("Minimum path_id:", df["path_id"].min())
    print("Maximum path_id:", df["path_id"].max())

    print(
        "All paths have 1,320 rows:",
        path_counts.eq(1320).all()
    )

    print(
        "Duplicate path/scenario rows:",
        df.duplicated([
            "path_id",
            "payout",
            "hedge_delay",
            "pips",
            "p_hour"
        ]).sum()
    )

    print("Payouts:", sorted(df["payout"].unique()))
    print("Hedge delays:", sorted(df["hedge_delay"].unique()))
    print("Pips:", sorted(df["pips"].unique()))
    print("p_hour:", sorted(df["p_hour"].unique()))

    del df
    gc.collect()

scenario_results = []

for filename in shard_files:

    df = pd.read_csv(
        filename,
        usecols=[
            "path_id",
            "payout",
            "hedge_delay",
            "pips",
            "p_hour",
            "cs_loss"
        ]
    )

    sample = df[
        (df["payout"] == 50000000) &
        (df["hedge_delay"] == 15) &
        (df["pips"] == 50) &
        (df["p_hour"] == 0.2)
    ][["path_id", "cs_loss"]].sort_values("path_id")

    print(filename, "rows in selected scenario:", len(sample))

    scenario_results.append(
        sample["cs_loss"].reset_index(drop=True)
    )

    del df
    gc.collect()

for i in range(4):
    for j in range(i + 1, 4):

        identical = scenario_results[i].equals(
            scenario_results[j]
        )

        print(
            f"Shard {i + 1} vs Shard {j + 1}:",
            "identical =", identical
        )
'''



group_cols = [
    "payout",
    "hedge_delay",
    "pips",
    "p_hour"
]

shard_summaries = []

for filename in shard_files:

    df = pd.read_csv(
        filename,
        usecols=group_cols + ["sd_cs_loss_diff"]
    )

    df = df[df["pips"] > 0].copy()

    # Needed to reconstruct the variance across all 20,000 paths.
    df["diff_sq"] = df["sd_cs_loss_diff"] ** 2

    summary = df.groupby(group_cols).agg(
        n=("sd_cs_loss_diff", "count"),
        sum_diff=("sd_cs_loss_diff", "sum"),
        sumsq_diff=("diff_sq", "sum"),
    )

    shard_summaries.append(summary)

    del df
    gc.collect()


combined = pd.concat(shard_summaries).groupby(
    level=group_cols
).sum()

combined["mean_diff"] = combined["sum_diff"] / combined["n"]
combined["sum_sq_err"] = combined["sumsq_diff"] - (combined["sum_diff"] ** 2) / combined["n"]
combined["std_diff"] = (combined["sum_sq_err"] / (combined["n"] - 1)) ** 0.5
combined["se_diff"] = combined["std_diff"] / (combined["n"] ** 0.5)



print("CS lower average loss:", (combined["mean_diff"] > 0).sum())
print("SD lower average loss:", (combined["mean_diff"] < 0).sum())

print("CS advantage > 2 SE:", (combined["mean_diff"] > 2 * combined["se_diff"]).sum())
print("SD advantage > 2 SE:", (combined["mean_diff"] < -2 * combined["se_diff"]).sum())

sd_wins = combined[combined["mean_diff"] < 0]
print(sd_wins.groupby(level="p_hour").size())

at_20pct = combined.xs(0.2, level="p_hour")

print("20% CS lower average loss:", (at_20pct["mean_diff"] > 0).sum())
print("20% SD lower average loss:", (at_20pct["mean_diff"] < 0).sum())


for payout in [5000000, 10000000, 25000000, 50000000]:

    payout_results = at_20pct.xs(payout, level="payout")

    print(
        payout,
        "SD wins:", (payout_results["mean_diff"] < 0).sum(),
        "SD wins > 2 SE:",
        (payout_results["mean_diff"] < -2 * payout_results["se_diff"]).sum()
    )



mechanism_summaries = []

for filename in shard_files:

    df = pd.read_csv(
        filename,
        usecols=group_cols + [
            "sd_cs_cost_diff",
            "sd_cs_pnl_diff"
        ]
    )

    df = df[
        (df["pips"] > 0) &
        (df["p_hour"] == 0.2)
    ]

    summary = df.groupby(group_cols).agg(
        sum_cost_diff=("sd_cs_cost_diff", "sum"),
        sum_pnl_diff=("sd_cs_pnl_diff", "sum"),
        n=("sd_cs_cost_diff", "count")
    )

    mechanism_summaries.append(summary)

    del df
    gc.collect()

mechanism_combined = pd.concat(mechanism_summaries).groupby(
    level=group_cols
).sum()

mechanism_combined["mean_cost_diff"] = mechanism_combined["sum_cost_diff"] / mechanism_combined["n"]
mechanism_combined["mean_pnl_diff"] = mechanism_combined["sum_pnl_diff"] / mechanism_combined["n"]

print(
    "SD lower average execution cost:",
    (mechanism_combined["mean_cost_diff"] < 0).sum()
)

print(
    "SD higher average hedge P&L:",
    (mechanism_combined["mean_pnl_diff"] > 0).sum()
)

tail_summaries = []

for filename in shard_files:

    df = pd.read_csv(
        filename,
        usecols=group_cols + ["sd_cs_loss_diff"]
    )

    df = df[df["pips"] > 0].copy()

    df["positive_diff"] = (df["sd_cs_loss_diff"] > 0).astype(int)

    summary = df.groupby(group_cols).agg(
        n=("sd_cs_loss_diff", "count"),
        n_positive=("positive_diff", "sum")
    )

    tail_summaries.append(summary)

    del df
    gc.collect()

tail_combined = pd.concat(tail_summaries).groupby(
    level=group_cols
).sum()

print(
    "Combinations with positive p95:",
    (tail_combined["n_positive"] > 0.05 * tail_combined["n"]).sum()
)

print(
    "Combinations with positive p99:",
    (tail_combined["n_positive"] > 0.01 * tail_combined["n"]).sum()
)


tail_20pct = tail_combined.xs(0.2, level="p_hour")

print(
    "At 20%, combinations where SD is worse than CS in the upper 5% of path outcomes:",
    (tail_20pct["n_positive"] > 0.05 * tail_20pct["n"]).sum()
)

print(
    "At 20%, combinations where SD is worse than CS in the upper 1% of path outcomes:",
    (tail_20pct["n_positive"] > 0.01 * tail_20pct["n"]).sum()
)

tail_20pct = tail_combined.xs(0.2, level="p_hour")

print(
    "At 20%, combinations where SD is worse than CS in the upper 5% of path outcomes:",
    (tail_20pct["n_positive"] > 0.05 * tail_20pct["n"]).sum()
)

print(
    "At 20%, combinations where SD is worse than CS in the upper 1% of path outcomes:",
    (tail_20pct["n_positive"] > 0.01 * tail_20pct["n"]).sum()
)


best_summaries = []

for filename in shard_files:

    df = pd.read_csv(
        filename,
        usecols=group_cols + ["cs_loss", "sd_loss"]
    )

    df = df[df["pips"] > 0]

    summary = df.groupby(group_cols).agg(
        n=("cs_loss", "count"),
        sum_cs_loss=("cs_loss", "sum"),
        sum_sd_loss=("sd_loss", "sum")
    )

    best_summaries.append(summary)

    del df
    gc.collect()


best_combined = pd.concat(best_summaries).groupby(
    level=group_cols
).sum()

best_combined["mean_cs_loss"] = (
    best_combined["sum_cs_loss"] / best_combined["n"]
)

best_combined["mean_sd_loss"] = (
    best_combined["sum_sd_loss"] / best_combined["n"]
)


best_by_environment = best_combined.groupby(
    level=["payout", "hedge_delay", "p_hour"]
).agg(
    best_cs_loss=("mean_cs_loss", "min"),
    best_sd_loss=("mean_sd_loss", "min")
)


print(
    "Across all unwind settings, best CS lower loss:",
    (best_by_environment["best_cs_loss"]
     < best_by_environment["best_sd_loss"]).sum()
)

at_2_5pct = best_by_environment.xs(0.025, level="p_hour")

print(
    "At 2.5% unwind, best CS lower loss:",
    (at_2_5pct["best_cs_loss"]
     < at_2_5pct["best_sd_loss"]).sum()
)


'''
JUMP DIFFUSION RESULTS FROM HERE

'''

jump_shard_files = [
    "exercise2A_shard_1_numba_5000.csv",
    "exercise2A_shard_2_numba_5000.csv"
]

jump_group_cols = [
    "panel",
    "jump_week_probability",
    "jump_sigma_log",
    "payout",
    "hedge_delay",
    "pips",
    "p_hour"
]

jump_summaries = []

for filename in jump_shard_files:

    for df in pd.read_csv(
        filename,
        usecols=jump_group_cols + [
            "sd_cs_loss_diff",
            "sd_cs_cost_diff",
            "sd_cs_pnl_diff"
        ],
        chunksize=1_000_000
    ):

        df = df[df["pips"] > 0].copy()

        df["diff_sq"] = df["sd_cs_loss_diff"] ** 2

        summary = df.groupby(jump_group_cols).agg(
            n=("sd_cs_loss_diff", "count"),
            sum_diff=("sd_cs_loss_diff", "sum"),
            sumsq_diff=("diff_sq", "sum"),
            sum_cost_diff=("sd_cs_cost_diff", "sum"),
            sum_pnl_diff=("sd_cs_pnl_diff", "sum")
        )

        jump_summaries.append(summary)

jump_combined = pd.concat(jump_summaries).groupby(
    level=jump_group_cols
).sum()

jump_combined["mean_diff"] = (
    jump_combined["sum_diff"] / jump_combined["n"]
)

jump_combined["sum_sq_err"] = (
    jump_combined["sumsq_diff"]
    - (jump_combined["sum_diff"] ** 2) / jump_combined["n"]
)

jump_combined["std_diff"] = (
    jump_combined["sum_sq_err"] / (jump_combined["n"] - 1)
) ** 0.5

jump_combined["se_diff"] = (
    jump_combined["std_diff"] / (jump_combined["n"] ** 0.5)
)

jump_combined["mean_cost_diff"] = (
    jump_combined["sum_cost_diff"] / jump_combined["n"]
)

jump_combined["mean_pnl_diff"] = (
    jump_combined["sum_pnl_diff"] / jump_combined["n"]
)


print("Number of jump comparisons:", len(jump_combined))

print(
    "CS lower average loss:",
    (jump_combined["mean_diff"] > 0).sum()
)

print(
    "SD lower average loss:",
    (jump_combined["mean_diff"] < 0).sum()
)

print(
    "SD advantage > 2 SE:",
    (jump_combined["mean_diff"] < -2 * jump_combined["se_diff"]).sum()
)

print(
    "SD lower average execution cost:",
    (jump_combined["mean_cost_diff"] < 0).sum()
)

print(
    "SD higher average hedge P&L:",
    (jump_combined["mean_pnl_diff"] > 0).sum()
)


for payout in [5000000, 10000000, 25000000, 50000000]:

    payout_jump = jump_combined.xs(
        payout,
        level="payout"
    )

    print(
        payout,
        "SD wins:",
        (payout_jump["mean_diff"] < 0).sum(),
        "SD wins > 2 SE:",
        (
            payout_jump["mean_diff"]
            < -2 * payout_jump["se_diff"]
        ).sum()
    )


    # ============================================================
# FINISH PUBLICATION-CRITICAL JUMP RESULTS: J10-J18
# ============================================================

# ------------------------------------------------------------
# J10: Where do SD mean-loss wins occur by payout?
# No file reread: uses jump_combined already in memory.
# ------------------------------------------------------------

print("\n--- J10: SD wins by payout ---")

for payout in [5000000, 10000000, 25000000, 50000000]:

    payout_jump = jump_combined.xs(
        payout,
        level="payout"
    )

    print(
        payout,
        "SD wins:",
        (payout_jump["mean_diff"] < 0).sum(),
        "SD wins > 2 SE:",
        (
            payout_jump["mean_diff"]
            < -2 * payout_jump["se_diff"]
        ).sum()
    )


# ------------------------------------------------------------
# J11: How do SD wins change with jump severity?
# Draft selected OVERLAY environments.
# No file reread.
# ------------------------------------------------------------

print("\n--- J11: selected overlay environments ---")

selected_environments = [
    (0.01,  0.005),
    (0.025, 0.03),
    (0.05,  0.03),
    (0.10,  0.03)
]

for p_week, jump_sigma in selected_environments:

    env = jump_combined.xs(
        ("overlay", p_week, jump_sigma),
        level=[
            "panel",
            "jump_week_probability",
            "jump_sigma_log"
        ]
    )

    print(
        f"p_week={p_week}, jump_sigma={jump_sigma}",
        "SD wins:",
        (env["mean_diff"] < 0).sum(),
        "SD wins > 2 SE:",
        (
            env["mean_diff"]
            < -2 * env["se_diff"]
        ).sum()
    )


# ============================================================
# Reconstruct which paths actually jumped at p_week = 5%.
# Needed only for J13-J15.
# This reproduces the jump-event construction in Sim_2.
# It does NOT rerun hedging.
# ============================================================

def jumped_paths_for_week(seed, p_week, n_paths=5000):

    max_week_mean = -math.log(1 - 0.10)

    jump_rng = np.random.Generator(
        np.random.PCG64(seed + 10000000)
    )

    jump_counts = jump_rng.poisson(
        max_week_mean,
        size=n_paths
    )

    total_jumps = int(jump_counts.sum())

    jump_path_ids = np.repeat(
        np.arange(n_paths, dtype=np.int32),
        jump_counts.astype(np.int64)
    )

    # Must reproduce this draw because it precedes the thinning marks.
    jump_rng.integers(
        1,
        7 * 24 * 60 + 1,
        size=total_jumps,
        dtype=np.int32
    )

    thinning_marks = jump_rng.random(total_jumps)

    target_week_mean = -math.log(1 - p_week)

    keep = thinning_marks <= (
        target_week_mean / max_week_mean + 1e-15
    )

    return set(jump_path_ids[keep].tolist())


jump_seed_by_file = {
    "exercise2A_shard_1_numba_5000.csv": 20260810,
    "exercise2A_shard_2_numba_5000.csv": 20260811
}

jumped_5pct_by_file = {
    filename: jumped_paths_for_week(seed, 0.05)
    for filename, seed in jump_seed_by_file.items()
}


# ============================================================
# ONE LAST LARGE-FILE PASS
#
# Simultaneously gathers:
# J12b-J12d representative jump cases
# J13-J15 realised-jump vs no-jump results
# J16-J17 adverse-path results
# J18 independently best CS vs independently best SD
# ============================================================

tail_best_summaries = []
representative_parts = []

for filename in jump_shard_files:

    jumped_5pct = jumped_5pct_by_file[filename]

    for df in pd.read_csv(
        filename,
        usecols=jump_group_cols + [
            "path_id",
            "cs_loss",
            "sd_loss",
            "sd_cs_loss_diff"
        ],
        chunksize=1_000_000
    ):

        positive = df[df["pips"] > 0].copy()

        # J16-J17: count path outcomes where SD loses more than CS.
        positive["positive_diff"] = (
            positive["sd_cs_loss_diff"] > 0
        ).astype(np.int8)

        summary = positive.groupby(
            jump_group_cols
        ).agg(
            n=("sd_cs_loss_diff", "count"),
            n_positive=("positive_diff", "sum"),
            sum_cs_loss=("cs_loss", "sum"),
            sum_sd_loss=("sd_loss", "sum")
        )

        tail_best_summaries.append(summary)


        # J12-J15 representative setup:
        # £50m, 50 pips, 15-minute hedging, p_hour = 20%.
        representative_mask = (
            (positive["payout"] == 50000000)
            & (positive["hedge_delay"] == 15)
            & (positive["pips"] == 50)
            & (positive["p_hour"] == 0.20)
            & (positive["jump_sigma_log"] == 0.03)
            & (
                (
                    (positive["panel"] == "overlay")
                    & positive["jump_week_probability"].isin(
                        [0.05, 0.10]
                    )
                )
                |
                (
                    (positive["panel"] == "variance_matched")
                    & (positive["jump_week_probability"] == 0.10)
                )
            )
        )

        rep = positive.loc[
            representative_mask,
            [
                "panel",
                "jump_week_probability",
                "jump_sigma_log",
                "path_id",
                "sd_cs_loss_diff"
            ]
        ].copy()

        if len(rep) > 0:

            rep["actual_jump"] = False

            is_5pct_overlay = (
                (rep["panel"] == "overlay")
                & (rep["jump_week_probability"] == 0.05)
                & (rep["jump_sigma_log"] == 0.03)
            )

            rep.loc[
                is_5pct_overlay,
                "actual_jump"
            ] = rep.loc[
                is_5pct_overlay,
                "path_id"
            ].isin(jumped_5pct)

            representative_parts.append(rep)

        del df, positive
        gc.collect()


# ============================================================
# Combine that final pass.
# ============================================================

tail_best_combined = pd.concat(
    tail_best_summaries
).groupby(
    level=jump_group_cols
).sum()


# ------------------------------------------------------------
# J16-J17
#
# Plain-English question:
# In adverse path outcomes, does SD end up losing more than CS?
# ------------------------------------------------------------

print("\n--- J16-J17: adverse path outcomes ---")

print(
    "Combinations where SD is worse than CS in the upper 5% of path outcomes:",
    (
        tail_best_combined["n_positive"]
        > 0.05 * tail_best_combined["n"]
    ).sum()
)

print(
    "Combinations where SD is worse than CS in the upper 1% of path outcomes:",
    (
        tail_best_combined["n_positive"]
        > 0.01 * tail_best_combined["n"]
    ).sum()
)


# ------------------------------------------------------------
# J18
#
# Give CS its own best positive bend.
# Give SD its own best positive shift.
# Compare the two independently optimised choices.
# ------------------------------------------------------------

tail_best_combined["mean_cs_loss"] = (
    tail_best_combined["sum_cs_loss"]
    / tail_best_combined["n"]
)

tail_best_combined["mean_sd_loss"] = (
    tail_best_combined["sum_sd_loss"]
    / tail_best_combined["n"]
)

jump_best_by_environment = tail_best_combined.groupby(
    level=[
        "panel",
        "jump_week_probability",
        "jump_sigma_log",
        "payout",
        "hedge_delay",
        "p_hour"
    ]
).agg(
    best_cs_loss=("mean_cs_loss", "min"),
    best_sd_loss=("mean_sd_loss", "min")
)

print("\n--- J18: independently best CS versus independently best SD ---")

print(
    "Number of best-vs-best comparisons:",
    len(jump_best_by_environment)
)

print(
    "Best CS lower average loss:",
    (
        jump_best_by_environment["best_cs_loss"]
        < jump_best_by_environment["best_sd_loss"]
    ).sum()
)


# ------------------------------------------------------------
# J12b-J12d
#
# Representative jump cases:
# mean SD loss - CS loss
# and 99th percentile of that pathwise difference.
# ------------------------------------------------------------

representative = pd.concat(
    representative_parts,
    ignore_index=True
)

representative_summary = representative.groupby(
    [
        "panel",
        "jump_week_probability",
        "jump_sigma_log"
    ]
)["sd_cs_loss_diff"].agg(
    n="count",
    mean_diff="mean",
    p99_diff=lambda x: x.quantile(0.99)
)

print("\n--- J12b-J12d: representative jump cases ---")
print(representative_summary)


# ------------------------------------------------------------
# J13-J15
#
# In the 5% weekly / 3% jump-vol OVERLAY case:
# compare paths with an actual realised jump
# against paths with no realised jump.
# ------------------------------------------------------------

rep_5pct_overlay = representative[
    (representative["panel"] == "overlay")
    & (representative["jump_week_probability"] == 0.05)
    & (representative["jump_sigma_log"] == 0.03)
]

conditional_summary = rep_5pct_overlay.groupby(
    "actual_jump"
)["sd_cs_loss_diff"].agg(
    n="count",
    mean_diff="mean",
    p99_diff=lambda x: x.quantile(0.99)
)

print("\n--- J13-J15: 5% / 3% overlay, actual jump versus no jump ---")
print(conditional_summary)


# ============================================================
# J12a: continuous-diffusion representative baseline.
#
# This scans the FOUR MUCH SMALLER Exercise 1 files.
# ============================================================

continuous_representative_parts = []

for filename in shard_files:

    for df in pd.read_csv(
        filename,
        usecols=[
            "payout",
            "hedge_delay",
            "pips",
            "p_hour",
            "sd_cs_loss_diff"
        ],
        chunksize=1_000_000
    ):

        mask = (
            (df["payout"] == 50000000)
            & (df["hedge_delay"] == 15)
            & (df["pips"] == 50)
            & (df["p_hour"] == 0.20)
        )

        continuous_representative_parts.append(
            df.loc[
                mask,
                "sd_cs_loss_diff"
            ].to_numpy()
        )

        del df
        gc.collect()


continuous_representative = np.concatenate(
    continuous_representative_parts
)

print("\n--- J12a: continuous representative case ---")

print(
    "n:",
    len(continuous_representative)
)

print(
    "Mean SD loss - CS loss:",
    continuous_representative.mean()
)

print(
    "99th percentile SD loss - CS loss:",
    np.quantile(
        continuous_representative,
        0.99
    )
)
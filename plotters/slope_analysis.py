import pandas as pd
import matplotlib.pyplot as plt
from plotting_utils import (
    add_flops_to_df,
    import_times_new_roman,
)
from matplotlib import font_manager
import argparse
import numpy as np
import os
import sys
from scipy.optimize import OptimizeResult
import time
import concurrent.futures
from itertools import product
from tqdm import tqdm

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)
from depth_width import optimize_params


import_times_new_roman(font_manager, plt, font_size=24)
plt.rcParams["lines.linewidth"] = 4


def create_param_search_array(num_parameters, target_grid_size=3125):
    """Create parameter search arrays trying to hit target grid size.

    Args:
        num_parameters: Number of parameters (e.g., 2 for depth/width)
        target_grid_size: Target total number of points in the grid
    """
    # Try reasonable ranges for each type of parameter
    max_range = 10
    min_range = 3
    best_combo = None
    best_diff = float("inf")

    # Calculate target subdivisions per dimension
    # For n parameters: (subdiv)^(2n) * err_subdiv ≈ target_grid_size
    target_per_dim = target_grid_size ** (1 / (2 * num_parameters + 1))
    target_per_dim = int(round(target_per_dim))

    # Search in a small window around the target
    search_window = 2
    min_search = max(min_range, target_per_dim - search_window)
    max_search = min(max_range, target_per_dim + search_window)

    for exp_subdivs in range(min_search, max_search + 1):
        for coef_subdivs in range(min_search, max_search + 1):
            for err_subdivs in range(min_search, max_search + 1):
                size = (
                    (exp_subdivs ** (num_parameters))
                    * (coef_subdivs ** (num_parameters))
                    * err_subdivs
                )
                diff = abs(size - target_grid_size)
                if diff < best_diff:
                    best_diff = diff
                    best_combo = (exp_subdivs, coef_subdivs, err_subdivs)

    exp_subdivs, coef_subdivs, err_subdivs = best_combo
    param_search_array = []

    # Exponents (0 to 2.5)
    for _ in range(num_parameters):
        param_search_array.append(np.linspace(0, 2.5, exp_subdivs))

    # Coefficients (0 to 30)
    for _ in range(num_parameters):
        param_search_array.append(np.linspace(0, 30, coef_subdivs))

    # Error term (-1 to 1.5)
    param_search_array.append(np.linspace(-1, 1.5, err_subdivs))

    actual_grid_size = (
        (exp_subdivs**num_parameters) * (coef_subdivs**num_parameters) * err_subdivs
    )
    # print(f"Target grid size: {target_grid_size:,}")
    # print(f"Actual grid size: {actual_grid_size:,}")
    # print(f"Using subdivisions: exponents={exp_subdivs}, coefficients={coef_subdivs}, error={err_subdivs}")

    return param_search_array


def grid_search(
    show_df,
    depth_width,
    depth_width_parameters,
    num_processes,
    delta,
    num_subdivisions,
):
    assert not (
        depth_width and depth_width_parameters
    ), "can't have depth_width and depth_width_parameters true at the same time"
    if depth_width:
        num_parameters = 3
        data = show_df[["width", "depth", "tokens"]].to_numpy()
    elif depth_width_parameters:
        num_parameters = 4
        data = show_df[["width", "depth", "params_active_precise", "tokens"]].to_numpy()
    else:
        data = show_df[["params_active_precise", "tokens"]].to_numpy()
        num_parameters = 2
    losses = show_df["final_loss"].to_numpy()

    param_search_array = create_param_search_array(
        num_parameters=num_parameters,
        target_grid_size=num_subdivisions,
    )

    best_loss = np.inf
    best_result = OptimizeResult()

    global_start_time = time.time()
    num_workers = num_processes if num_processes else os.cpu_count() // 2
    count = 0

    # print(f"Number of workers: {num_workers}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map each set of parameters to the optimize_params function, along with the necessary variables
        futures = {
            executor.submit(
                optimize_params,
                params,
                data,
                losses,
                num_parameters,
                delta,
            ): params
            for params in product(*param_search_array)
        }
        # print(f"Total futures submitted: {len(futures)}")

        # As results become available, check if they are better
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), disable=False
        ):

            result = future.result(timeout=60)
            count += 1
            if count % 10 == 0:
                global_total_time = time.time() - global_start_time
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_result = result
                # print(f"New best loss: {best_loss}", flush=True)

        global_total_time = time.time() - global_start_time
        # print(f"All tasks completed in {global_total_time} seconds.")

        # print(f"Best params: {best_result.x}")
        return dict(
            success=best_result.success,
            loss=best_loss,
            exponents=best_result.x[:num_parameters],
            coefficients=best_result.x[num_parameters : (num_parameters * 2)],
            irreducible_error=best_result.x[num_parameters * 2],
        )


def perform_main_analysis_ours_with_subdivision(
    show_df,
    num_processes,
    depth_width=False,
    depth_width_parameters=False,
    delta=1e-4,
    num_subdivisions=3125,
):
    out = [
        grid_search(
            show_df,
            depth_width,
            depth_width_parameters,
            num_processes,
            delta,
            num_subdivisions=num_subdivisions,
        )
    ]
    return pd.DataFrame(out)


def analyze_slope_progression(df, analyses, use_bins=False, ds_type="hot"):
    """Analyze how slope changes as we exclude lower values for multiple x/y combinations."""
    plt.figure(figsize=(12, 8))

    analysis = analyses[0]

    x_column = analysis["x_column"]
    y_axis_key = analysis["y_axis_key"]
    base_label = analysis["x_column"]

    approach = analysis["approach"]
    approach_3_parameters = approach == "A3_parameters"

    cutoff = 0
    grid_sizes = np.arange(3000, 10000, 1000)
    # then add up to 50000 in steps of 5000
    grid_sizes = np.concatenate([grid_sizes, np.arange(10000, 50000, 5000)])
    # then add up to 100000 in steps of 10000
    grid_sizes = np.concatenate([grid_sizes, np.arange(50000, 100000, 10000)])
    deltas = [1e-3, 1e-4, 1e-5]
    for delta in deltas:
        slopes = []
        percentages = []
        huber_losses = []
        for grid_size in grid_sizes:

            num_processes = 96
            if approach_3_parameters:
                out_dict = perform_main_analysis_ours_with_subdivision(
                    df,
                    num_processes,
                    delta=delta,
                    num_subdivisions=grid_size,
                )
                exponents = out_dict["exponents"].item()
                best_loss = out_dict["loss"].item()
                tokens_exp = exponents[1]  # beta
                params_exp = exponents[0]  # alpha
                a = tokens_exp / (tokens_exp + params_exp)
                b = 1 - a
            else:
                print("analysis method not found")
                exit()

            if x_column == "tokens":
                slope = b
            elif x_column == "params_active_precise":
                slope = a
            else:
                raise ValueError(f"Unknown x_column {x_column}")

            slopes.append(slope)
            percentages.append(cutoff)
            huber_losses.append(best_loss)

        label = f"{grid_size}"
        print(grid_sizes)
        print(slopes)
        line = plt.plot(grid_sizes, slopes, "o-", label=f"{delta}")[0]

        # Add Huber loss annotations to each point
        for x, y, loss in zip(grid_sizes, slopes, huber_losses):
            plt.annotate(
                f"{loss:.2e}",  # Format loss in scientific notation
                (x, y),
                xytext=(5, 5),  # Offset text slightly above and right of the point
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("Grid Size")
    plt.ylabel("Params Exp / (Params Exp + Tokens Exp)")
    plt.title("Scaling Law vs Fitted Tokens vs Grid Size")

    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    save_name = "slope_progression_combined_grid_axis"
    plt.savefig(f"../figures/{save_name}_{ds_type}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with --lr_ablation flag.")
    parser.add_argument("--over_100", action="store_true")
    args = parser.parse_args()
    ds_type = "hot"
    if args.over_100:
        ds_type += "_100b+"

    # Load the data once
    df_json = f"../wandb_dfs/wandb_df_for_fitting_{ds_type}.jsonl"
    df = pd.read_json(df_json, orient="records", lines=True)
    df = add_flops_to_df(df)

    # Define the analyses we want to run
    analyses = [
        {
            "x_column": "tokens",
            "y_axis_key": "params_active_precise",
            "label": "Tokens",
            "approach": "A3_parameters",
        },
    ]

    # Run the analyses
    analyze_slope_progression(df, analyses=analyses, use_bins=True, ds_type=ds_type)

import os
from matplotlib.axes import Axes
import pandas as pd
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.linear_model import (
    LinearRegression,
)  # scipy version can give p-values for H_0: slope=zero
import json
from matplotlib import font_manager
import argparse
from plotting_utils import (
    Y_Axis,
    flops_per_token_gqa,
    import_times_new_roman,
    plot_others,
    param_counter,
)
from matplotlib.lines import Line2D

import_times_new_roman(font_manager, plt, font_size=24)
plt.rcParams["lines.linewidth"] = 4

###### condense these functions into one file later ######
SAVE_DIR = "../figures"
marker_size = 7


def get_resource_hull(df, loss_key: str, resource_key: str):
    # Filter the rows to keep only positive resource values for meaningful convex hull calculation
    df = df[df[resource_key] > 0]

    max_resource = df[resource_key].max()
    max_loss = df[loss_key].max()
    vantage_point = [
        max_resource * 2,
        max_loss * 2,
    ]  # Add a high point so we get the lower edge of the simplex

    # Prepare points including the vantage point for hull calculation
    points = np.log(
        np.vstack(
            (
                vantage_point,  # Lower vantage point for boundary
                df[[resource_key, loss_key]].to_numpy(),  # Original data points
            )
        )
    )

    # Calculate the convex hull
    hull = ConvexHull(points)
    hull_indices = hull.vertices  # Get indices of hull points

    valid_hull_indices = hull_indices[hull_indices != 0] - 1

    hull_points = df.iloc[valid_hull_indices].copy()

    # Sort the points by the resource values to ensure they are in convex order
    hull_points = hull_points.sort_values(by=resource_key).reset_index(drop=True)

    # enforce decreasing y
    hull_points = hull_points[
        (hull_points[loss_key].diff().fillna(-1) < 0) | (hull_points.index == 0)
    ].reset_index(drop=True)

    return hull_points


def get_bucket_minimizers(df, loss_key: str, resource_key: str, n_buckets: int = 20):
    """
    Split the data (where resource_key > 0) into n_buckets (using logarithmic bins)
    and, in each bucket, select the row with the lowest loss.
    """
    df_pos = df[df[resource_key] > 0].copy()
    # Create bins in log-space since your x-axis is log-scaled
    bins = np.logspace(
        np.log10(df_pos[resource_key].min()),
        np.log10(df_pos[resource_key].max()),
        n_buckets + 1,
    )
    df_pos["bucket"] = pd.cut(df_pos[resource_key], bins=bins, include_lowest=True)
    # For each bucket, choose the row with the minimum loss.
    minimizers = df_pos.groupby("bucket", observed=True).apply(
        lambda sub_df: sub_df.loc[sub_df[loss_key].idxmin()]
    )
    return minimizers.reset_index(drop=True)


def plot_line_of_best_fit(
    minimizers,
    x_axis_key,
    y_axis_key,
    ax,
    x_points,
    save_name=None,
    label="Ours",
    color="#1f77b4",
):
    law = LinearRegression().fit(
        np.log(minimizers[x_axis_key].to_numpy()).reshape(-1, 1),
        np.log(minimizers[y_axis_key]),
    )

    model_params = {"coef": law.coef_.tolist(), "intercept": law.intercept_}
    model_params |= {"approach": 1, "label": save_name}

    if save_name is not None:
        with open(
            f"parameters/approach_1_linear_regression_{save_name}.json", "w"
        ) as f:
            json.dump(model_params, f)

    if ax is not None:
        if color == "#1f77b4":
            ax.scatter(
                minimizers[x_axis_key],
                minimizers[y_axis_key],
                color="black",
                marker="x",
                s=81,
                zorder=3,
            )
        ax.plot(
            np.exp(x_points),
            np.exp(law.predict(x_points)),
            label=label,
            color=color,
            # label=f"{y_axis_key}={np.exp(law.intercept_):.3e} * {x_axis_key}^{law.coef_[0]:.3}",
        )
        # if y_axis_key != "tokens_per_param":
        if True:
            annotation_label = f"{y_axis_key}={np.exp(law.intercept_):.3e} * {x_axis_key}^{law.coef_[0]:.3}"
            ax.text(
                0.95,  # X-coordinate (95% of the way across the axes)
                0.05,  # Y-coordinate (5% above the bottom of the axes)
                annotation_label,
                fontsize=10,
                color="red",
                ha="right",  # Align the text to the right
                va="bottom",  # Align the text to the bottom
                transform=ax.transAxes,  # Use axes-relative coordinates
                bbox=dict(
                    boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"
                ),
            )

    model_params |= {
        "x_minimizers": minimizers[x_axis_key].to_numpy(),
        "y_minimizers": minimizers[y_axis_key].to_numpy(),
    }

    return model_params


def add_existing_lines(
    xs: NDArray, params_ax: Axes, tokens_ax: Axes, tokens_per_param_ax: Axes
):
    # chinchilla approach 1 table
    names = ["Chinchilla-epochai-reported", "Kaplan"]
    if params_ax is not None:
        plot_others(names, params_ax, xs, Y_Axis.PARAMS)
    if tokens_ax is not None:
        plot_others(names, tokens_ax, xs, Y_Axis.DATA)
    if tokens_per_param_ax is not None:
        plot_others(names, tokens_per_param_ax, xs, Y_Axis.TOKENS_PER_PARAM)


# Function to categorize parameter size into groups
def categorize_params(param_size):
    if param_size < 90e6:
        return "50M"
    elif param_size < 450e6:
        return "100M"
    elif param_size < 900e6:
        return "500M"
    elif param_size < 1.5e9:
        return "1B"
    else:
        return "2B"


# Define linestyles for each parameter group
param_group_linestyles = {
    "50M": "-",
    "100M": "--",
    "500M": "-.",
    "1B": ":",
    # "2B": (0, (3, 1, 1, 1))  # Dash-dot-dot
    "2B": "-",
}

if __name__ == "__main__":
    loss_key = "final_loss"
    # second_col_df_name = "FLOPs"
    # second_col_df_name = "GPU_hours"
    save_postfix = "_gpu_hours"

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(20, 5), sharey=True
    )  # Slightly wider figure
    for second_col_df_name, axes, add_legend in [
        ("FLOPs", ax1, False),
        ("GPU_hours", ax2, True),
    ]:

        data_path = "../wandb_dfs/wandb_df_for_fitting_hot_477.jsonl"
        save_postfix_2 = ""

        # Add error handling for file reading
        try:
            with open(data_path, "r") as f:
                # Read the file line by line
                lines = f.readlines()
                if not lines:
                    raise ValueError(f"File {data_path} is empty")
                df = pd.DataFrame([json.loads(line) for line in lines])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find the data file at {data_path}. Please check if the file exists and the path is correct."
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {data_path}: {str(e)}")

        df = df[df["step"] > 0]

        df.rename(columns={"params_active_precise": "params"}, inplace=True)
        save_postfix += save_postfix_2

        df["legend_name"] = (
            df["run_name"].str.replace("PyGemma-", "").str.replace("_pretrain", "")
        )

        df["flops_per_tok_col"] = 6 * df.params.to_numpy()
        df["flops_per_tok_me"] = flops_per_token_gqa(
            df.width.to_numpy(), df.depth.to_numpy()
        )  # same name as resolving
        df["flops_ratio"] = df.flops_per_tok_me / df.flops_per_tok_col
        df["flops_per_token"] = df.flops_per_tok_me.astype(
            float
        )  # even int64 may not be big enough!

        df["FLOPs"] = df.flops_per_token * df.tokens

        df["GPU_hours"] = df.apply(
            lambda row: row.seconds_per_step * row.step * row.num_nodes * 8 / 3600,
            axis=1,
        )
        df["aspect_ratio"] = df.width / df.depth
        df["tokens_per_param"] = df["tokens"] / df["params"]

        # Add parameter group to the dataframe
        df["param_group"] = df["params"].apply(categorize_params)

        # Find min and max aspect ratios
        min_aspect = df["aspect_ratio"].min()
        max_aspect = df["aspect_ratio"].max()

        # Print range for debugging
        print(f"Aspect ratio range: {min_aspect} to {max_aspect}")

        # Find quantiles of aspect ratio for better color distribution
        quantiles = [0.25, 0.5, 0.75]  # Quartiles
        aspect_quantiles = (
            [df["aspect_ratio"].min()]
            + [df["aspect_ratio"].quantile(q) for q in quantiles]
            + [df["aspect_ratio"].max()]
        )
        print(f"Aspect ratio quantiles: {aspect_quantiles}")

        # Define a blue-to-red color spectrum with green/yellow in the middle
        viridis = plt.cm.viridis
        quartile_colors = [
            viridis(0.9),  # Yellow/green end of viridis for small width/depth ratio
            viridis(0.6),  # Green-blue middle
            viridis(0.3),  # Blue-purple middle
            viridis(0.0),  # Dark purple/blue end for large width/depth ratio
        ]

        # Dictionary to store min/max points for each parameter group
        group_start_points = {}
        group_end_points = {}

        bucket_minimizers_gpuhours = get_bucket_minimizers(
            df, loss_key=loss_key, resource_key=second_col_df_name, n_buckets=750
        )

        gpuhours_minimizers = get_resource_hull(
            df, loss_key=loss_key, resource_key=second_col_df_name
        )
        hull_run_names = set(gpuhours_minimizers["run_name"])

        run_colors = {}
        # First pass to collect all points
        for i, (run_name, group) in enumerate(df.groupby("run_name")):
            # Get parameter group and aspect ratio
            param_group = group["param_group"].iloc[0]
            aspect_ratio = group["aspect_ratio"].iloc[0]

            # Get linestyle from parameter group
            line_style = param_group_linestyles.get(param_group, "-")

            # Determine which quartile this aspect ratio falls into
            quartile_idx = 0
            for i in range(1, len(aspect_quantiles)):
                if aspect_ratio <= aspect_quantiles[i]:
                    quartile_idx = i - 1
                    break

            # Get color from quartile
            color = quartile_colors[quartile_idx]
            run_colors[run_name] = color
            print(
                f"{param_group} {aspect_ratio:.2f} → Quartile {quartile_idx+1} → Color: {color}"
            )

            # Plot the curve
            alpha_val = 0.4  # 1.0  # if run_name in hull_run_names else 0.4
            # color_val = color if run_name in hull_run_names else
            color_val = "gray"
            axes.plot(
                group[second_col_df_name],
                group[loss_key],
                line_style,
                color=color_val,
                alpha=alpha_val,
            )

            # Track start point (first x, y)
            x_start = group[second_col_df_name].iloc[0]
            y_start = group[loss_key].iloc[0]

            # Track end point (last x, y)
            x_end = group[second_col_df_name].iloc[-1]
            y_end = group[loss_key].iloc[-1]

            # Initialize or update group boundaries
            if param_group not in group_start_points:
                group_start_points[param_group] = {
                    "x_min": x_start,
                    "x_max": x_start,
                    "y_min": y_start,
                    "y_max": y_start,
                }
                group_end_points[param_group] = {
                    "x_min": x_end,
                    "x_max": x_end,
                    "y_min": y_end,
                    "y_max": y_end,
                }
            else:
                # Update start boundaries
                if x_start < group_start_points[param_group]["x_min"]:
                    group_start_points[param_group]["x_min"] = x_start
                if x_start > group_start_points[param_group]["x_max"]:
                    group_start_points[param_group]["x_max"] = x_start
                if y_start < group_start_points[param_group]["y_min"]:
                    group_start_points[param_group]["y_min"] = y_start
                if y_start > group_start_points[param_group]["y_max"]:
                    group_start_points[param_group]["y_max"] = y_start

                # Update end boundaries
                if x_end < group_end_points[param_group]["x_min"]:
                    group_end_points[param_group]["x_min"] = x_end
                if x_end > group_end_points[param_group]["x_max"]:
                    group_end_points[param_group]["x_max"] = x_end
                if y_end < group_end_points[param_group]["y_min"]:
                    group_end_points[param_group]["y_min"] = y_end
                if y_end > group_end_points[param_group]["y_max"]:
                    group_end_points[param_group]["y_max"] = y_end

        second_col_axis_name = second_col_df_name.replace("_", " ")
        axes.tick_params(labelbottom=True, labelleft=True)
        axes.set_xlabel(second_col_axis_name)
        axes.set_xscale("log")
        if not add_legend:
            axes.set_ylabel("Val Loss")
        axes.set_yscale("log")

        # More robust approach to fix the y-axis formatting
        from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

        # First apply a scalar formatter that doesn't use scientific notation
        axes.yaxis.set_minor_formatter(
            ScalarFormatter(useOffset=False, useMathText=False)
        )

        # Plot the convex hull
        axes.plot(
            gpuhours_minimizers[second_col_df_name],
            gpuhours_minimizers[loss_key],
            ":",
            label="Pareto Frontier (Convex Hull)" if add_legend else None,
            color="red",
        )

        for _, row in gpuhours_minimizers.iterrows():
            color = run_colors.get(row["run_name"], "black")  # fallback if not found
            axes.scatter(
                row[second_col_df_name],
                row[loss_key],
                color=color,
                marker="x",
                s=200,  # 81,
                zorder=3,
            )

        N_PTS = 4  # multiple points in case we use a non-log scale

        flops_xs = np.exp(
            np.linspace(
                np.log(df[df.FLOPs > 0].FLOPs.min()),
                np.log(df.FLOPs.max()),
                N_PTS,
            )
        )
        gpu_hours_xs = np.exp(
            np.linspace(
                np.log(df[df[second_col_df_name] > 0][second_col_df_name].min()),
                np.log(df[second_col_df_name].max()),
                N_PTS,
            )
        )

        flops_xs = flops_xs.reshape(-1, 1)
        gpu_hours_xs = gpu_hours_xs.reshape(-1, 1)

        # Add grid
        axes.grid(True, which="major", linestyle="--", linewidth=0.5)

        # Move convex hull legend to bottom left
        if add_legend:
            # convex_legend = fig.legend(loc="lower center", bbox_to_anchor=(0.45, -0.05))

            # Create parameter group legend for linestyles
            param_group_handles = [
                Line2D([0], [0], color="black", linestyle=style, label=group)
                for group, style in param_group_linestyles.items()
            ]

            # Create a custom legend for aspect ratio quartiles
            convex_handle = Line2D(
                [0], [0], color="red", linestyle=":", lw=5, label="Convex Hull"
            )

            aspect_handles = []  # Start with convex hull handle
            aspect_labels = []

            for i in range(4):
                low = aspect_quantiles[i]
                high = aspect_quantiles[i + 1]
                aspect_handles.append(Line2D([0], [0], color=quartile_colors[i], lw=5))
                end = ""
                if i == 0:
                    end = "\n(Deep)"
                if i == 3:
                    end = "\n(Wide)"
                aspect_labels.append(f"{low:.1f}-{high:.1f}{end}")

            aspect_handles.append(convex_handle)
            aspect_labels.append("Convex Hull")
            # Add the aspect ratio legend with better positioning
            # Place it outside the plot area on the right
            aspect_legend = fig.legend(
                aspect_handles,
                aspect_labels,
                loc="lower center",
                title="Width/Depth Ratio",
                ncol=5,
                bbox_to_anchor=(0.45, -0.25),
            )

    # Adjust layout to make room for legends
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for legend

    prefix = f"{SAVE_DIR}/figure_2{'_6N' if all(df.flops_per_token == df.flops_per_tok_col) else ''}{save_postfix}"
    fig.savefig(prefix + "_shade_wd.pdf", bbox_inches="tight")

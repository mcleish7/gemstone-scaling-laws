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
from collections import OrderedDict
from plotting_utils import (
    Y_Axis,
    model_styles,
    flops_per_token_gqa,
    import_times_new_roman,
    plot_others,
    param_counter,
)
from approach_1_extra_plots import flops_accounting, plot_efficiency_vs_loss

import_times_new_roman(font_manager, plt, font_size=28)
plt.rcParams["lines.linewidth"] = 4

###### condense these functions into one file later ######
SAVE_DIR = "figures"
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
        if y_axis_key != "tokens_per_param":
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script with --lr_ablation flag.")
    parser.add_argument("--lr_ablation", action="store_true")
    parser.add_argument("--cooldown", action="store_true")
    parser.add_argument("--no_embeds", action="store_true")
    parser.add_argument("--over_100", action="store_true")
    parser.add_argument("--over_120_only", action="store_true")
    parser.add_argument("--flops_comparison_plot", action="store_true")
    parser.add_argument("--overspending_plot", action="store_true")
    args = parser.parse_args()

    loss_key = "final_loss"
    second_col_df_name = "GPU_hours"
    save_postfix = "_gpu_hours"

    if args.lr_ablation:
        data_path = "wandb_dfs/wandb_df_for_fitting_lr_ablation_hot_477.jsonl"
        save_postfix_2 = "_lr_ablation"
    elif args.cooldown:
        data_path = "wandb_dfs/wandb_df_for_fitting_cool_end.jsonl"
        save_postfix_2 = "_cooldown"
    elif args.over_100:
        data_path = "wandb_dfs/wandb_df_for_fitting_hot_100b+_477.jsonl"
        save_postfix_2 = "_100b+"
    elif args.over_120_only:
        data_path = "wandb_dfs/wandb_df_for_fitting_hot_120b+_only_477.jsonl"
        save_postfix_2 = "_120b+_only"
    else:
        data_path = "wandb_dfs/wandb_df_for_fitting_hot_477.jsonl"
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
    if args.lr_ablation:
        df["run_name"] = df["run_name"].str.replace("cooler_", "")

    df.rename(columns={"params_active_precise": "params"}, inplace=True)

    if args.no_embeds:
        df["params"] = df.apply(
            lambda row: param_counter(row["width"], row["depth"], vocab_size=0), axis=1
        )
        save_postfix_2 += "_no_embeds"
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

    # makes other plots
    if args.flops_comparison_plot:  # turn on if we want to look at the FLOPs diff
        flops_accounting(df, save_postfix)
    if args.overspending_plot:  # plots the two sets of minimizers
        assert args.over_100, "need 300b data to do this plot"
        plot_efficiency_vs_loss(
            df, loss_key, second_col_df_name, save_postfix, get_resource_hull
        )

    nrows = 4
    ncols = 2
    fig, axes = plt.subplots(4, 2, figsize=(18, 16), sharex="col", sharey="row")
    fig_2, axes_2 = plt.subplots(2, 2, figsize=(18, 8), sharex="col", sharey="row")

    second_col_axis_name = second_col_df_name.replace("_", " ")
    for k, ax in enumerate([axes, axes_2]):
        for i in range(nrows):
            if k == 1 and i > 1:
                continue
            ax[i, 0].set_xlabel("FLOPs")  # only partially less wrong
            ax[i, 0].tick_params(labelbottom=True, labelleft=True)
            ax[i, 1].set_xlabel(second_col_axis_name)
            ax[i, 1].tick_params(labelbottom=True, labelleft=True)
        ax[0, 0].set_xscale("log")
        ax[0, 1].set_xscale("log")

    for j in range(ncols):
        axes[0, j].set_ylabel("Val Loss")
        axes[1, j].set_ylabel("Parameters")
        axes[2, j].set_ylabel("Tokens")
        axes[3, j].set_ylabel("Tokens/Param")
        axes_2[0, j].set_ylabel("Val Loss")
        axes_2[1, j].set_ylabel("Tokens/Param")

    for j in range(1, nrows):
        axes[j, 0].set_yscale("log")

    axes_2[1, 0].set_yscale("log")

    for i, (run_name, group) in enumerate(df.groupby("run_name")):
        color = model_styles[run_name]["color"]
        line_style = model_styles[run_name]["line_style"]

        # Plot FLOPs vs loss for each run_name in the first subplot
        for ax in [axes, axes_2]:
            ax[0, 0].plot(
                group["FLOPs"],
                group[loss_key],
                line_style,
                # label=group["legend_name"].iloc[0],
                color=color,  # Use a different color for each run_name
            )

            # Plot GPU_hours vs loss for each run_name in the second subplot
            ax[0, 1].plot(
                group[second_col_df_name],
                group[loss_key],
                line_style,
                # label=group["legend_name"].iloc[0],
                color=color,  # Use a different color for each run_name
            )

    flops_minimizers = get_resource_hull(df, loss_key=loss_key, resource_key="FLOPs")
    for ax in [axes, axes_2]:
        ax[0, 0].plot(
            flops_minimizers["FLOPs"],
            flops_minimizers[loss_key],
            ":",
            label="Convex Hull",
            color="black",
        )
        ax[0, 0].scatter(
            flops_minimizers["FLOPs"],
            flops_minimizers[loss_key],
            color="black",
            marker="x",
            s=81,
            zorder=3,
        )

    flops_minimizers.to_parquet(
        f"plotters/data_cache/figure_2_flops_mins_hot{'_'if save_postfix_2 != '' else ''}{save_postfix_2}.parquet",
        index=False,
    )

    bucket_minimizers_flops = get_bucket_minimizers(
        df, loss_key=loss_key, resource_key="FLOPs", n_buckets=750
    )
    bucket_minimizers_gpuhours = get_bucket_minimizers(
        df, loss_key=loss_key, resource_key=second_col_df_name, n_buckets=750
    )

    gpuhours_minimizers = get_resource_hull(
        df, loss_key=loss_key, resource_key=second_col_df_name
    )
    for ax in [axes, axes_2]:
        ax[0, 1].plot(
            gpuhours_minimizers[second_col_df_name],
            gpuhours_minimizers[loss_key],
            ":",
            label="Convex Hull",
            color="black",
        )
        ax[0, 1].scatter(
            gpuhours_minimizers[second_col_df_name],
            gpuhours_minimizers[loss_key],
            color="black",
            marker="x",
            s=81,
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

    for ax in [axes_2]:
        ax[1, 0].scatter(
            bucket_minimizers_flops["FLOPs"],
            bucket_minimizers_flops["tokens_per_param"],
            color="red",
            marker="x",
            s=36,
            zorder=2,
            alpha=0.3,
        )
        plot_line_of_best_fit(
            bucket_minimizers_flops,
            "FLOPs",
            "tokens_per_param",
            ax[1, 0],
            np.log(flops_xs),
            label="Binning",
            color="red",
        )
        ax[1, 1].scatter(
            bucket_minimizers_gpuhours[second_col_df_name],
            bucket_minimizers_gpuhours["tokens_per_param"],
            color="red",
            marker="x",
            s=36,
            zorder=2,
            alpha=0.3,
        )
        plot_line_of_best_fit(
            bucket_minimizers_gpuhours,
            second_col_df_name,
            "tokens_per_param",
            ax[1, 1],
            np.log(gpu_hours_xs),
            label="Binning",
            color="red",
        )

    ## FIGURE 2B: predict params ##
    plot_line_of_best_fit(
        flops_minimizers,
        "FLOPs",
        "params",
        axes[1, 0],
        np.log(flops_xs),
        save_name=f"params{save_postfix_2}",
    )

    plot_line_of_best_fit(
        gpuhours_minimizers,
        second_col_df_name,
        "params",
        axes[1, 1],
        np.log(gpu_hours_xs),
    )

    ## FIGURE 2C: predict tokens ##
    plot_line_of_best_fit(
        flops_minimizers,
        "FLOPs",
        "tokens",
        axes[2, 0],
        np.log(flops_xs),
        save_name=f"tokens{save_postfix_2}",
    )

    plot_line_of_best_fit(
        gpuhours_minimizers,
        second_col_df_name,
        "tokens",
        axes[2, 1],
        np.log(gpu_hours_xs),
    )

    ## FIGURE 2D: predict tokens ##
    for row_num, ax in [(3, axes), (1, axes_2)]:
        plot_line_of_best_fit(
            flops_minimizers,
            "FLOPs",
            "tokens_per_param",
            ax[row_num, 0],
            np.log(flops_xs),
            save_name=f"tokens_per_param{save_postfix_2}",
        )

        plot_line_of_best_fit(
            gpuhours_minimizers,
            second_col_df_name,
            "tokens_per_param",
            ax[row_num, 1],
            np.log(gpu_hours_xs),
        )

    add_existing_lines(flops_xs, axes[1, 0], axes[2, 0], axes[3, 0])
    add_existing_lines(flops_xs, None, None, axes_2[1, 0])
    handles, labels = [], []

    # Collect all handles and labels from subplots
    # Add shared legend
    for f, this_axes in [(fig, axes), (fig_2, axes_2)]:
        handles, labels = [], []
        for ax_row in this_axes:  # axes is a 2D array
            for ax in ax_row:
                ax.grid(True, which="major", linestyle="--", linewidth=0.5)
                h, l = ax.get_legend_handles_labels()
                handles.extend(h)
                labels.extend(l)

        unique = OrderedDict(zip(labels, handles))
        f.legend(
            unique.values(),
            unique.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1) if f == fig_2 else (0.5, -0.0),
            ncol=3 if f == fig_2 else 4,
        )
        f.tight_layout(rect=[0, 0.05, 1, 1])
        plt.grid()

    prefix = f"{SAVE_DIR}/figure_2{'_6N' if all(df.flops_per_token == df.flops_per_tok_col) else ''}{save_postfix}"
    fig.savefig(prefix + "_full.pdf", bbox_inches="tight")
    fig_2.savefig(prefix + ".pdf", bbox_inches="tight")

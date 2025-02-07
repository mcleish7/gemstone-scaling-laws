import numpy as np
from functools import partial
import pandas as pd
from plotting_utils import (
    flops_per_token_gqa,
    param_counter,
    param_counter_relaxed,
    Y_Axis,
    our_pred_func,
    plot_others,
    plot_industrial,
    add_flops_to_df,
    import_times_new_roman,
    add_a_line,
)
import matplotlib.pyplot as plt
import time
import swifter
import os
import json
import argparse
from matplotlib import font_manager
from collections import OrderedDict
from approach_3_brute_force_optim import optimize_row

import_times_new_roman(font_manager, plt, font_size=20)
plt.rcParams["lines.linewidth"] = 5


def get_laws(name, law_folder, width_depth_params=False):
    """
    wd_law takes input [width, depth, tokens]
    params_law takes input [params, tokens]
    """
    if width_depth_params:
        wd = pd.read_json(
            f"{law_folder}/approach_3_width_depth_parameters_grid_search_0_{name}.json"
        )
    else:
        wd = pd.read_json(
            f"{law_folder}/approach_3_width_depth_grid_search_0_{name}.json"
        )

    wd_exps = wd["exponents"].item()
    wd_coefs = wd["coefficients"].item()
    wd_e = wd["irreducible_error"].item()
    wd_law = partial(our_pred_func, exps=wd_exps, coefs=wd_coefs, e=wd_e)

    params = pd.read_json(
        f"{law_folder}/approach_3_parameters_grid_search_0_{name}.json"
    )
    params_exps = params["exponents"].item()
    params_coefs = params["coefficients"].item()
    params_e = params["irreducible_error"].item()
    params_law = partial(
        our_pred_func, exps=params_exps, coefs=params_coefs, e=params_e
    )

    return wd_law, params_law


def add_cols_to_df_apply(
    df,
    params_law,
    wd_law,
    width_depth_params,
    no_embeds,
    relaxed,
):
    df["FLOPs_per_token"] = df.swifter.apply(
        lambda row: flops_per_token_gqa(row["width"], row["depth"]), axis=1
    )
    df["FLOPs"] = df["FLOPs_per_token"] * df["tokens"]
    df["params"] = df.swifter.apply(
        lambda row: param_counter(row["width"], row["depth"]), axis=1
    )

    if relaxed and no_embeds:
        df["params_with_embeds"] = df["params"]
        df["params"] = df.swifter.apply(
            lambda row: param_counter_relaxed(row["width"], row["depth"], vocab_size=0),
            axis=1,
        )
    elif relaxed:
        df["params_with_embeds"] = df["params"]
        df["params"] = df.swifter.apply(
            lambda row: param_counter_relaxed(row["width"], row["depth"], vocab_size=0),
            axis=1,
        )
    elif no_embeds:
        df["params_with_embeds"] = df["params"]
        df["params"] = df.swifter.apply(
            lambda row: param_counter(row["width"], row["depth"], vocab_size=0), axis=1
        )
    else:
        df["params_with_embeds"] = df["params"]

    df["FLOPs_6N"] = 6 * df["params"] * df["tokens"]

    df["params_pred_loss"] = df.swifter.apply(
        lambda row: params_law([row["params"], row["tokens"]]), axis=1
    )
    df["wd_ratio"] = df.swifter.apply(lambda row: row["width"] / row["depth"], axis=1)

    if width_depth_params:
        df["wd_pred_loss"] = df.swifter.apply(
            lambda row: wd_law(
                [row["width"], row["depth"], row["params"], row["tokens"]]
            ),
            axis=1,
        )
    else:
        df["wd_pred_loss"] = df.swifter.apply(
            lambda row: wd_law([row["width"], row["depth"], row["tokens"]]), axis=1
        )

    return df


def make_data(
    add_cols_to_df_partial_with_laws,
    lr_ablation,
    cooldown,
    width_depth_params,
    over_120,
):
    widths = np.logspace(8, 17, 2**10 + 1, base=2).astype(int)
    tokens = np.logspace(9, 13.5, 500, base=10).astype(int)
    if over_120:
        tokens = np.logspace(8.5, 13.5, 500, base=10).astype(int)

    data = []
    widths = np.unique(widths)
    for width in widths:
        depths = np.logspace(
            max(0, np.log10(width // (2**11))), np.log10(width * (2**4)), 800, base=10
        ).astype(int)

        depths = np.unique(depths)
        depths = depths[depths > 0]

        repeated_depths = np.repeat(depths, tokens.size)
        repeated_widths = np.full(repeated_depths.shape, width)
        tiled_tokens = np.tile(tokens, depths.size)
        data.append(np.column_stack((repeated_depths, repeated_widths, tiled_tokens)))

    data = np.vstack(data)
    df = pd.DataFrame(data, columns=["depth", "width", "tokens"])

    if add_cols_to_df_partial_with_laws is not None:
        start_time = time.time()
        df = add_cols_to_df_partial_with_laws(df)
        elapsed_time = time.time() - start_time
        print(f"Time taken: {elapsed_time:.2f} seconds")

    return df


def plotting(
    df,
    x_axis_col,
    y_axis_col,
    ax,
    ax_2,
    num_buckets=100,
    label=None,
    color=None,
    loo=False,
    min_val=None,
    max_val=None,
    no_wd_ratio=False,
    markersize=6,
    save_postfix=None,
    add_our_data=None,
    s_param=80,
    relaxed=False,
    wd_law=None,
):
    # Define bucket edges based on x-axis values using log scale
    x_max = max_val if max_val is not None else df[x_axis_col].max()
    x_min = min_val if min_val is not None else df[x_axis_col].min()

    # Ensure x_min and x_max are positive for log scale
    if x_min <= 0 or x_max <= 0:
        raise ValueError(
            "x_axis_col must contain positive values for log scale binning."
        )

    log_bins = np.linspace(np.log10(x_min), np.log10(x_max), num=num_buckets + 1)
    bins = 10**log_bins  # Convert back from log scale to original scale

    # Assign each x-axis value to a bucket
    df["bucket"] = pd.cut(df[x_axis_col], bins=bins, include_lowest=True)

    # Initialize lists to store bucket left endpoints and best configurations
    bucket_lefts = []
    best_wd_ratios = []
    best_params = []
    best_tokens = []
    best_tokens_per_param = []
    best_widths = []
    best_depths = []

    mins = []
    # Iterate over each bucket
    for interval in df["bucket"].cat.categories:
        # Filter the DataFrame for the current bucket
        bucket_df = df[df["bucket"] == interval]

        if not bucket_df.empty:
            # Find the row with the minimum y_axis_col value in the current bucket
            min_row = bucket_df.loc[bucket_df[y_axis_col].idxmin()]

            if relaxed and not no_wd_ratio:
                assert wd_law is not None, "wd law is none cannot minimize"
                min_row = optimize_row(min_row, wd_law, relaxed)
            mins.append(min_row)

            # Append the left endpoint of the bucket
            bucket_lefts.append(interval.left)
            # Append the best configuration for each parameter
            best_wd_ratios.append(min_row["wd_ratio"])
            best_params.append(min_row["params"])
            best_tokens.append(min_row["tokens"])
            best_tokens_per_param.append(min_row["tokens"] / min_row["params"])
            best_widths.append(min_row["width"])
            best_depths.append(min_row["depth"])
    if save_postfix is not None:
        mins_df = pd.DataFrame(mins)
        mins_df.to_parquet(
            f"plotters/data_cache/mins_{save_postfix}.parquet", index=False
        )

    # Plotting for wd_ratio
    for axes in [ax, ax_2]:
        if axes is None:
            continue
        if not no_wd_ratio:
            axes[0].plot(
                bucket_lefts,
                best_wd_ratios,
                marker="o",
                linestyle="-",
                color="r",
                label=label,
                markersize=markersize,
            )
            axes[0].set_ylabel("Width-Depth\nRatio")
            axes[0].grid(True)
            if not loo and x_axis_col == "FLOPs":
                plot_industrial(
                    axes[0], "wd_ratio", plt, no_qwen=False, s_param=s_param
                )

        axes[1].plot(
            bucket_lefts,
            best_tokens_per_param,
            marker="o",
            linestyle="-",
            color="r",
            label=label,
            markersize=markersize,
        )
        axes[1].set_ylabel("Tokens/Param")
        axes[1].grid(True)

    # Plotting for params
    ax[2].plot(
        bucket_lefts,
        best_params,
        marker="o",
        linestyle="-",
        color="r",
        label=label,
        markersize=markersize,
    )
    ax[2].set_ylabel("Params")
    ax[2].grid(True)
    if not loo and x_axis_col == "FLOPs":
        plot_industrial(ax[2], Y_Axis.PARAMS, plt, no_qwen=False, s_param=s_param)

    # Plotting for tokens
    ax[3].plot(
        bucket_lefts,
        best_tokens,
        marker="o",
        linestyle="-",
        color="r" if color is None else color,
        label=label,
        markersize=markersize,
    )
    ax[3].set_ylabel("Tokens")
    ax[3].grid(True)
    if not loo and x_axis_col == "FLOPs":
        plot_industrial(ax[3], Y_Axis.DATA, plt, no_qwen=False, s_param=s_param)

    ## width and depth
    if not no_wd_ratio:
        ax[4].plot(
            bucket_lefts,
            best_widths,
            marker="o",
            linestyle="-",
            color="r" if color is None else color,
            label=label,
            markersize=markersize,
        )
        ax[4].set_ylabel("Width")
        ax[4].grid(True)
        ax[5].plot(
            bucket_lefts,
            best_depths,
            marker="o",
            linestyle="-",
            color="r" if color is None else color,
            label=label,
            markersize=markersize,
        )
        ax[5].set_ylabel("Depth")
        ax[5].grid(True)


def add_other_lines(
    params_ax, tokens_ax, tokens_per_param_ax, xs, law_folder, col_num, law_name
):
    labels = ["Chinchilla-epochai-reported", "Kaplan"]
    if params_ax is not None:
        plot_others(labels, params_ax, xs, Y_Axis.PARAMS)
    if tokens_ax is not None:
        plot_others(labels, tokens_ax, xs, Y_Axis.DATA)
    if tokens_per_param_ax is not None:
        plot_others(labels, tokens_per_param_ax, xs, Y_Axis.TOKENS_PER_PARAM)

    def predict(x, coef, intercept):
        return np.exp(coef[0] * np.log(x) + intercept)

    approach_1_postfix_map = {"_cool_end": "_cooldown"}
    approach_1_postfix = f"_{law_name}"
    approach_1_postfix = approach_1_postfix_map.get(
        approach_1_postfix, approach_1_postfix
    )
    approach_1_postfix = approach_1_postfix.replace("_hot", "")

    if (col_num == 1) and (tokens_ax is not None) and ([params_ax is not None]):
        # params approach 1
        with open(
            f"parameters/approach_1_linear_regression_params{approach_1_postfix}.json",
            "r",
        ) as f:
            regression_params = json.load(f)
        predictions = predict(
            xs, regression_params["coef"], regression_params["intercept"]
        )
        params_ax.plot(xs, predictions, color="green", label="Approach 1")

        # tokens approach 1
        with open(
            f"parameters/approach_1_linear_regression_tokens{approach_1_postfix}.json",
            "r",
        ) as f:
            regression_params = json.load(f)
        predictions = predict(
            xs, regression_params["coef"], regression_params["intercept"]
        )
        tokens_ax.plot(xs, predictions, color="green", label="Approach 1")

        if os.path.exists(
            f"{law_folder}/approach_3_parameters_grid_search_0_{law_name}.json"
        ):
            with open(
                f"{law_folder}/approach_3_parameters_grid_search_0_{law_name}.json", "r"
            ) as file:

                data = json.load(file)

                A = np.exp(data["coefficients"]["0"][0])
                B = np.exp(data["coefficients"]["0"][1])
                alpha = data["exponents"]["0"][0]
                beta = data["exponents"]["0"][1]
                label = "Approach 3 params, Chinchilla eq 4"

            for axes, y_axis in [(params_ax, Y_Axis.PARAMS), (tokens_ax, Y_Axis.DATA)]:
                add_a_line(
                    ax=axes,
                    xs=xs,
                    params_coeff=A,
                    tokens_coeff=B,
                    params_exp=alpha,
                    tokens_exp=beta,
                    label=label,
                    y_axis=y_axis,
                    color_arg="blue",
                )


def scatter_flops_vs_loss(df, x_axis_col, y_axis_col, ax):
    ax.scatter(df[x_axis_col], df[y_axis_col], label=y_axis_col, alpha=0.5)
    ax.set_xlabel(x_axis_col)
    ax.set_ylabel(y_axis_col)
    ax.set_title(f"{x_axis_col} vs {y_axis_col}")
    ax.legend(loc="upper right")
    ax.set_xscale("log")  # log scale x axis
    ax.set_yscale("log")  # log scale y axis
    ax.grid(True)


def plot_flops_vs_loss(df, law_name, save_postfix):
    law_name = law_name.replace("_no_embeds", "")
    if law_name == "hot":
        our_data = pd.read_json(
            f"wandb_dfs/wandb_df_for_fitting_hot_477.jsonl",
            orient="records",
            lines=True,
        )
    else:
        our_data = pd.read_json(
            f"wandb_dfs/wandb_df_for_fitting_{law_name}.jsonl",
            orient="records",
            lines=True,
        )
    our_data = add_flops_to_df(our_data)
    our_data = our_data[our_data["tokens"] > 0]

    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 10), sharey="row", sharex="col"
    )

    ax[0].scatter(
        our_data["FLOPs"], our_data["final_loss"], label="Our Runs", alpha=0.3
    )
    ax[1].scatter(
        our_data["FLOPs"], our_data["final_loss"], label="Our Runs", alpha=0.3
    )

    if len(df) > 10000:  # sample if too many points to handle
        step = len(df) // 10000
        df_sampled = df.iloc[::step].head(10000)
    else:
        df_sampled = df.copy

    scatter_flops_vs_loss(df_sampled, "FLOPs", "wd_pred_loss", ax[0])
    scatter_flops_vs_loss(df_sampled, "FLOPs", "params_pred_loss", ax[1])
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig(f"figures/approach_3_brute_force_flops_vs_pred_loss_{save_postfix}.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Script with --lr_ablation flag.")
    parser.add_argument("--lr_ablation", action="store_true")
    parser.add_argument("--cooldown", action="store_true")
    parser.add_argument("--over_100", action="store_true")
    parser.add_argument("--over_120", action="store_true")
    parser.add_argument("--ignore_cache", action="store_true")
    parser.add_argument("--width_depth_params", action="store_true")
    parser.add_argument("--no_embeds", action="store_true")
    parser.add_argument("--relaxed", action="store_true")
    parser.add_argument("--law_folder", type=str, default="parameters_delta-4")
    args = parser.parse_args()

    law_name = "hot"
    if args.lr_ablation:
        law_name = "lr_ablation_hot"
    elif args.cooldown:
        law_name = "cool_end"
    elif args.over_100:
        law_name = "hot_100b+"
    elif args.over_120:
        law_name = "hot_120b+_only"

    if args.no_embeds:
        law_name += "_no_embeds"

    law_folder = args.law_folder
    num_buckets = 13
    ignore_cache = args.ignore_cache

    save_postfix = f"{law_name}{'_width_depth_params'if args.width_depth_params else ''}{'_relaxed' if args.relaxed else ''}"
    data_path = f"plotters/data_cache/{law_folder}_{save_postfix}.parquet"

    wd_law, params_law = get_laws(
        law_name,
        law_folder,
        width_depth_params=args.width_depth_params,
    )
    os.makedirs("data_cache", exist_ok=True)
    if os.path.exists(data_path) and not ignore_cache:
        print(
            "reading from cache, remember this means that changes to the data creation are not being used"
        )
        df = pd.read_parquet(data_path)
    else:
        add_cols_to_df_partial_with_laws = partial(
            add_cols_to_df_apply,
            params_law=params_law,
            wd_law=wd_law,
            width_depth_params=args.width_depth_params,
            no_embeds=args.no_embeds,
            relaxed=args.relaxed,
        )
        df = make_data(
            add_cols_to_df_partial_with_laws,
            args.lr_ablation,
            args.cooldown,
            args.width_depth_params,
            args.over_120,
        )
        df.to_parquet(data_path)

    print(f"{df['params'].max():.2e}")
    print(f"{df['tokens'].max():.2e}")
    print(f"{df['FLOPs'].max():.2e}")

    x_axis = "FLOPs"
    min_val = 1e18
    max_val = 1e26

    # take out the rows we don't care about
    df = df[(df[x_axis] >= min_val) & (df[x_axis] <= max_val)]

    plot_flops_vs_loss(df, law_name, save_postfix)
    print(f"Dataset is {len(df)} rows")

    fig, ax = plt.subplots(
        nrows=6, ncols=2, figsize=(16, 18), sharey="row", sharex="col"
    )
    fig_2, ax_2 = plt.subplots(
        nrows=1, ncols=2, figsize=(16, 6), sharey=False, sharex=True
    )

    plotting(
        df,
        x_axis_col=x_axis,
        y_axis_col="wd_pred_loss",
        ax=ax[:, 0],
        ax_2=ax_2,
        num_buckets=num_buckets,
        min_val=min_val,
        max_val=max_val,
        save_postfix=f"{save_postfix}_wd",
        add_our_data=law_name,
        relaxed=args.relaxed,
        wd_law=wd_law,
    )
    plotting(
        df,
        x_axis_col=x_axis,
        y_axis_col="params_pred_loss",
        ax=ax[:, 1],
        ax_2=None,
        num_buckets=num_buckets,
        min_val=min_val,
        max_val=max_val,
        no_wd_ratio=True,
        label="Approach 3 Brute Force",
        save_postfix=f"{save_postfix}_params",
        relaxed=args.relaxed,
        wd_law=wd_law,
    )

    # add in the other lines
    flops_xs = np.exp(
        np.linspace(
            np.log(df[df.FLOPs > 0].FLOPs.min()),
            np.log(df.FLOPs.max()),
            100,
        )
    )

    for i in range(2):  # for left and right column
        add_other_lines(
            ax[2, i],
            ax[3, i],
            ax[1, i],
            flops_xs,
            law_folder,
            col_num=i,
            law_name=law_name,
        )
        add_other_lines(
            None,
            None,
            ax_2[1],
            flops_xs,
            law_folder,
            col_num=i,
            law_name=law_name,
        )
        ax[5, i].set_xlabel("FLOPs")
        ax_2[i].set_xlabel("FLOPs")

    unique = OrderedDict(
        (l, h)
        for h, l in (
            list(zip(*ax[3, 1].get_legend_handles_labels()))
            + list(zip(*ax[3, 0].get_legend_handles_labels()))
        )
    )
    labels, handles = zip(*unique.items())
    fig.legend(
        handles,
        labels,
        loc="lower center",  # Adjust this as needed
        bbox_to_anchor=(0.5, 0.0),  # Change to (0.5, 1.05) for top placement
        ncol=4,  # Number of columns for the legend
    )
    unique = OrderedDict(
        (l, h)
        for h, l in (
            list(zip(*ax_2[0].get_legend_handles_labels()))
            + list(zip(*ax_2[1].get_legend_handles_labels()))
        )
    )
    labels, handles = zip(*unique.items())
    fig_2.legend(
        handles,
        labels,
        loc="lower center",  # Adjust this as needed
        bbox_to_anchor=(0.5, 0.0),  # Change to (0.5, 1.05) for top placement
        ncol=4,  # Number of columns for the legend
    )

    for i, axis in enumerate(ax.flatten()):
        axis.set_xscale("log")  # log scale x axis
        axis.set_yscale("log")  # log scale y axis

    for i, axis in enumerate(ax_2.flatten()):
        axis.set_xscale("log")  # log scale x axis
        axis.set_yscale("log")  # log scale y axis

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    fig_2.tight_layout(rect=[0, 0.34, 1, 1])
    fig.savefig(f"figures/approach_3_brute_force_{save_postfix}_{x_axis}_full.pdf")
    fig_2.savefig(f"figures/approach_3_brute_force_{save_postfix}_{x_axis}.pdf")


if __name__ == "__main__":
    main()

# %%
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from plotting_utils import (
    flops_per_token_gqa,
    our_pred_func,
    param_counter_relaxed,
    import_times_new_roman,
)
from functools import partial
import scipy.optimize as opt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import json

import_times_new_roman(matplotlib.font_manager, plt, font_size=28)

# %%
is_wd_law = True
mins = pd.read_parquet(
    "data_cache/mins_hot_100b+_width_depth_params_relaxed_wd.parquet"
)
mins["params_active_precise"] = mins["params"]
# %%
n_rows = len(mins)
indices = np.linspace(0, n_rows - 1, 5, dtype=int)
indices = indices[1:]
indices = [2, 5, 9, 12]
df = mins.iloc[indices].reset_index(drop=True)
df


# %%
def get_laws(name, law_folder="parameters_delta-4", width_depth_params=True):
    """
    wd_law takes input [width, depth, tokens]
    params_law takes input [params, tokens]
    """
    if width_depth_params:
        wd = pd.read_json(
            f"../{law_folder}/approach_3_width_depth_parameters_grid_search_0_{name}.json"
        )
    else:
        wd = pd.read_json(
            f"../{law_folder}/approach_3_width_depth_grid_search_0_{name}.json"
        )

    wd_exps = wd["exponents"].item()
    wd_coefs = wd["coefficients"].item()
    wd_e = wd["irreducible_error"].item()
    wd_law = partial(our_pred_func, exps=wd_exps, coefs=wd_coefs, e=wd_e)

    params = pd.read_json(
        f"../{law_folder}/approach_3_parameters_grid_search_0_{name}.json"
    )
    params_exps = params["exponents"].item()
    params_coefs = params["coefficients"].item()
    params_e = params["irreducible_error"].item()
    params_law = partial(
        our_pred_func, exps=params_exps, coefs=params_coefs, e=params_e
    )
    e = wd_e if is_wd_law else params_e
    return wd_law, params_law, e


wd_law, params_law, irreducible_error = get_laws(
    name="hot_100b+", width_depth_params=is_wd_law
)

# %%


def flops_tokens_to_model_size(flops, tokens):
    flops_per_token = flops / tokens

    # minimise wd_law(width, depth, tokens)
    # s.t.
    # flops_per_token_gqa(width, depth)=flops_per_token
    def constraint(params):
        width, depth = params
        return flops_per_token_gqa(width, depth) - flops_per_token

    width_grid_step = 128
    depth_grid_step = 8
    width_values = np.arange(256, 9216 + width_grid_step, width_grid_step)
    depth_values = np.arange(2, 128 + depth_grid_step, depth_grid_step)

    best_result = None
    best_width, best_depth = None, None

    # Perform grid search
    for width_init in width_values:
        for depth_init in depth_values:
            initial_guess = [width_init, depth_init]
            try:
                # Perform optimization
                result = opt.minimize(
                    lambda params: wd_law(
                        [
                            params[0],
                            params[1],
                            param_counter_relaxed(params[0], params[1]),
                            tokens,
                        ]
                    ),  # Objective function
                    initial_guess,  # Initial guess
                    constraints={"type": "eq", "fun": constraint},  # Constraint
                    bounds=[(256, None), (1, None)],  # Bounds for width and depth
                )

                # Check if optimization was successful
                if result.success:
                    if best_result is None or result.fun < best_result.fun:
                        best_result = result
                        best_width, best_depth = result.x
            except Exception as e:
                print(f"Optimization failed for init {initial_guess}: {e}")

    if best_result is not None:
        return best_width, best_depth
    else:
        return None


def get_new_row(row, overtrain_factor):
    overtrained_tokens = row["tokens"] * np.sqrt(overtrain_factor)
    # now need to find model size such that we have the same FLOPs
    overtrained_model_size = flops_tokens_to_model_size(
        row["FLOPs"], overtrained_tokens
    )
    if overtrained_model_size is None:
        # failed to optimise
        return None

    overtrained_flops_per_token = flops_per_token_gqa(*overtrained_model_size)
    overtrained_flops = overtrained_flops_per_token * overtrained_tokens
    overtrained_params = param_counter_relaxed(*overtrained_model_size)

    overtrained_wd_loss = wd_law(
        [*overtrained_model_size, overtrained_params, overtrained_tokens]
    )
    overtrained_params_loss = params_law([overtrained_params, overtrained_tokens])

    # what would happen with this model size if we didn't increase the tokens
    undertrained_wd_loss = wd_law(
        [*overtrained_model_size, overtrained_params, row["tokens"]]
    )
    undertrained_params_loss = params_law([overtrained_params, row["tokens"]])
    print(overtrained_wd_loss)
    new_row = {
        "FLOPs": overtrained_flops,
        "wd_pred_loss": overtrained_wd_loss,
        "params_pred_loss": overtrained_params_loss,
        "params_active_precise": overtrained_params,
        "FLOPs_per_token": overtrained_flops_per_token,
        "width": overtrained_model_size[0],
        "depth": overtrained_model_size[1],
        "tokens": overtrained_tokens,
        "undertrained_wd_pred_loss": undertrained_wd_loss,
        "undertrained_params_pred_loss": undertrained_params_loss,
    }
    return new_row


# %%
overtrained = {}
overtrain_factors = [2**i for i in range(-4, 5)]  # Powers of 2 from -3 to 3
for overtrain_factor in tqdm(overtrain_factors, desc="Processing factors"):

    with ProcessPoolExecutor(max_workers=96) as executor:
        # Convert DataFrame rows to a list of dictionaries (or tuples)
        rows = [row for _, row in df.iterrows()]

        # Use map to process rows in parallel
        results = executor.map(get_new_row, rows, [overtrain_factor] * len(rows))
    new_rows = list(results)
    # remove the None values where we failed to optimise
    new_rows = [row for row in new_rows if row is not None]

    temp_df = pd.DataFrame(new_rows)
    # If we're on the boundary this is not a solution.
    # In terms of brute force we have reached the "edge of the box"
    temp_df = temp_df[temp_df["width"] > 257]
    temp_df = temp_df[temp_df["depth"] > 2]
    overtrained[overtrain_factor] = temp_df

# %%
wd_or_params = "wd" if is_wd_law else "params"
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, x_axis in enumerate(
    [
        f"{wd_or_params}_pred_loss",
        "params_active_precise",
        "tokens",
    ]
):

    for k, (overtrain_size, overtrained_df) in enumerate(overtrained.items()):
        axes[i].plot(
            overtrained_df["FLOPs"],
            overtrained_df[x_axis],
            marker="x",
            linestyle="-",
            # color=cmap[k],
            label=f"x{overtrain_size} {'params' if not wd_or_params else ''} Law",
            linewidth=2.5,
        )

    axes[i].set_xscale("log")
    axis_postfix = ""
    if i != 0:
        axes[i].set_yscale("log")
        axis_postfix = " (Log Scale)"

    axes[i].grid(True)
    axis_name_dict = {
        "depth": "Depth",
        "width": "Width",
        "tokens": "Tokens",
        "params_active_precise": "Params",
        "wd_pred_loss": "Loss",
    }
    axes[i].set_ylabel(axis_name_dict[x_axis] + axis_postfix)

# Adding labels and title
handles, labels = axes[
    0
].get_legend_handles_labels()  # Collect handles and labels from one of the axes
fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.xlabel("Total FLOPs")
plt.tight_layout()
plt.savefig(
    f"../figures/parabola_overtrained_predictions_{wd_or_params}_hot.pdf",
    bbox_inches="tight",
)
# %%
with open(f"../data/our_val_loss_data.json", "r") as json_file:
    data = json.load(json_file)

industry_df = pd.DataFrame(data)

industry_df["plot_name"] = industry_df["hf_name"].apply(lambda x: x.split("/")[-1])
unique_names = industry_df["plot_name"].unique()
num_colors = len(unique_names)
colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

# %%
plt.clf()
import_times_new_roman(matplotlib.font_manager, plt, font_size=18)

num_points = 4
overtrain_sizes = list(overtrained.keys())
to_plot = [[] for _ in range(num_points)]
titles = []
for overtrain_size, overtrained_df in overtrained.items():
    for row_index, (_, row) in enumerate(overtrained_df.iterrows()):
        to_plot[row_index].append(row["wd_pred_loss"])
        titles.append(row["FLOPs"])

fig, axes = plt.subplots(
    nrows=num_points,
    ncols=1,
    figsize=(8, 12),
    sharex=True,
    gridspec_kw={"hspace": 0.35},
)
for i in range(num_points):
    axes[i].plot(overtrain_sizes, to_plot[i], marker="o", label="Our Predicted Loss")
    axes[i].set_xscale("log")
    axes[i].set_ylabel(f"Loss")
    axes[i].set_title(f"FLOPs = {titles[i]:.5e}")

    bound = 0.01
    subset_industry_df = industry_df[
        (industry_df[f"loss"] >= (min(to_plot[i]) - bound))
        & (industry_df[f"loss"] <= (max(to_plot[i]) + bound))
    ]

    for color, name in zip(colors, unique_names):
        subset = subset_industry_df[subset_industry_df["plot_name"] == name]
        if subset.empty:
            continue
        for index, row in subset.iterrows():
            if name == "Llama-3.1-70B":
                continue
            axes[i].axhline(
                y=row[f"loss"],
                color=color,
                linestyle="--",
                label=name,
                linewidth=2.5,
            )

    axes[i].legend(loc="center left", bbox_to_anchor=(1, 0.5))
axes[3].set_xlabel("Overtraining Factor (log)")

plt.tight_layout()
plt.savefig(
    f"../figures/parabola_overtrained_predictions_{wd_or_params}_hot_2.pdf",
    bbox_inches="tight",
)
# %%

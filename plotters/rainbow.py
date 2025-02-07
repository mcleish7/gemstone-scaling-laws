# %%
import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from plotting_utils import (
    Y_Axis,
    add_a_line,
    plot_others,
    import_times_new_roman,
)

import_times_new_roman(matplotlib.font_manager, plt, font_size=18)

# %%
# We'll read the approach 1 lines from these:
temps = {
    "hot": "<=100b Tokens",
    "hot_no_embeds": "<=100b Tokens No Embeds",
    "cool_end": "Cooldown",
    "cool_end_no_embeds": "Cooldown No Embeds",
    "lr_ablation_hot": "LR Ablation",
    "lr_ablation_hot_no_embeds": "LR Ablation No Embeds",
    "hot_sampled_like_chinchilla": "Chinchilla Reduced Sampling (Our Data)",
    "lr_ablation_hot_sampled_like_chinchilla": "LR Ablation Chinchilla\nReduced Sampling (Our Data)",
    "slim_chinchilla": "Chinchilla Reduced Sampling\n(Hoffman et al. data)",
    "hot_120b+_only": ">120b Tokens Only",
    "hot_120b+_only_no_embeds": ">120b Tokens Only No Embeds",
    "hot_100b+": "All Tokens",
    "hot_100b+_no_embeds": "All Tokens No Embeds",
    "hot_100b+_512x": "All Tokens, width=512 Only",
}

main_lines = []
for temperature, legend_name in temps.items():
    json_path = (
        f"../parameters_delta-4/approach_3_parameters_grid_search_0_{temperature}.json"
    )
    if not os.path.exists(json_path):
        print(f"Warning: File not found: {json_path}, skipping.")
        continue

    with open(json_path, "r") as file:
        data = json.load(file)

    A = data["coefficients"]["0"][0]
    B = data["coefficients"]["0"][1]
    alpha = data["exponents"]["0"][0]
    beta = data["exponents"]["0"][1]

    main_lines.append(
        {
            "type": "main",
            "A": A,
            "B": B,
            "alpha": alpha,
            "beta": beta,
            "line_style": "--" if "No Embeds" in legend_name else "-",
            "label": legend_name,
        }
    )

# %%
y_axis = Y_Axis.PARAMS
# Approach 1 lines
approach1_lines = []
approach1_specs = [
    ("", "-", "<=100b Tokens"),
    ("_cooldown", "--", "Cooldown"),
    ("_lr_ablation", ":", "LR Ablation"),
    ("_100b+", "-", "All Tokens"),
    ("_120b+_only", ":", ">120b Tokens Only"),
]
law_folder_approach_1 = "../parameters"
for law_name, line_style, legend_name in approach1_specs:
    for embeds in ["", "_no_embeds"]:
        # Build the filename
        if y_axis == Y_Axis.PARAMS:
            fname = f"approach_1_linear_regression_params{law_name}{embeds}.json"
        else:
            # y_axis == Y_Axis.DATA (or TOKENS_PER_PARAM, but you only had two in snippet)
            fname = f"approach_1_linear_regression_tokens{law_name}{embeds}.json"

        json_path = os.path.join(law_folder_approach_1, fname)
        if not os.path.exists(json_path):
            print(f"Warning: Approach 1 file not found: {json_path}, skipping.")
            continue

        with open(json_path, "r") as f:
            regression_params = json.load(f)

        # Typically: regression_params["coef"] = [slope], regression_params["intercept"] = ...
        coef = regression_params["coef"]
        intercept = regression_params["intercept"]

        label = (
            f"Approach 1 {legend_name}{' No Embeds' if embeds == '_no_embeds' else ''}"
        )
        if label == f"Approach 1 >120b Tokens Only No Embeds":
            label = f"Approach 1 >120b Tokens Only\nNo Embeds"
        approach1_lines.append(
            {
                "label": label,
                "type": "approach1",
                "coef": coef,
                "intercept": intercept,
                "line_style": ":" if embeds == "_no_embeds" else "-.",
            }
        )

# %%
# Combine all lines & define the final order
all_lines = main_lines + approach1_lines


def get_line_value(line_info, x, y_axis):
    """
    Return the y-axis value at the given FLOPs = x for a single line_info dict.
    This is used to decide the sorting order (ascending by value at x).
    Effectively is just the plotting function but evaluated for one value
    """
    if line_info["type"] == "main":
        alpha = line_info["alpha"]
        beta = line_info["beta"]
        A = np.exp(line_info["A"])
        B = np.exp(line_info["B"])
        a = beta / (alpha + beta)
        b = 1 - a
        G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
        if y_axis == Y_Axis.PARAMS:
            return G * ((x / 6) ** a)
        elif y_axis == Y_Axis.DATA:
            return G ** (-1) * (x / 6) ** b
        elif y_axis == Y_Axis.TOKENS_PER_PARAM:
            # ratio = tokens / params
            return (G ** (-1) * (x / 6) ** b) / (G * (x / 6) ** a)
        else:
            raise ValueError("Unsupported y_axis in get_line_value (main).")
    else:
        # Approach 1 line: we have "coef" (list) and "intercept" (scalar).
        coef = line_info["coef"]  # e.g. [slope]
        intercept = line_info["intercept"]
        y = np.exp(coef[0] * np.log(x) + intercept)
        return y


x_sort = 1e28
all_lines_sorted = sorted(
    all_lines, key=lambda d: get_line_value(d, x_sort, y_axis=y_axis)
)

# Build color map
n_lines = len(all_lines_sorted)
colors = cm.rainbow(np.linspace(0, 1, n_lines))  # from red to violet

# %%
# Make the plot
plt.rcParams["lines.linewidth"] = 3

fig, ax = plt.subplots(figsize=(8, 6))
flops_xs = np.array([1e18, 1e28]).reshape(-1, 1)

# Plot each line
for i, line_info in enumerate(all_lines_sorted):
    color = colors[i]
    if line_info["type"] == "main":
        # It's a main line with A,B,alpha,beta
        add_a_line(
            ax=ax,
            xs=flops_xs,
            params_coeff=np.exp(line_info["A"]),
            tokens_coeff=np.exp(line_info["B"]),
            params_exp=line_info["alpha"],
            tokens_exp=line_info["beta"],
            label=line_info["label"],
            y_axis=y_axis,
            color_arg=color,
            line_style_arg=line_info["line_style"],
        )
    else:
        # It's an "approach1" line with a regression slope/intercept
        coef = line_info["coef"]  # e.g., [slope]
        intercept = line_info["intercept"]
        preds = np.exp(coef[0] * np.log(flops_xs) + intercept)
        ax.plot(
            flops_xs,
            preds,
            label=line_info["label"],
            color=color,
            linestyle=line_info["line_style"],
        )

names = ["Chinchilla-epochai-reported", "Kaplan"]
plot_others(names, ax, flops_xs, y_axis)

# Configure scales
ax.set_xscale("log")
ax.set_yscale("log")
axis_fontsize = 24
# Axis labels
ax.set_xlabel("FLOPs", fontsize=axis_fontsize)
if y_axis == Y_Axis.PARAMS:
    ax.set_ylabel("Params", fontsize=axis_fontsize)
elif y_axis == Y_Axis.DATA:
    ax.set_ylabel("Tokens", fontsize=axis_fontsize)
else:
    ax.set_ylabel("Tokens per Parameter ratio", fontsize=axis_fontsize)

tick_fontsize = 22
ax.tick_params(axis="x", labelsize=tick_fontsize)
ax.tick_params(axis="y", labelsize=tick_fontsize)
ax.grid(visible=True, which="major", linestyle="-", linewidth=0.75, color="gray")

# Legend & layout
ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), ncol=2)
fig.subplots_adjust(bottom=0.2)
fig.savefig("../figures/rainbow.pdf", bbox_inches="tight")

# %%

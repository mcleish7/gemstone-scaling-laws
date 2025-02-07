# %%
import pandas as pd
import matplotlib.pyplot as plt
import math
import matplotlib
from plotting_utils import import_times_new_roman

import_times_new_roman(matplotlib.font_manager, plt, font_size=32)
plt.rcParams["lines.linewidth"] = 3.5
# %%
df = pd.read_csv("../data/preflight_mup_olmo_variant.csv")
# %%
exclude_keystring = "inverse_sqrt_n_embd"


def filter_df(df, exclude_keystring):
    return df[~df["Name"].str.contains(exclude_keystring)]


df = filter_df(df, exclude_keystring)


# %%
def name_to_cols(name):
    # eg "PyGemma-2560x8_8-2048_32nodes_sinitTrue_lr2e-01_scalerinverse_sqrt_n_embd"
    # to {"width": 2560, "depth": 8, "lr": 2e-01, "scaler": "inverse_sqrt"}

    parts = name.split("-")
    width = int(parts[1].split("x")[0])
    depth = int(parts[1].split("x")[1].split("_")[0])
    lr = float(name.split("lr")[1].split("_scaler")[0])
    scaler = name.split("_scaler")[1].split("_n_embd")[0]
    return {"width": width, "depth": depth, "lr": lr, "scaler": scaler}


parsed_cols = df["Name"].apply(name_to_cols)
df[["width", "depth", "lr", "scaler"]] = pd.DataFrame(parsed_cols.tolist())
df["global_loss"] = df["scaling/global_loss"]
df.drop(columns=["Name", "_wandb", "scaling/global_loss"], inplace=True)
df


# %%
# group the df by width, depth, and scaler
# and plot the global_loss vs lr for each group
# connect the points with lines
def to_label_str(tuple):
    return f"{tuple[0]} x {tuple[1]}"


grouped = df.groupby(["width", "depth", "scaler"])
fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
for name, group in grouped:
    group = group.sort_values("lr")

    axes[0].plot(
        group["lr"], group["global_loss"], marker="o", label=to_label_str(name)
    )

    effective_lr = [
        lr * width * math.sqrt(depth)
        for lr, width, depth in zip(group["lr"], group["width"], group["depth"])
    ]
    group["effective_lr"] = effective_lr
    axes[1].plot(
        effective_lr, group["global_loss"], marker="o", label=to_label_str(name)
    )

for i in range(0, 2):
    axes[i].set_xscale("log")
    axes[i].set_ylim(0, 15)
    axes[i].grid()
    axes[i].xaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10.0, numticks=10)
    )

axes[0].set_xlabel(r"lr$_{base}$")
axes[1].set_xlabel(r"lr$_{eff}$")

axes[0].set_ylabel("Loss")


unique_handles_labels = set()  # Use a set to store unique (handle, label) pairs
handles, labels = [], []

for ax in axes:  # Iterate over each axislr
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in unique_handles_labels:
            unique_handles_labels.add(label)
            handles.append(handle)
            labels.append(label)

# Create a single legend for the entire figure, placed below the plot
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0.5, -0.57),
    title="Width x Depth",
    columnspacing=0.7,
    handletextpad=0.3,
)

plt.savefig("../figures/mup.pdf", bbox_inches="tight")
# %%

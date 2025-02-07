# %%
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from plotters.plotting_utils import (
    model_styles,
    add_flops_to_df,
    import_times_new_roman,
)

import_times_new_roman(font_manager, plt, font_size=24)
plt.rcParams["lines.linewidth"] = 2.5

# %%
df = pd.read_json(
    "wandb_dfs/wandb_df_for_fitting_hot_100b+_477.jsonl", orient="records", lines=True
)
dict_column = "final_loss"
df = add_flops_to_df(df)
df["legend_name"] = (
    df["run_name"].str.replace("PyGemma-", "").str.replace("_pretrain", "")
)
# %%
plt.figure(figsize=(14, 6))
cmap = plt.get_cmap("tab20", len(df["run_name"].unique()))

for i, (run_name, group) in enumerate(df.groupby("run_name")):
    color = model_styles[run_name]["color"]
    line_style = model_styles[run_name]["line_style"]

    plt.plot(
        group["tokens"],
        group["final_loss"],
        line_style,
        label=group["legend_name"].iloc[0],
        # Labels each line with the legend name of the group
        color=color,  # Use a different color for each run_name
    )

plt.xlabel("Tokens (log)")
plt.ylabel("Loss")
plt.legend(
    title="Width x Depth",
    bbox_to_anchor=(0.5, -0.2),
    loc="upper center",
    ncol=5,
)

plt.grid(which="major")
plt.xscale("log")

plt.savefig(
    f"figures/wandb_loss_vs_tokens.pdf",
    format="pdf",
    bbox_inches="tight",
)

# %%

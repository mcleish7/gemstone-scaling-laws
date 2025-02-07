# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%

epoch_df = pd.read_csv("data/epoch_chinchilla_svg_extracted_data.csv")
epoch_df["Training Tokens"] = epoch_df["Training FLOP"] / (6.0 * epoch_df["Model Size"])
epoch_df = epoch_df[["Model Size", "Training Tokens", "Training FLOP", "loss"]].dropna()
epoch_df.rename(
    columns={
        "Model Size": "params_active_precise",
        "Training Tokens": "tokens",
        "Training FLOP": "FLOPs",
        "loss": "final_loss",
    },
    inplace=True,
)

# fmt: off
slim_chinchilla_df = epoch_df[
    (
        ((epoch_df["params_active_precise"] >= 6e7) & (epoch_df["params_active_precise"] <= 8e7)) |
        ((epoch_df["params_active_precise"] >= 9e7) & (epoch_df["params_active_precise"] <= 11e7)) |
        ((epoch_df["params_active_precise"] >= 45e7) & (epoch_df["params_active_precise"] <= 50.5e7)) |
        ((epoch_df["params_active_precise"] >= 0.9e9) & (epoch_df["params_active_precise"] <= 1.1e9)) |
        ((epoch_df["params_active_precise"] >= 1.8e9) & (epoch_df["params_active_precise"] <= 2.2e9))
    )
].sort_values(by="tokens")

# fmt: on
print(len(slim_chinchilla_df))

# %%
# got by uncommenting the lines above and looking at the rows in the dataset
params_to_tokens_chinchilla = {
    50e6: [3.15e9, 6.53e9, 1.19e10, 1.3e10, 2.51e10],
    100e6: [3.24e9, 7.47e9, 1.3e10, 1.37e10],
    500e6: [5.76e9, 1.25e10, 2.4e10, 4.71e10],
    1e9: [1.57e9, 4.73e9, 7.73e9, 8.78e9, 1.58e10, 1.64e10, 3.27e10],
    2e9: [8.12e8, 8.05e9, 1.63e10, 2.43e10, 3.27e10, 4.88e10, 8.12e10, 1.24e11],
}
rounded_params_to_tokens = {
    key: [round(val / 2e9) * 2e9 for val in values]
    for key, values in params_to_tokens_chinchilla.items()
}
rounded_params_to_tokens[70e6] = rounded_params_to_tokens[50e6]
rounded_params_to_tokens

# %%
lr_ablation = False
if lr_ablation:
    df_json = f"wandb_dfs/wandb_df_for_fitting_lr_ablation_hot_477.jsonl"
    slim_chinchilla_df = slim_chinchilla_df[slim_chinchilla_df["tokens"] < 100e9]
else:
    df_json = f"wandb_dfs/wandb_df_for_fitting_hot_100b+_477.jsonl"

df = pd.read_json(
    df_json,
    orient="records",
    lines=True,
)
df["rounded_params"] = df["params_active_precise"].apply(
    lambda x: int(float(f"{x:.0e}"))
)  # removes the +/- 5% boundary so we can group over param counts
df["tokens"] = df["tokens"].round(-9)


# %%
filtered_df = df[
    df.apply(
        lambda row: row["tokens"]
        in rounded_params_to_tokens.get(row["rounded_params"], []),
        axis=1,
    )
]
df["wd_ratio"] = df["width"] / df["depth"]

my_list = [66758784, 94776832, 102643200, 497457920, 1047397120, 1904850944]
#  384x13 (29.54), 512x11 (46.54), 512x13 (39.38), 1280x15 (85.33), 1792x18 (99.56), 2048x27 (75.85)
filtered_df = filtered_df[filtered_df["params_active_precise"].isin(my_list)]
filtered_df
# %%
## SAVES DATA
if False:
    filtered_df.to_json(
        f"wandb_dfs/wandb_df_for_fitting{'_lr_ablation' if lr_ablation else ''}_hot_sampled_like_chinchilla.jsonl",
        orient="records",
        lines=True,
    )
    slim_chinchilla_df.to_json(
        f"wandb_dfs/wandb_df_for_fitting{'_lr_ablation' if lr_ablation else ''}_slim_chinchilla.jsonl",
        orient="records",
        lines=True,
    )

# %%
## makes the plot with chinchilla points and rough lines on

plt.figure(figsize=(10, 6))  # Optional: Adjust the size of the plot
plt.scatter(
    slim_chinchilla_df["params_active_precise"],
    slim_chinchilla_df["tokens"],
    alpha=1.0,
    label=f"Chinchilla ({len(slim_chinchilla_df)} points)",
)  # Use alpha for better visibility
plt.scatter(
    filtered_df["params_active_precise"],
    filtered_df["tokens"],
    alpha=0.7,
    label=f"Us ({len(filtered_df)} points)",
)  # Use alpha for better visibility

plt.xlabel("params")  # Label for x-axis
plt.ylabel("tokens")  # Label for y-axis
plt.grid(True)  # Optional: Add a grid for better readability
plt.legend()

# %%

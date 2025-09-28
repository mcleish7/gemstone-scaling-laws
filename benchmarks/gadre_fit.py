import pandas as pd
import scipy
import re
from plot_lighteval import param_counter, flops_per_token_gqa
import numpy as np
import matplotlib.pyplot as plt


def get_eval_df():
    def extract_model_info(model_name):
        """
        Extracts:
        - token_count: For "step" models (e.g. Gemstone-1792x18-step_00047700), token_count is the number after "step_";
                        For "main" models (e.g. Gemstone-3072x12-main), token_count is set to 80000.
        - model_label: A string "width x depth" (e.g. "1792x18") for legend.
        - width: The first extracted number.
        - depth: The second extracted number.
        """
        pattern = r"Gemstone-(\d+)x(\d+)-(?:step_(\d+)|main)"
        match = re.search(pattern, model_name)
        if match:
            width = int(match.group(1))
            depth = int(match.group(2))
            step_token = match.group(3)
            token_count = int(step_token) if step_token is not None else 83475
            prec_token_count = token_count * (2048 * 2048)
            token_count = round(prec_token_count / 1_000_000_000)
            model_label = f"{width}x{depth}"
            return token_count, prec_token_count, model_label, width, depth
        else:
            raise ValueError("Couldn't extract info from model name: " + model_name)

    # Create new DataFrame columns based on model names
    df_models = pd.read_parquet("light_eval_df.parquet")

    # Map each model (index) to its token count and model label.
    df_models[["token_count", "prec_token_count", "model_label", "width", "depth"]] = (
        pd.DataFrame(
            [extract_model_info(m) for m in df_models.index], index=df_models.index
        )
    )
    df_models["params"] = df_models.apply(
        lambda row: param_counter(row["width"], row["depth"]), axis=1
    )
    df_models["FLOPs_per_token"] = df_models.apply(
        lambda row: flops_per_token_gqa(row["width"], row["depth"]), axis=1
    )
    df_models["FLOPs"] = df_models["FLOPs_per_token"] * df_models["prec_token_count"]

    columns_to_average = [
        "arc:challenge",
        "arc:easy",
        "commonsense_qa",
        "hellaswag",
        "mmlu",
        "openbookqa",
        "piqa",
        "siqa",
        "winogrande",
    ]
    df_models["avg_acc"] = df_models[columns_to_average].mean(axis=1)
    df_models["avg_err"] = 1 - df_models["avg_acc"]
    df = df_models  

    df["run_name"] = (
        "Gemstone-" + df["width"].astype(str) + "x" + df["depth"].astype(str)
    )

    return df


def main():
    df = get_eval_df()
    loss_df = pd.read_json(
        "../wandb_dfs/wandb_df_for_fitting_hot_100b+_477.jsonl",
        orient="records",
        lines=True,
    )
    loss_df["run_name"] = loss_df["run_name"].str.replace(
        "PyGemma-", "Gemstone-", regex=False
    )
    loss_df["run_name"] = loss_df["run_name"].str.replace("_pretrain", "", regex=False)
    print(loss_df.columns)
    print(loss_df.head())

    df = df[["prec_token_count", "avg_err", "FLOPs", "run_name"]]

    loss_df = loss_df[["run_name", "tokens", "final_loss"]]
    merged_df = pd.merge(
        df,
        loss_df,
        how="inner",
        left_on=["run_name", "prec_token_count"],
        right_on=["run_name", "tokens"],
    )
    print(merged_df.head())
    print(len(df))
    print(len(merged_df))

    df = merged_df[["avg_err", "final_loss"]]

    x_data = df["final_loss"].values
    y_data = df["avg_err"].values

    ## https://github.com/mlfoundations/scaling/blob/a003c4913793ac2ae7ef87b28ecb562955d026d5/scaling/shared.py#L206
    def error_func(L, k, gamma, epsilon):
        return epsilon - k * np.exp(-gamma * L)

    popt, pcov = scipy.optimize.curve_fit(error_func, x_data, y_data, maxfev=10000)

    # Unpack fitted parameters
    k_fit, gamma_fit, epsilon_fit = popt

    # Generate smooth x values for plotting the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 200)
    y_fit = error_func(x_fit, *popt)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_data, y_data, label="Data", color="blue", alpha=0.6)
    plt.plot(x_fit, y_fit, label="Fitted Curve", color="red", linewidth=3)
    plt.xlabel("Loss")
    plt.ylabel("Average Top-1 Error")
    plt.gca().invert_xaxis()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    plt.grid(True)

    plt.savefig("gadre.pdf", bbox_inches="tight")

if __name__ == "__main__":
    main()

import pandas as pd
import os
from gadre_fit import get_eval_df
import numpy as np
import scipy
import matplotlib.pyplot as plt
import json

def collect_model_revision_losses(task, task_loss_dir):
    rows = []
    for model_name in os.listdir(task_loss_dir):
        model_path = os.path.join(task_loss_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        # model_revision_losses[model_name] = {}

        for revision in os.listdir(model_path):
            revision_path = os.path.join(model_path, revision)
            loss_file = os.path.join(revision_path, f"losses_{task}.json")

            if not os.path.isfile(loss_file):
                continue

            with open(loss_file, "r") as f:
                task_losses = json.load(f)

            # Flatten all task losses into one list
            all_losses = [loss for losses in task_losses.values() for loss in losses]
            if all_losses:
                avg_loss = np.mean(all_losses)
                # model_revision_losses[model_name][revision] = avg_loss
                rows.append(
                    {
                        "model_name": model_name,
                        "step": revision,
                        "val": avg_loss,
                        "task": task,
                    }
                )
            else:
                assert False, f"missing {model_name}, {revision}, {task}"

    return pd.DataFrame(rows)

def make_df():
    dfs = []
    for task in [
        "mmlu",
        "arce",
        "arcc",
        # "siqa",
        # "piqa",
        # "openbook",
        # "coqa",
        "hellaswag",
        # "winogrande",
    ]:
        this_df = collect_model_revision_losses(
            task, task_loss_dir="bhagia_task_losses"
        )
        dfs.append(this_df)

    df = pd.concat(dfs, ignore_index=True)
    df = df.groupby(["model_name", "step"], as_index=False)["val"].mean()

    def compute_tokens(row):
        if row["step"] == "main":
            step_val = 83475
        else:
            step_val = int(row["step"].split("_")[-1])
        return 2048 * 2048 * step_val

    df["tokens"] = df.apply(compute_tokens, axis=1)
    df = df.rename(columns={"val": "loss"})

    return df


def main():
    df = make_df()

    eval_df = get_eval_df()

    eval_df = eval_df[["prec_token_count", "avg_acc", "FLOPs", "run_name"]]
    df = df[["model_name", "tokens", "loss"]]
    merged_df = pd.merge(
        eval_df,
        df,
        how="inner",
        left_on=["run_name", "prec_token_count"],
        right_on=["model_name", "tokens"],
    )

    df = merged_df[["avg_acc", "loss"]]

    x_data = df["loss"].values
    y_data = df["avg_acc"].values

    def acc_func(L, a, k, L0, b):
        return a / (1 + np.exp(-k * (L - L0))) + b

    p0 = [0.5, 1.0, np.median(x_data), 0.1]
    a0 = y_data.max() - y_data.min()
    # Rough center L0 as the median of x
    L0_0 = np.median(x_data)
    # Baseline b as the minimum of y
    b0 = y_data.min()
    # Steepness k—start with something order-1
    k0 = 1.0

    p0 = [a0, k0, L0_0, b0]
    popt, pcov = scipy.optimize.curve_fit(acc_func, x_data, y_data, p0=p0, maxfev=10000)

    # Generate smooth x values for plotting the fitted curve
    x_fit = np.linspace(min(x_data), max(x_data), 200)
    y_fit = acc_func(x_fit, *popt)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.scatter(x_data, y_data, label="Data", color="blue", alpha=0.6)
    plt.plot(x_fit, y_fit, label="Fitted Curve", color="red", linewidth=3)
    plt.xlabel("Loss")
    plt.ylabel("Accuracy")
    plt.gca().invert_xaxis()
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    plt.grid(True)

    plt.savefig("bhagia.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()

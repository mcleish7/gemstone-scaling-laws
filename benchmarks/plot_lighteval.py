import os
import json
import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import font_manager
from itertools import cycle


def import_times_new_roman(this_font_manager, this_plt, font_size=16):
    this_font_manager.fontManager.addfont(
        f"../plotters/Times New Roman.ttf"
    )
    this_plt.rcParams["font.family"] = "Times New Roman"
    this_plt.rcParams["font.size"] = font_size


import_times_new_roman(font_manager, plt, font_size=16)

VOCAB_RESOLV = 50432
VOCAB_OURS = 50304
SEQ_LEN = 2048
WORLD_BATCH_SIZE = 2048.0
HEAD_SIZE = 128
EXPAND_FACTOR = 4.0


def flops_per_token_gqa(
    width,
    depth,
    vocab_size=VOCAB_OURS,
    queries_per_group=2,
    seq_len=SEQ_LEN,
    print_splits=False,
    return_mlp_attn_split=False,
):
    """
    Ashwinee gist: https://gist.github.com/kiddyboots216/32d8799ff307b2c3f32de854150a45c2
    David gist: https://gist.github.com/dymil/49f5d6721302f62687584ffccd1f4f15
    Estimates FLOPs per token for our setup. Carefully checked to be about as accurate as this construct can be.

    Some details (negligible even for extremely wide models) omitted, including:
    * numerically stable softmax
    * softmax addition only being over rows
    * dot products being only n-1 additions (fused multiply-add exists anyway)

    Note that this is 1.08-1.81x bigger than 7N (or 6N, if backward is only 2x), dependent on shape/size!
    TODO: check with https://github.com/sovrasov/flops-counter.pytorch ?"""
    num_qheads = width / HEAD_SIZE
    num_kvheads = (
        2 * num_qheads / queries_per_group
    )  # BOTH, so checks out for Gemma 2 2B, say

    embeddings = 0  # 2.0 * seq_len * vocab_size * width # 0 if sparse lookup, backward has FLOPs still but negligible

    attention = 2.0 * seq_len * (num_qheads + num_kvheads) * width * HEAD_SIZE
    attention += (
        3.5 * seq_len * (num_qheads + num_kvheads / 2) * HEAD_SIZE
    )  # RoPE, as implemented here/GPT-NeoX
    # score FLOPs are halved because causal => triangular mask => usable sparsity?
    kq_logits = 1.0 * seq_len * seq_len * HEAD_SIZE * num_qheads  # TODO double-check
    softmax = 3.0 * seq_len * seq_len * num_qheads
    softmax_q_red = 2.0 * seq_len * seq_len * HEAD_SIZE * num_qheads
    final_linear = 2.0 * seq_len * width * HEAD_SIZE * num_qheads
    attn_bwd = (
        2.0 * attention
        + 2.5 * (kq_logits + softmax + softmax_q_red)
        + 2.0 * final_linear
    ) * depth
    attention += kq_logits + softmax + softmax_q_red + final_linear

    ffw_size = EXPAND_FACTOR * width
    dense_block = (
        6.0 * seq_len * width * ffw_size
    )  # three matmuls instead of usual two b/c GEGLU
    # TODO: GeLU on half, but tanh part of constant is https://stackoverflow.com/a/66993354
    dense_block += (
        10 * seq_len * ffw_size
    )  # 7 for other ops: 3 for cubic, two additions, two scalar mults
    dense_block += 2.0 * width * seq_len  # both/sandwich residual additions
    rmsnorm = 2 * 7.0 * width * seq_len  # TODO: check

    if return_mlp_attn_split:
        return {
            "attn_fwd": attention / seq_len,
            "attn_bwd": attn_bwd / seq_len,
            "mlp_fwd": (ffw_size + dense_block + rmsnorm) / seq_len,
            "mlp_bwd": (depth * (dense_block + rmsnorm)) / seq_len,
        }

    final_rms_norm = 7.0 * width * seq_len  # one last RMSNorm
    final_logits = 2.0 * seq_len * width * vocab_size
    nonattn_bwd = 2.0 * (
        embeddings + depth * (dense_block + rmsnorm) + final_rms_norm + final_logits
    )
    forward_pass = (
        embeddings
        + depth * (attention + dense_block + rmsnorm)
        + final_rms_norm
        + final_logits
    )
    backward_pass = attn_bwd + nonattn_bwd  # flash attention
    if print_splits:
        print(
            np.array(
                [
                    width,
                    depth,  # kq_logits/attention, softmax/attention, softmax_q_red/attention, final_linear/attention,
                    attention * depth,
                    attention * depth / forward_pass,
                ]
            ).T
        )
    return (forward_pass + backward_pass) / seq_len


def param_counter(width, depth, vocab_size=VOCAB_OURS):
    # hardcoded decisions from plot_feasible
    head_size = 128
    n_head = int(width / 128)
    n_query_groups = int(n_head / 2)
    intermediate_size = 4 * width

    # Embedding layer parameters
    embedding_params = vocab_size * width

    # Attention parameters: attn + proj
    attn_shape = (n_head + 2 * n_query_groups) * head_size
    attn_params = (width * attn_shape) + (head_size * n_head * width)

    # MLP parameters: fc_1 + fc_2 + proj
    mlp_params = (
        (width * intermediate_size)
        + (width * intermediate_size)
        + (intermediate_size * width)
    )

    # RMSNorm parameters: 2 per block
    norm_params_per_block = 2 * width

    # Total per block
    total_block_params = attn_params + mlp_params + norm_params_per_block

    # All layers (blocks)
    total_params = total_block_params * depth

    # Final RMSNorm and LM Head
    final_norm_params = width
    lm_head_params = width * vocab_size

    # Total model parameters
    return total_params + embedding_params + final_norm_params + lm_head_params


# Function to extract token count and model label from model name.
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


if __name__ == "__main__":
    df = pd.read_parquet("light_eval_df.parquet")
    df_err = pd.read_parquet("light_eval_df_err.parquet")

    # Create new DataFrame columns based on model names
    df_models = df.copy()
    # Map each model (index) to its token count and model label.
    df_models[["token_count", "prec_token_count", "model_label", "width", "depth"]] = (
        pd.DataFrame(
            [extract_model_info(m) for m in df_models.index], index=df_models.index
        )
    )
    df_err[["token_count", "prec_token_count", "model_label", "width", "depth"]] = (
        pd.DataFrame([extract_model_info(m) for m in df_err.index], index=df_err.index)
    )
    df_models["params"] = df_models.apply(
        lambda row: param_counter(row["width"], row["depth"]), axis=1
    )
    df_models["FLOPs_per_token"] = df_models.apply(
        lambda row: flops_per_token_gqa(row["width"], row["depth"]), axis=1
    )
    df_models["FLOPs"] = df_models["FLOPs_per_token"] * df_models["prec_token_count"]

    # df_models = df_models.sort_values(by="params")
    sorted_model_labels = (
        df_models.sort_values(by="params")["model_label"].unique().tolist()
    )


    # Identify benchmark tasks (exclude the extra columns)
    benchmark_tasks = [col for col in df.columns]

    random_chance = {
        "arc:challenge": 0.25,
        "arc:easy": 0.25,
        "commonsense_qa": 0.2,
        "hellaswag": 0.25,
        "mmlu": 0.25,
        "openbookqa": 0.25,
        "piqa": 0.5,
        "siqa": 0.33,
        "winogrande": 0.5,
    }

    n_tasks = len(benchmark_tasks)
    ncols = 3  # adjust as desired
    nrows = math.ceil(n_tasks / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, task in enumerate(benchmark_tasks):
        ax = axes[idx]
        # Define a list of markers to cycle through.
        markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x", "*"]
        # Iterate over each model shape.
        for i, model in enumerate(sorted_model_labels):
            marker = markers[i % len(markers)]
            # Select data for the current model and sort by FLOPs.
            model_data = df_models[df_models["model_label"] == model].sort_values("FLOPs")
            model_err = df_err.loc[model_data.index, task]
            ax.errorbar(
                model_data["FLOPs"],
                model_data[task],
                yerr=model_err,
                label=model,
                marker=marker,
                linestyle="--" if i < 11 else "-",
                markersize=8,
            )
        # Plot the horizontal random-chance line for this benchmark.
        ax.axhline(y=random_chance[task], color="black", label="Random", linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("FLOPs")
        ax.set_ylabel("acc_norm")
        ax.set_title(f"{task}")
        ax.grid(True)
        # ax.legend(title="Model Shape", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Remove any extra subplots if there are more axes than tasks.
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Model Shape", bbox_to_anchor=(0.9, 0.3), loc="lower left"
    )
    plt.subplots_adjust(hspace=0.35)

    # plt.tight_layout()
    # Save the figure with a filename that includes the task name.
    plt.savefig(
        f"lighteval_line_plot.pdf",
        bbox_inches="tight",
    )
    plt.close()


    def compute_avg_stderr(row):
        # Filter out missing stderr values.
        valid = row.dropna()
        n = len(valid)
        if n == 0:
            return 0
        # Compute combined stderr: sqrt(sum(error^2)) / n
        return np.sqrt(np.sum(np.square(valid))) / n
        # return np.sqrt(np.sum(np.square(valid)) / n**2)  # =>Ashwinee
        # Ashwinee: (sum(x*x for x in stderr) / len(stderr)**2)**0.5


    df_models["avg_stderr"] = df_err[benchmark_tasks].apply(compute_avg_stderr, axis=1)
    df_models["avg_benchmark"] = df_models[benchmark_tasks].mean(axis=1)

    # Create a line plot of the average benchmark score versus token count for each model.
    import_times_new_roman(font_manager, plt, font_size=20)
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o", "s", "^", "D", "v", ">", "<", "p", "h", "x", "*"]

    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    for i, model in enumerate(sorted_model_labels):
        color = next(color_cycle)
        marker = markers[i % len(markers)]
        model_data = df_models[df_models["model_label"] == model].sort_values("FLOPs")

        # Plot error bars without adding to the legend
        ax.errorbar(
            model_data["FLOPs"],
            model_data["avg_benchmark"],
            yerr=model_data["avg_stderr"],
            label="_nolegend_",
            marker=marker,
            linestyle="--",
            linewidth=2.5,
            markersize=12 if marker == "*" else 8,
            markeredgewidth=2 if marker == "x" else None,
            color=color,
        )

        # Add dummy plot for legend only (without error bars)
        ax.plot(
            model_data["FLOPs"],
            model_data["avg_benchmark"],
            label=model,
            marker=marker,
            linestyle="--",
            linewidth=2.5,
            markersize=12 if marker == "*" else 8,
            markeredgewidth=2 if marker == "x" else None,
            color=color,
        )

    ax.set_xscale("log")
    ax.set_xlabel("FLOPs")
    ax.set_ylabel("Accuracy (Normalized)")
    ax.legend(title="Model Shape", bbox_to_anchor=(1.05, 1), loc="upper left", ncol=3)
    ax.grid(True)

    plt.savefig(
        f"lighteval_average_benchmark_line_plot.pdf",
        bbox_inches="tight",
    )
    plt.close()

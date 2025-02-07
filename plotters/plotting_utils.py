import enum
import json
import numpy as np
from numpy.typing import NDArray
from numpy import number
import pandas as pd

Y_Axis = enum.Enum("Y_Axis", ["DATA", "PARAMS", "TOKENS_PER_PARAM"])

# Give consistent styles everywhere
try:
    with open("plotters/model_styles.json", "r") as file:
        model_styles = json.load(file)
except:
    with open("model_styles.json", "r") as file:
        model_styles = json.load(file)
# example of how to apply in a map
# df["color"] = df["run_name"].map(lambda x: model_styles[x]["color"])
# df["line_style"] = df["run_name"].map(lambda x: model_styles[x]["line_style"])


def import_times_new_roman(this_font_manager, this_plt, font_size=16):
    try:
        this_font_manager.fontManager.addfont(f"plotters/Times New Roman.ttf")
    except:
        this_font_manager.fontManager.addfont(f"Times New Roman.ttf")
    this_plt.rcParams["font.family"] = "Times New Roman"
    this_plt.rcParams["font.size"] = font_size


def our_pred_func(data, exps, coefs, e):
    """Our prediction function"""
    return np.sum(np.exp(coefs - (exps * np.log(data))), axis=-1) + np.exp(e)


def add_a_line_alpha_beta(
    ax, xs, A, alpha, a, B, beta, b, label, y_axis, color_arg=None, line_style_arg=None
):
    color, line_style = color_arg, line_style_arg
    if label in model_styles:
        color = model_styles[label]["color"]
        line_style = model_styles[label]["line_style"]
    if label == "Chinchilla-epochai-reported":
        label = "Chinchilla Law"

    # apply equation 4 from chinchilla paper
    G = ((alpha * A) / (beta * B)) ** (1 / (alpha + beta))
    if y_axis == Y_Axis.DATA:
        ax.plot(
            xs,
            (G ** (-1) * (xs / 6) ** b),
            label=label,
            color=color,
            linestyle=line_style,
        )  # uncomment to see data scaling only
    elif y_axis == Y_Axis.PARAMS:
        ax.plot(
            xs,
            (G * ((xs / 6) ** a)),
            label=label,
            color=color,
            linestyle=line_style,
        )  # uncomment to see param scaling only
    elif y_axis == Y_Axis.TOKENS_PER_PARAM:
        ax.plot(
            xs,
            (G ** (-1) * (xs / 6) ** b) / (G * (xs / 6) ** a),
            label=label,
            color=color,
            linestyle=line_style,
        )


def add_a_line(
    ax,
    xs,
    params_coeff,
    tokens_coeff,
    params_exp,
    tokens_exp,
    label,
    y_axis,
    color_arg=None,
    line_style_arg=None,
):
    A = params_coeff
    alpha = params_exp
    B = tokens_coeff
    beta = tokens_exp
    a = beta / (alpha + beta)
    b = 1 - a
    add_a_line_alpha_beta(
        ax, xs, A, alpha, a, B, beta, b, label, y_axis, color_arg, line_style_arg
    )


def add_model_lines(models, ax, xs, y_axis):
    """
    models = list of model dicts
        dict keys:
            approach: which apporach 1 or 3
    ax = axis to plot on
    xs = x's to use to plot the line
    """
    for model in models:
        if model["approach"] == 1:
            """
            Dict keys: coef, intercept, label
            """
            ys = model["coef"] * xs + model["intercept"]
            ax.plot(np.exp(xs), np.exp(ys), label=model["label"])

        elif model["approach"] == 3:
            if "params_coeff" in model:
                """
                Dict keys:
                    params_coeff
                    tokens_coeff
                    params_exp
                    tokens_exp
                    label
                """
                add_a_line(
                    ax,
                    xs,
                    np.exp(model["params_coeff"]),
                    np.exp(model["tokens_coeff"]),
                    model["params_exp"],
                    model["tokens_exp"],
                    model["label"],
                    y_axis,
                )
            elif "coefficients" in model:
                """
                2nd option so we can hand our models straight in
                Dict keys:
                    coefficients
                    exponents
                    label
                """
                add_a_line(
                    ax,
                    xs,
                    params_coeff=np.exp(model["coefficients"][0]),
                    tokens_coeff=np.exp(model["coefficients"][1]),
                    params_exp=model["exponents"][0],
                    tokens_exp=model["exponents"][1],
                    label=model["label"],
                    y_axis=y_axis,
                )
            else:
                print("Could not match approach 3 dict")
                exit()
        else:
            print("Can only be approach 1 or 2")
            exit()


# tuple: (As, Bs, alphas, betas)
other_peoples_constants = {
    "chinchilla-epochai-reported": (
        np.log(482.00572),
        np.log(2085.43420),
        0.3478,
        0.3658,
    ),  # from here: https://arxiv.org/pdf/2404.10102
    "chinchilla-fit-epoch": (
        np.log(477.84171252965143),
        np.log(2143.8637880335505),
        0.34731265761033453,
        0.3671826173946711,
    ),  # From here: https://github.com/epoch-research/analyzing-chinchilla/blob/main/data_analysis.ipynb, using the og way of fitting
}


def plot_others(names, ax, xs, y_axis):
    to_plot = []
    to_plot_kaplan = []
    for name in names:
        if name.lower() not in other_peoples_constants:
            print(f"{name} not available")
            continue
        A, B, alpha, beta = other_peoples_constants[name.lower()]
        this_dict = {
            "approach": 3,
            "params_coeff": A,
            "tokens_coeff": B,
            "params_exp": alpha,
            "tokens_exp": beta,
            "label": name,
        }
        if "kaplan" not in name.lower():
            to_plot.append(this_dict)
        else:
            to_plot_kaplan.append(this_dict)
    if to_plot_kaplan:
        # https://github.com/formll/resolving-scaling-law-discrepancies/blob/b20ccd4300de08eeafcd39edef63540a71d2fcd4/plotting.py#L84
        if y_axis == Y_Axis.PARAMS:
            ax.plot(
                xs,
                1.6e9 * (xs / (1e15 * 24 * 60 * 60)) ** 0.88,
                ":",
                color="gray",
                label="Kaplan Law",
            )
        elif y_axis == Y_Axis.DATA:
            ax.plot(
                xs,
                xs / (1.6e9 * (xs / (1e15 * 24 * 60 * 60)) ** 0.88) / 6,
                ":",
                color="gray",
                label="Kaplan Law",
            )
        elif y_axis == Y_Axis.TOKENS_PER_PARAM:
            ax.plot(
                xs,
                xs / (1.6e9 * (xs / (1e15 * 24 * 60 * 60)) ** 0.88) ** 2 / 6,
                ":",
                color="gray",
                label="Kaplan Law",
            )
        else:
            print("kaplan x axis not made")
            exit()
    add_model_lines(to_plot, ax, xs, y_axis)


industrial_tokens = {
    "google/Gemma-2b": 3e12,
    "google/Gemma-7b": 6e12,
    "google/Gemma-2-2b": 2e12,  # https://arxiv.org/pdf/2408.00118
    "google/Gemma-2-9b": 8e12,
    "google/Gemma-2-27b": 13e12,
    "meta-llama/Llama-2-7b-hf": 2e12,
    "meta-llama/Llama-2-13b-hf": 2e12,
    "meta-llama/Llama-2-70b-hf": 2e12,
    "meta-llama/Llama-3-8B": 15e12,
    "meta-llama/Llama-3.1-70B": 15e12,
    "meta-llama/Llama-3.2-1B": 9e12,  # https://huggingface.co/blog/llama32
    "meta-llama/Llama-3.2-3B": 9e12,
    "Qwen/Qwen2-0.5B": 12e12,
    "Qwen/Qwen2-1.5B": 7e12,
    "Qwen/Qwen2-7B": 7e12,
}


def assign_marker(plot_name):
    if "Llama" in plot_name:
        return "x"
    elif "Gemma" in plot_name:
        return "o"
    elif "Qwen2.5" in plot_name:
        return "+"
    elif "Qwen" in plot_name:
        return "^"
    else:
        return "."


def plot_industrial(ax, y_axis, plt, no_qwen=False, s_param=None):
    try:
        with open(f"../data/our_val_loss_data.json", "r") as json_file:
            data = json.load(json_file)
    except:
        with open(f"data/our_val_loss_data.json", "r") as json_file:
            data = json.load(json_file)

    df = pd.DataFrame(data)

    df = df[~df["hf_name"].str.contains("Qwen2.5")]  # no token counts here
    if no_qwen:
        df = df[~df["hf_name"].str.contains("Qwen")]

    df["tokens"] = df["hf_name"].map(industrial_tokens)
    df["FLOPs_6N"] = 6 * df["tokens"] * df["params"]  # 6ND
    df["plot_name"] = df["hf_name"].apply(lambda x: x.split("/")[-1])
    df["wd_ratio"] = df["width"] / df["depth"]
    df["marker"] = df["plot_name"].apply(assign_marker)

    if y_axis == Y_Axis.DATA:
        y_axis = "tokens"
    elif y_axis == Y_Axis.PARAMS:
        y_axis = "params"
    elif y_axis in ["loss", "wd_ratio"]:
        y_axis = y_axis
    elif y_axis == "tokens/param":
        df["tokens/param"] = df["tokens"] / df["params"]
        y_axis = y_axis
    else:
        print(f"axis ({y_axis}) not found for industry models")
        exit()

    unique_names = df["plot_name"].unique()
    num_colors = len(unique_names)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))
    s = 40 if s_param is None else s_param
    for color, name in zip(colors, unique_names):
        subset = df[df["plot_name"] == name]
        ax.scatter(
            subset["FLOPs_6N"],
            subset[y_axis],
            label=name,
            color=color,
            alpha=1.0,
            s=s,
            marker=subset["marker"].iloc[0],
        )


VOCAB_RESOLV = 50432
VOCAB_OURS = 50304
SEQ_LEN = 2048
WORLD_BATCH_SIZE = 2048.0
HEAD_SIZE = 128
EXPAND_FACTOR = 4.0


def flops_per_token_gqa(
    width: NDArray[number] | number,
    depth: NDArray[number] | number,
    vocab_size=VOCAB_OURS,
    queries_per_group=2,
    seq_len=SEQ_LEN,
    print_splits=False,
    return_mlp_attn_split=False,
):
    """
    Estimates FLOPs per token for our setup. Carefully checked to be about as accurate as this construct can be.

    Some details (negligible even for extremely wide models) omitted, including:
    * numerically stable softmax
    * softmax addition only being over rows
    * dot products being only n-1 additions (fused multiply-add exists anyway)
    """
    num_qheads = width / HEAD_SIZE
    num_kvheads = (
        2 * num_qheads / queries_per_group
    )  # BOTH, so checks out for Gemma 2 2B

    embeddings = 0  # 2.0 * seq_len * vocab_size * width # 0 if sparse lookup, backward has FLOPs still but negligible

    attention = 2.0 * seq_len * (num_qheads + num_kvheads) * width * HEAD_SIZE
    attention += (
        3.5 * seq_len * (num_qheads + num_kvheads / 2) * HEAD_SIZE
    )  # RoPE, as implemented here/GPT-NeoX
    # score FLOPs are halved because causal => triangular mask => usable sparsity?
    kq_logits = 1.0 * seq_len * seq_len * HEAD_SIZE * num_qheads
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
    dense_block += (
        10 * seq_len * ffw_size
    )  # 7 for other ops: 3 for cubic, two additions, two scalar mults
    dense_block += 2.0 * width * seq_len  # both/sandwich residual additions
    rmsnorm = 2 * 7.0 * width * seq_len

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


def param_counter_relaxed(width, depth, vocab_size=VOCAB_OURS):
    # hardcoded decisions from plot_feasible, relaxed to be differentiable
    head_size = 128
    n_head = width / 128
    n_query_groups = n_head / 2
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


def add_flops_to_df(df):
    df["FLOPs_per_token"] = df.apply(
        lambda row: flops_per_token_gqa(row["width"], row["depth"]), axis=1
    )
    df["FLOPs"] = df["FLOPs_per_token"] * df["tokens"]
    return df

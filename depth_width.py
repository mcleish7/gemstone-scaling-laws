import time
import os
import argparse
from itertools import product
import concurrent.futures
import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy import number
import pandas as pd
from scipy.optimize import minimize, OptimizeResult
from scipy.special import logsumexp, huber
from tqdm import tqdm
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import logsumexp


def log_pred(
    exps: NDArray[number], coefs: NDArray[number], e: ArrayLike, data: NDArray[number]
):
    """Predict the log-loss from the power law given the parameter values for exponents and coefficients, and irreducible error e
    data: 1 or 2D array"""
    # predictions = np.logaddexp(a - alpha * np.log(F), e) original line from resolving code
    # our OG version: np.log(np.sum(np.exp(coefs - (exps * np.log(data))), axis=-1) + np.exp(e))
    arr_1 = coefs - (exps * np.log(data))
    arr_2 = np.full((arr_1.shape[0], 1), e)
    cat = np.concatenate((arr_1, arr_2), axis=1)
    return logsumexp(cat, axis=-1)


def huber_loss_objective(
    params: NDArray[number],
    data: NDArray[number],
    losses: NDArray[number],
    num_parameters: int,
    delta: float,
    weights: ArrayLike = 1,
):
    """
    https://github.com/formll/resolving-scaling-law-discrepancies/blob/main/analysis.py#L19
    """
    exps = params[:num_parameters]  # First three are exponents
    coefs = params[num_parameters : (2 * num_parameters)]  # Next three are coefficients
    e = params[2 * num_parameters]  # Last one is irreducible error

    predictions = log_pred(exps, coefs, e, data)

    return np.sum(
        huber(delta, np.log(losses) - predictions) * weights
    )  # log as we predict log loss


def optimize_params(
    initial_params: ArrayLike,
    data: NDArray[number],
    losses: NDArray[np.float64],
    num_parameters: int,
    delta: float,
):
    result = minimize(
        huber_loss_objective,
        initial_params,
        args=(
            data,
            losses,
            num_parameters,
            delta,
        ),
        method="L-BFGS-B",
    )

    return result


def grid_search(
    show_df,
    depth_width,
    depth_width_parameters,
    num_processes,
    delta,
):
    assert not (
        depth_width and depth_width_parameters
    ), "can't have depth_width and depth_width_parameters true at the same time"
    if depth_width:
        num_parameters = 3
        data = show_df[["width", "depth", "tokens"]].to_numpy()
    elif depth_width_parameters:
        num_parameters = 4
        data = show_df[["width", "depth", "params_active_precise", "tokens"]].to_numpy()
    else:
        data = show_df[["params_active_precise", "tokens"]].to_numpy()
        num_parameters = 2
    losses = show_df["final_loss"].to_numpy()

    param_search_array = []
    # https://github.com/epoch-research/analyzing-chinchilla/blob/main/data_analysis.ipynb?short_path=e1185cd#L886
    for _ in range(num_parameters):
        param_search_array.append(np.arange(0, 2.5, 0.5))  # exp, 5 possible values
    for _ in range(num_parameters):
        param_search_array.append(np.arange(0, 30, 5))  # coefficient, 6 values
    param_search_array.append(np.arange(-1, 1.5, 0.5))  # error, 5 values
    # 135,000 permutations for 3 terms

    best_loss = np.inf
    best_result = OptimizeResult()

    global_start_time = time.time()
    num_workers = num_processes if num_processes else os.cpu_count() // 2
    count = 0

    print(f"Number of workers: {num_workers}")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Map each set of parameters to the optimize_params function, along with the necessary variables
        futures = {
            executor.submit(
                optimize_params,
                params,
                data,
                losses,
                num_parameters,
                delta,
            ): params
            for params in product(*param_search_array)
        }
        print(f"Total futures submitted: {len(futures)}")

        # As results become available, check if they are better
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Running optimize_params tasks...",
        ):

            result = future.result(timeout=60)
            count += 1
            if count % 10 == 0:
                global_total_time = time.time() - global_start_time
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_result = result
                print(f"New best loss: {best_loss}", flush=True)

        global_total_time = time.time() - global_start_time
        print(f"All tasks completed in {global_total_time} seconds.")

        print(f"Best params: {best_result.x}")
        return dict(
            success=best_result.success,
            loss=best_loss,
            exponents=best_result.x[:num_parameters],
            coefficients=best_result.x[num_parameters : (num_parameters * 2)],
            irreducible_error=best_result.x[num_parameters * 2],
        )


def perform_main_analysis_ours(
    show_df,
    num_processes,
    delta,
    depth_width=True,
    depth_width_parameters=False,
):
    out = [
        grid_search(
            show_df,
            depth_width,
            depth_width_parameters,
            num_processes,
            delta,
        )
    ]
    return pd.DataFrame(out)


def main(args):
    assert args.num_parameters in [2, 3, 4], "num params mismatch"
    depth_width = args.num_parameters == 3
    depth_width_parameters = args.num_parameters == 4

    if (
        args.cool_begin
        or args.cool_end
        or args.hot
        or args.lr_ablation_hot
        or args.hot_over_100
        or args.hot_over_120
        or args.hot_over_100_512x
        or args.like_chinchilla
        or args.like_chinchilla_lr_ablation
        or args.lr_ablation_slim_chinchilla
        or args.slim_chinchilla
    ):
        if args.cool_begin:
            postfix = "cool_begin"
        if args.cool_end:
            postfix = "cool_end"
        elif args.hot:
            postfix = "hot"
        elif args.lr_ablation_hot:
            postfix = "lr_ablation_hot"
        elif args.hot_over_100:
            postfix = "hot_100b+"
        elif args.hot_over_100_512x:
            postfix = "hot_100b+_512x"
        elif args.hot_over_120:
            postfix = "hot_120b+_only"
        elif args.like_chinchilla:
            postfix = "hot_sampled_like_chinchilla"
        elif args.like_chinchilla_lr_ablation:
            postfix = "lr_ablation_hot_sampled_like_chinchilla"
        elif args.lr_ablation_slim_chinchilla:
            postfix = "lr_ablation_slim_chinchilla"
        elif args.slim_chinchilla:
            postfix = "slim_chinchilla"

        df = pd.read_json(
            f"wandb_dfs/wandb_df_for_fitting_{postfix}.jsonl",
            orient="records",
            lines=True,
        )

        if args.no_embeds:
            df["params_active_precise"] = df["params_active_precise"] - (
                50304 * df["width"]
            )  # 50304 = our vocab size

        summary_df = perform_main_analysis_ours(
            df,
            args.num_processes,
            args.delta,
            depth_width,
            depth_width_parameters,
        )

    else:
        print("FLAG not found to optimize")
        exit()

    print(summary_df)

    num_params_str = "_width_depth_"
    if args.num_parameters == 2:
        num_params_str = "_parameters_"
    elif args.num_parameters == 4:
        num_params_str = "_width_depth_parameters_"

    postfix = ""
    if args.cool_begin:
        postfix = "_cool_begin"
    if args.cool_end:
        postfix = "_cool_end"
    elif args.hot:
        postfix = "_hot"
    elif args.lr_ablation_hot:
        postfix = "_lr_ablation_hot"
    elif args.hot_over_100:
        postfix = "_hot_100b+"
    elif args.hot_over_100_512x:
        postfix = "_hot_100b+_512x"
    elif args.hot_over_120:
        postfix = "_hot_120b+_only"
    elif args.like_chinchilla:
        postfix = "_hot_sampled_like_chinchilla"
    elif args.like_chinchilla_lr_ablation:
        postfix = "_lr_ablation_hot_sampled_like_chinchilla"
    elif args.lr_ablation_slim_chinchilla:
        postfix = "_lr_ablation_slim_chinchilla"
    elif args.slim_chinchilla:
        postfix = "_slim_chinchilla"

    postfix += "_no_embeds" if args.no_embeds else ""

    save_path = "parameters" if args.save_path is None else args.save_path
    os.makedirs(save_path, exist_ok=True)
    summary_df.to_json(
        f"{save_path}/approach_3{num_params_str}grid_search_0{postfix}.json"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit a scaling law")
    parser.add_argument(
        "--num_parameters",
        type=int,
        default=3,
        help="Number of parameters to process (default: 3)",
    )
    parser.add_argument("--delta", type=float, default=1e-4, help="Delta in Huber Loss")
    parser.add_argument("--num_processes", type=int, default=None)
    parser.add_argument("--save_path", type=str, default=None)

    # look through wandb_extraction to see what there are
    parser.add_argument("--cool_begin", action="store_true")
    parser.add_argument("--cool_end", action="store_true")
    parser.add_argument("--hot", action="store_true")
    parser.add_argument("--hot_over_100", action="store_true")
    parser.add_argument("--hot_over_100_512x", action="store_true")
    parser.add_argument("--hot_over_120", action="store_true")
    parser.add_argument("--lr_ablation_hot", action="store_true")

    parser.add_argument("--like_chinchilla", action="store_true")
    parser.add_argument("--like_chinchilla_lr_ablation", action="store_true")
    parser.add_argument("--slim_chinchilla", action="store_true")
    parser.add_argument("--lr_ablation_slim_chinchilla", action="store_true")

    parser.add_argument("--no_embeds", action="store_true")

    args = parser.parse_args()
    main(args)

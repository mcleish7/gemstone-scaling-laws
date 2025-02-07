from plotting_utils import flops_per_token_gqa, param_counter_relaxed, param_counter
from scipy.optimize import minimize

counter = 0
total = 0


def optimize_row(min_row, wd_law, relaxed):
    if relaxed:
        param_counter_to_use = param_counter_relaxed
    else:
        param_counter_to_use = param_counter

    global counter, total
    min_row = min_row.copy()

    law = wd_law
    # min law(w,d,p,t)
    # s.t. tokens*flops_per_token_gqa(w,d) = FLOPs

    def objective(params):
        w, d, t = params
        return law([w, d, param_counter_to_use(w, d), t])  # Adapt to your inputs

    def constraint(params):
        w, d, t = params
        return t * flops_per_token_gqa(w, d) - min_row["FLOPs"]

    bounds = [(256, None), (2, None), (1, None)]
    width, depth, tokens = min_row["width"], min_row["depth"], min_row["tokens"]
    best_result = None
    for poss_width in [int(width * (2**i)) for i in range(-3, 4)]:
        for poss_depth in [int(depth * (2**i)) for i in range(-3, 4)]:
            for poss_tokens in [int(tokens * (2**i)) for i in range(-3, 4)]:
                result = minimize(
                    objective,
                    [poss_width, poss_depth, poss_tokens],
                    constraints={"type": "eq", "fun": constraint},
                    bounds=bounds,
                    options={"maxiter": 500},
                    method="SLSQP",
                )
                if best_result is None:
                    best_result = result
                elif result.fun < best_result.fun:
                    best_result = result

    total += 1
    if best_result.fun < min_row["wd_pred_loss"]:
        w, d, t = best_result.x
        min_row["params"] = param_counter_to_use(w, d)
        min_row["tokens"] = t
        min_row["width"] = w
        min_row["depth"] = d
        min_row["wd_ratio"] = w / d
        min_row["wd_pred_loss"] = best_result.fun
        min_row["FLOPs"] = t * flops_per_token_gqa(w, d)
        counter += 1
        print(f"{counter}/{total}")

    return min_row

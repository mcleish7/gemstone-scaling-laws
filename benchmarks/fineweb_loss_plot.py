from datasets import load_dataset, load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import glob
from natsort import natsorted
from tqdm import tqdm
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from plot_lighteval import flops_per_token_gqa, param_counter
import zstandard as zstd
import numpy as np
import matplotlib.lines as mlines
from matplotlib import font_manager


def import_times_new_roman(this_font_manager, this_plt, font_size=16):
    try:
        this_font_manager.fontManager.addfont(f"../plotters/Times New Roman.ttf")
    except:
        this_font_manager.fontManager.addfont(f"Times New Roman.ttf")
    this_plt.rcParams["font.family"] = "Times New Roman"
    this_plt.rcParams["font.size"] = font_size


import_times_new_roman(font_manager, plt, font_size=32)

def preprocess_dclm():
    file_path = (
        "/fs/cml-projects/math-dataset/dclm_jsonls/shard_00000000_processed.jsonl.zst"
    )

    to_save = []
    count = 0
    with open(file_path, "rb") as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            # Use a buffered reader to handle text decoding
            import io

            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            for line in text_stream:
                data = json.loads(line)
                count += 1
                to_save.append({"data": data["text"]})

    dataset = Dataset.from_list(to_save)
    dataset.save_to_disk("/fs/cml-projects/math-dataset/dclm")


def preprocess(output_dir, chunk_size=2049, max_tokens=10_000_000, dclm=False):
    # Take only first 10M tokens
    if dclm:
        fw = load_from_disk("/fs/cml-projects/math-dataset/dclm")
        data_key = "data"
    else:
        fw = load_from_disk("/fs/cml-projects/math-dataset/fineweb_10b")
        data_key = "text"
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/Gemstone-768x45")
    # tokenizer.add_bos_token = True

    chunk_index = 0
    total_tokens = 0
    for example in fw:
        text = example[data_key]  # Ensure correct field name
        tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ][0]

        if len(tokens) >= chunk_size:
            tokens = tokens[:chunk_size]
            torch.save(tokens, f"{output_dir}/chunk_{chunk_index:06d}.pt")

            chunk_index += 1
            total_tokens += chunk_size

            if total_tokens >= max_tokens:
                break


@torch.no_grad()
def run():
    parser = argparse.ArgumentParser(description="Script to load a model by name.")
    parser.add_argument(
        "--model_name", type=str, help="Name of the model to load."
    )
    parser.add_argument("--edu", action="store_true", help="Enable verbose output")
    parser.add_argument("--dclm", action="store_true", help="Enable verbose output")
    parser.add_argument("--plot", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--process_outputs", action="store_true", help="Enable verbose output"
    )
    args = parser.parse_args()
    model_name = args.model_name
    if args.plot:
        plot()
        exit()
    if args.process_outputs:
        process_outputs()
        exit()

    json_path = f"evals/fineweb{'_edu' if args.edu else ''}{'_dclm' if args.dclm else ''}_losses/{model_name.split('/')[1]}.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            loss_store = json.load(f)
    else:
        loss_store = {}

    for revision in [f"step_{str(477*step).zfill(8)}" for step in range(5, 175, 5)]:
        if revision in loss_store:
            print(f"Skipping revision {revision}, already processed.")
            continue

        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision).to(
            "cuda"
        )
        model.eval()

        running_loss = 0.0
        counter = 0
        try:
            pt_files = glob.glob(
                f"fineweb{'_edu' if args.edu else ''}{'_dclm' if args.dclm else ''}_10m_tokens_gemstone_tokenized/*.pt"
            )
        except:
            raise FileNotFoundError("Need to tokenize 10M of fineweb/fineweb_edu/dclm to run this. Look at the `preprocess functions in fineweb_loss_plot for how to do this.")

        pt_files = natsorted(pt_files)
        for idx, file in enumerate(tqdm(pt_files)):
            row = torch.load(file).to(model.device).unsqueeze(0)
            loss = model(row, labels=row.clone()).loss

            running_loss += loss.item()
            counter += 1

        final_loss = running_loss / counter

        loss_store[revision] = final_loss

        with open(json_path, "w") as f:
            json.dump(loss_store, f)


def round_sig(x, sig=1):
    return round(x, -int(np.floor(np.log10(abs(x)))) + (sig - 1)) if x != 0 else 0


def get_our_data():
    df = pd.read_json(
        f"../wandb_dfs/wandb_df_for_fitting_hot_100b+.jsonl",
        orient="records",
        lines=True,
    )
    steps = [11925 * i for i in range(1, 8)]
    steps = [47700, 59625, 71550, 83475]
    df = df[df["step"].isin(steps)]

    df = df.rename(
        columns={
            "params_active_precise": "params",
            "run_name": "model",
            "final_loss": "loss",
            "step": "revision",
        }
    )
    df["model"] = df["model"].str.replace("PyGemma", "Gemstone")
    df["model"] = df["model"].str.replace("_pretrain", "")

    df["params_round"] = df["params"].apply(round_sig)
    df.loc[df["params_round"] == 90000000, "params_round"] = 100000000
    df.loc[df["params_round"] == 70000000, "params_round"] = 100000000

    df["flops_per_token"] = df.apply(
        lambda row: flops_per_token_gqa(row["width"], row["depth"]), axis=1
    )
    df["total_flops"] = df["flops_per_token"] * df["tokens"]
    df["GPU_Hours"] = (
        df["revision"] * df["seconds_per_step"] * df["num_nodes"] * 8
    ) / (60 * 60)

    return df

def plot():
    dataset_labels = {
        (False, True): "DCLM",
        (False, False): "FineWeb",
        (True, False): "FineWeb-Edu",
    }

    all_dfs = []

    for edu, dclm in dataset_labels:
        if dclm:
            json_files = glob.glob(
                os.path.join(
                    f"../other_validation_losses/dclm_losses",
                    "*.json",
                )
            )
        else:
            json_files = glob.glob(
                os.path.join(
                    f"../other_validation_losses/fineweb{'_edu' if edu else ''}{'_dclm' if dclm else ''}_losses",
                    "*.json",
                )
            )

        all_data = []
        for file_path in json_files:
            with open(file_path, "r") as f:
                data = json.load(f)
                model_name = file_path.split("/")[-1].split(".")[0]
                shape = model_name.split("-")[1]
                width, depth = shape.split("x")[0], shape.split("x")[1]
                for k, v in data.items():
                    if k == "main":
                        k = 83475
                    else:
                        k = int(k.split("_")[1])

                    all_data.append(
                        {
                            "model": model_name,
                            "revision": int(k),
                            "loss": v,
                            "width": int(width),
                            "depth": int(depth),
                            "dataset": dataset_labels[(edu, dclm)],
                        }
                    )

        df = pd.DataFrame(all_data)
        df["tokens"] = df["revision"] * 2048 * 2048

        df["params"] = df.apply(
            lambda row: param_counter(row["width"], row["depth"]), axis=1
        )
        df["params_round"] = df["params"].apply(round_sig)
        df.loc[df["params_round"] == 90000000, "params_round"] = 100000000
        df.loc[df["params_round"] == 70000000, "params_round"] = 100000000

        df["flops_per_token"] = df.apply(
            lambda row: flops_per_token_gqa(row["width"], row["depth"]), axis=1
        )
        df["total_flops"] = df["flops_per_token"] * df["tokens"]
        df = df[df["params_round"] >= 500_000_000]

        all_dfs.append(df)

    # all_dfs = []
    # Combine all datasets
    dolma_df = get_our_data()
    dolma_df = dolma_df[dolma_df["params_round"] >= 500_000_000]
    dolma_df["dataset"] = "Dolma"

    all_dfs.append(dolma_df)
    full_df = pd.concat(all_dfs, ignore_index=True)

    # Setup styles
    dataset_colors = {
        "FineWeb-Edu": "tab:blue",
        "DCLM": "tab:green",
        "FineWeb": "tab:red",
        "Dolma": "tab:orange",
    }

    # Unique marker per model
    plt.figure(figsize=(10, 8))

    all_models = sorted(full_df["model"].unique())
    # fmt: off
    markers = [
        "x", "o", "v", "^", "s", "P", "*", "D", "<", ">", "h", "8", "H", "|", "_"
    ]
    # fmt: on
    model_markers = {
        model: markers[i % len(markers)] for i, model in enumerate(all_models)
    }

    for model_name in all_models:
        for dataset, df_dataset in full_df.groupby("dataset"):
            df_plot = df_dataset[df_dataset["model"] == model_name].sort_values(
                "revision"
            )

            if df_plot.empty:
                continue

            label = (
                f"{model_name.replace('Gemstone-', '')}" if dataset == "Dolma" else None
            )

            plt.plot(
                df_plot["total_flops"],
                df_plot["loss"],
                label=label,
                linestyle=("-"),  # consistent style now
                color=dataset_colors[dataset],
                marker=model_markers[model_name],
                markersize=12,
                linewidth=2,
            )

    plt.xscale("log")
    plt.xlabel("Total FLOPs (log scale)")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tick_params(axis="both", which="both", labelsize=24)

    # Create legend: one for model identities, one for datasets
    handles, labels = plt.gca().get_legend_handles_labels()

    grey_handles = [
        mlines.Line2D(
            [],
            [],
            color="grey",
            linestyle=h.get_linestyle(),
            marker=h.get_marker(),
            markersize=h.get_markersize(),
            label=label,
        )
        for h, label in zip(handles, labels)
    ]
    # Dataset legend manually
    dataset_handles = [
        mlines.Line2D([], [], color=color, label=name)
        for name, color in dataset_colors.items()
    ]

    legend1 = plt.legend(
        handles=grey_handles,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.25),
        ncol=4,
        # title="Model",
        fontsize=19,
    )

    legend2 = plt.legend(
        handles=dataset_handles,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.15),
        ncol=4, 
        fontsize=19,
    )

    plt.gca().add_artist(legend1)
    plt.tight_layout()
    plt.savefig(
        "fineweb_combined_loss_lines.pdf",
        bbox_inches="tight",
        bbox_extra_artists=[legend1, legend2],
    )


def process_outputs():
    for data in ["edu", "dclm"]:
        path = f"fineweb_{data}_losses"
        models = os.listdir(path)

        dfs = []
        for model in models:
            model_name = model.replace(".json", "")
            shape = model_name.split("-")[1]
            width, depth = shape.split("x")[0], shape.split("x")[1]

            with open(f"{path}/{model}", "r") as f:
                loss_store = json.load(f)

            expected_keys = [
                f"step_{str(477*step).zfill(8)}" for step in range(5, 175, 5)
            ] + ["main"]
            missing_keys = [key for key in expected_keys if key not in loss_store]
            if missing_keys:
                print(f"{model_name} {data} missing keys: {missing_keys}")

            df = pd.DataFrame(list(loss_store.items()), columns=["step", "final_loss"])

            df["run_name"] = model_name
            df["width"] = int(width)
            df["depth"] = int(depth)

            df["step"] = df["step"].apply(
                lambda x: 83475 if x == "main" else int(x.split("_")[1])
            )
            df["tokens"] = df["step"] * 2048 * 2048
            df["params_active_precise"] = df.apply(
                lambda row: param_counter(row["width"], row["depth"]), axis=1
            )

            df["num_nodes"] = None
            df["seconds_per_step"] = None

            dfs.append(df)

        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_json(f"{path}_combined.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    run()

import matplotlib.pyplot as plt
from matplotlib import font_manager
from plotting_utils import import_times_new_roman
import numpy as np


def flops_accounting(df, save_postfix):
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        df["width"],
        df["depth"],
        c=df["flops_ratio"],
        cmap="viridis",
        s=60,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Our FLOPs per Token / (6 $\\times$ parameters)", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    ax.set_xlabel("Width", fontsize=30)
    ax.set_ylabel("Depth", fontsize=30)

    plt.savefig(f"figures/flops_accounting{save_postfix}.pdf", bbox_inches="tight")
    exit()


def plot_efficiency_vs_loss(
    df, loss_key, second_col_df_name, save_postfix, get_resource_hull
):
    """
    Creates two subplots showing overspend heatmaps on width/depth axes for FLOPs and GPU Hours
    """
    import_times_new_roman(font_manager, plt, font_size=36)
    print("\nStarting plot_efficiency_vs_loss")

    # Filter for valid data points at ~300B tokens
    token_filtered_df = df[
        (df["width"] > 0)
        & (df["depth"] > 0)
        & (df["FLOPs"] > 0)
        & (df[second_col_df_name] > 0)
        & ((df["tokens"] >= 299e9) & (df["tokens"] <= 301e9))
    ]

    # Get the optimal frontiers from the filtered data
    flops_minimizers = get_resource_hull(
        token_filtered_df, loss_key=loss_key, resource_key="FLOPs"
    )
    gpu_minimizers = get_resource_hull(
        token_filtered_df, loss_key=loss_key, resource_key=second_col_df_name
    )

    def get_percentage_overspend(point, frontier, resource_key):
        """Calculate percentage overspend compared to frontier"""
        point_loss = point[loss_key]
        idx = (frontier[loss_key] - point_loss).abs().idxmin()
        frontier_point = frontier.iloc[idx]
        percentage_overspend = (
            point[resource_key] / frontier_point[resource_key] - 1
        ) * 100
        return percentage_overspend, frontier_point

    # Calculate overspend for each configuration
    flops_efficiencies = []
    gpu_efficiencies = []
    widths = []
    depths = []
    configs = []

    for idx, point in token_filtered_df.iterrows():
        flops_overspend, _ = get_percentage_overspend(point, flops_minimizers, "FLOPs")
        gpu_overspend, _ = get_percentage_overspend(
            point, gpu_minimizers, second_col_df_name
        )

        flops_efficiencies.append(flops_overspend)
        gpu_efficiencies.append(gpu_overspend)
        widths.append(point["width"])
        depths.append(point["depth"])
        configs.append((point["width"], point["depth"]))

    # Create figure with two subplots (1x2)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

    def setup_width_depth_subplot(
        ax, values, title, frontier_df=None, show_legend=False
    ):
        """
        Setup width/depth subplot with frontier points overlay
        """
        # Calculate 75th percentile for colormap normalization
        vmax = np.percentile(values, 75)

        # Find extreme points (above 75th percentile)
        extreme_indices = [i for i, v in enumerate(values) if v > vmax]
        extreme_indices = sorted(
            extreme_indices, key=lambda i: values[i], reverse=True
        )[:3]

        # Main scatter plot with colormap starting at 0
        scatter = ax.scatter(
            widths,
            depths,
            c=[min(v, vmax) for v in values],
            cmap="YlOrRd",
            s=300,
            edgecolors="black",
            linewidth=0.5,
            norm=plt.Normalize(vmin=0, vmax=vmax),
        )

        # Annotate extreme points with smart positioning
        annotation_positions = [
            (10, 10),  # default offset
            (10, -15),  # below point
            (-60, 10),  # left of point
        ]

        for idx, i in enumerate(extreme_indices):
            ax.annotate(
                f"{values[i]:.0f}%",
                (widths[i], depths[i]),
                xytext=annotation_positions[idx],
                textcoords="offset points",
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
                fontsize=24,
            )

        # Add frontier points
        if frontier_df is not None:
            frontier_scatter = ax.scatter(
                frontier_df["width"],
                frontier_df["depth"],
                color="red",
                marker="*",
                s=500,
                label="Point on Frontier",
                zorder=5,
            )
            if show_legend:
                # Place legend in bottom left corner
                ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))

        ax.set_xlabel("Width")
        ax.set_ylabel("Depth")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", which="major")
        ax.set_title(title)

        return scatter

    # Create the two heatmap plots
    scatter1 = setup_width_depth_subplot(
        ax1,
        flops_efficiencies,
        "FLOPs Overspend by Architecture",
        flops_minimizers,
        show_legend=True,
    )  # Only show legend on left plot
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("FLOPs % Overspend", fontsize=30)

    scatter2 = setup_width_depth_subplot(
        ax2,
        gpu_efficiencies,
        "GPU Hours Overspend by Architecture",
        gpu_minimizers,
        show_legend=False,
    )
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label("GPU Hours % Overspend", fontsize=30)

    # Remove the central figure legend
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"figures/efficiency_vs_loss{save_postfix}.pdf", bbox_inches="tight")
    plt.close()

    # Print some debug information
    print("\nPoints with highest FLOPs overspend:")
    for i in sorted(
        range(len(flops_efficiencies)),
        key=lambda i: flops_efficiencies[i],
        reverse=True,
    )[:5]:
        print(f"\nPoint: W{configs[i][0]}/D{configs[i][1]}")
        print(f"Loss: {token_filtered_df.iloc[i][loss_key]:.4f}")
        print(f"FLOPs Overspend: {flops_efficiencies[i]:.1f}%")
        print(f"GPU Hours Overspend: {gpu_efficiencies[i]:.1f}%")
    exit()

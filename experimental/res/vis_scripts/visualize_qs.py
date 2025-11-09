import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from tex_setup import set_size

def load_q_and_actions(q_filename, actions_filename):
    """Load Q-values and corresponding actions."""
    qs = jnp.load(q_filename)               # shape (num_points, something)
    actions = jnp.load(actions_filename)    # shape (N, 3)
    return np.array(qs), np.array(actions)


def visualize_multiple_relative(file_pairs, titles=None, normalize=False, show_actions=True):
    """
    Visualize multiple Q-value voxel grids side-by-side in a publication-ready style.
    Each subplot has its own color scale: vmin = dataset min, vmax = vmin + 30.
    
    Args:
        file_pairs: list of (q_file, actions_file)
        titles: optional list of subplot titles
        normalize: if True, z-score normalize Q-values before plotting
        show_actions: if True, overlay action points
    """
    # === Load data ===
    data_rel = []
    for q_file, actions_file in file_pairs:
        q_mean, actions = load_q_and_actions(q_file, actions_file)
        if q_mean.ndim > 1:
            q_mean = q_mean.min(axis=1)
        if normalize:
            q_mean = (q_mean - q_mean.mean()) / (q_mean.std() + 1e-8)
        data_rel.append((q_mean, actions))

    n = len(data_rel)
    fig = plt.figure(figsize=set_size(width_fraction=0.9, height_fraction=0.3, subplots=(1, n)))
    axes = []

    for i, (q_mean, actions) in enumerate(data_rel):
        # --- Per-dataset color scale ---
        q_min = q_mean.min()
        # round below
        q_min = np.floor(q_min)
        q_max = q_min + 25
        

        norm = colors.Normalize(vmin=q_min, vmax=q_max)
        cmap = cm.coolwarm



        # Clip q-values to range
        q_mean = np.clip(q_mean, q_min, q_max)
     

        # --- Reshape into voxel grid ---
        grid_res = int(round(len(q_mean) ** (1/3)))
        if grid_res**3 != len(q_mean):
            print(f"⚠️ Warning: expected {grid_res**3} points, got {len(q_mean)}")
        q_grid = q_mean.reshape((grid_res, grid_res, grid_res))
        axes_lin = np.linspace(-1, 1, grid_res + 1)
        X, Y, Z = np.meshgrid(axes_lin, axes_lin, axes_lin, indexing="ij")

        # --- 3D subplot ---
        ax = fig.add_subplot(1, n, i + 1, projection='3d')
        axes.append(ax)

        # Map colors
        facecolors = cmap(norm(q_grid))
        facecolors[..., -1] = 0.5  # semi-transparent

        # Plot voxels
        ax.voxels(X, Y, Z, np.ones_like(q_grid, dtype=bool),
                  facecolors=facecolors, edgecolor=None)

        # Overlay actions
        if show_actions:
            ax.scatter(actions[:, 0], actions[:, 1], actions[:, 2],
                       c='r', s=40, label='Actions')

        # Compute gradient for analysis
        dx = axes_lin[1] - axes_lin[0]
        dq_da1, dq_da2, dq_da3 = np.gradient(q_grid, dx, dx, dx)
        grad_norm = np.sqrt(dq_da1**2 + dq_da2**2 + dq_da3**2)
        print("Q-value stats", f"min {q_grid.min():.3e}, max {q_grid.max():.3e}, mean {q_grid.mean():.3e}")
        print(f"Gradient stats: min {grad_norm.min():.3e}, ",
              f"max {grad_norm.max():.3e}, mean {grad_norm.mean():.3e}")

        # Axis labels (LaTeX style)
        ax.set_xlabel(r'$\mathbf{a_1}$', labelpad=-18)
        ax.set_ylabel(r'$\mathbf{a_2}$', labelpad=-18)
        ax.set_zlabel(r'$\mathbf{a_3}$', labelpad=-18)

        # Hide ticks and axes lines
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.w_xaxis.line.set_color((0,0,0,0))
        ax.w_yaxis.line.set_color((0,0,0,0))
        ax.w_zaxis.line.set_color((0,0,0,0))
        ax.grid(False)

        # Axis limits
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])

        # Title
        if titles:
            ax.set_title(titles[i])

        # --- Add per-subplot horizontal colorbar ---
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])  # required
        cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal',
                            fraction=0.05, pad=0.08)
        cbar.set_ticks([q_min, q_max])
        cbar.set_ticklabels([f"{q_min:.0f}", f"{q_max:.0f}"])

    # --- Save figure ---
    filename = file_pairs[0][0].replace(".npy", "_q_multiple.pdf")
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.18, top=0.9, wspace=0.3)
    fig.savefig("compare.pdf", bbox_inches='tight', pad_inches=0.2, dpi=300)

def visualize(q_file, actions_file, title=None, normalize=False):
    """
    Visualize a single Q-value voxel grid from file with vertical colorbar on the right, optimized for publication.

    Args:
        q_file (str): Path to .npy file with Q-values.
        actions_file (str): Path to .npy file with action coordinates.
        title (str): Ignored, no title is displayed.
        normalize (bool): If True, show scale-invariant (z-score) Q-values.
    """
    # Load data
    q_mean, actions = load_q_and_actions(q_file, actions_file)

    # Use minimum over actions if Q is 2D
    if q_mean.ndim > 1:
        q_mean = q_mean.min(axis=1)

    num_points = len(q_mean)
    grid_res = int(round(num_points ** (1 / 3)))
    if grid_res**3 != num_points:
        print(f"⚠️ Warning: expected {grid_res**3} points, got {num_points}")

    if normalize:
        q_mean = (q_mean - q_mean.mean()) / (q_mean.std() + 1e-8)

    q_grid = q_mean.reshape((grid_res, grid_res, grid_res))
    axes_lin = np.linspace(-1, 1, grid_res + 1)
    X, Y, Z = np.meshgrid(axes_lin, axes_lin, axes_lin, indexing="ij")

    # Compute gradient for analysis
    dx = axes_lin[1] - axes_lin[0]
    dq_da1, dq_da2, dq_da3 = np.gradient(q_grid, dx, dx, dx)
    grad_norm = np.sqrt(dq_da1**2 + dq_da2**2 + dq_da3**2)
    print("Q-value stats", f"min {q_grid.min():.3e}, max {q_grid.max():.3e}, mean {q_grid.mean():.3e}")
    print(f"Gradient stats: min {grad_norm.min():.3e}, ",
          f"max {grad_norm.max():.3e}, mean {grad_norm.mean():.3e}")

    # Set up colormap and normalization
    cmap = cm.coolwarm
    norm = colors.Normalize(vmin=q_grid.min(), vmax=q_grid.max())
    facecolors = cmap(norm(q_grid))
    facecolors[..., -1] = 0.5  # Transparency for clarity

    # Create figure with slightly larger size to accommodate all elements
    fig = plt.figure(figsize=set_size(fraction=0.3, subplots=(1, 1)))
    ax = fig.add_subplot(111, projection="3d")

    # Plot voxels
    ax.voxels(X, Y, Z, np.ones_like(q_grid, dtype=bool),
              facecolors=facecolors, edgecolor=None)

    # Set axis labels with LaTeX formatting, placed right next to axes
    ax.set_xlabel(r'$\mathbf{a_1}$', labelpad=-18)
    ax.set_ylabel(r'$\mathbf{a_2}$', labelpad=-18)
    ax.set_zlabel(r'$\mathbf{a_3}$', labelpad=-18)

    # Remove all axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.w_xaxis.line.set_color((0,0,0,0))  # fully transparent
    ax.w_yaxis.line.set_color((0,0,0,0))
    ax.w_zaxis.line.set_color((0,0,0,0))
    ax.grid(False)

    # Set axis limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    # Add vertical colorbar on the right with min/max ticks, no label
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        location='left',
        fraction=0.03,
        pad=0.08,
        shrink=0.5,
        aspect=15
    )
    cbar_ticks = [q_grid.min(), q_grid.max()]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{int(round(x))}" for x in cbar_ticks])
    # Save figure with explicit bounding box to include LaTeX labels

    filename = q_file.replace(".npy", "_q_single.pdf")
    fig.savefig(filename, bbox_inches="tight", pad_inches=0.2, dpi=300)


print("cql vs edac expert")
visualize_multiple_relative(
        [
            ("3d_vis/new_figures/sacn+10+cql+lag=0.5_seed=0_hopper-expert-v2.npy",
             "3d_vis/new_figures/sacn+10+cql+lag=0.5_seed=0_hopper-expert-v2_actions.npy"),
            ("3d_vis/new_figures/sacn+10+none+lag=0.5_seed=0_hopper-expert-v2.npy",
             "3d_vis/new_figures/sacn+10+none+lag=0.5_seed=0_hopper-expert-v2_actions.npy"),
        ],
        titles=["$\mathcal{R}_{CQL}$", "$\mathcal{R}_{EDAC}$"],
        show_actions=False
)


print("done")



             





visualize(
    "sacn+pbrl+lag=0.5_seed=0_hopper-expert-v2.npy",
    "sacn+pbrl+lag=0.5_seed=0_hopper-expert-v2_actions.npy"
)

visualize(
    "sacn+pbrl+lag=0.5_seed=0_hopper-medium-replay-v2.npy",
    "sacn+pbrl+lag=0.5_seed=0_hopper-medium-replay-v2_actions.npy"
)

print("iql vs sac medium replay")
visualize_multiple_relative(
        [
            ("iql_q_values.npy",
             "iql_matching_actions.npy"),
            ("sacn+10+none+lag=0.5+edac_seed=0_hopper-medium-replay-v2.npy",
             "sacn+10+none+lag=0.5+edac_seed=0_hopper-medium-replay-v2_actions.npy"),
        ],

        titles=[],
        show_actions=False,
)

print("end")
visualize_multiple_relative(
        [
            ("iql_expert_q_values.npy",
             "iql_matching_actions.npy"),
            ("sacn+10+none+lag=0.5+edac_seed=0_hopper-expert-v2.npy",
             "sacn+10+none+lag=0.5+edac_seed=0_hopper-expert-v2_actions.npy"),
        ],

        titles=[],
        show_actions=False,
)


# expert sac
visualize("sacn+10+none+lag=0.5+edac_seed=0_hopper-expert-v2.npy",
          "sacn+10+none+lag=0.5+edac_seed=0_hopper-expert-v2_actions.npy")

visualize("sacn+10+none+lag=0.5+edac_seed=0_hopper-medium-replay-v2.npy",
          "sacn+10+none+lag=0.5+edac_seed=0_hopper-medium-replay-v2_actions.npy")

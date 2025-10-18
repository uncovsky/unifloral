import jax
import jax.numpy as jnp
import os
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

latex_textwidth = 361.34999
tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)

def set_size(width=latex_textwidth, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2
    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)


def visualize_q_vals(args, agent_state, dataset, q_apply):

    n_per_axis = 10
    action_dim = dataset.action.shape[1]
    axes = [jnp.linspace(-1, 1, n_per_axis) for _ in range(action_dim)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    actions = jnp.stack([m.flatten() for m in mesh], axis=-1)
    n_samples = actions.shape[0]

    close_states = jnp.linalg.norm(dataset.obs - dataset.obs[0], axis=1)
    close_indices = jnp.argsort(close_states)[:20]
    matching_actions = dataset.action[close_indices]

    state = dataset.obs[0].reshape(1, -1).repeat(n_samples, axis=0)
    q_values = q_apply(agent_state.vec_q.params, state, actions)
    name = f"sacn+{args.num_critics}+{args.critic_regularizer}+lag={args.critic_lagrangian}_seed={args.seed}_{args.dataset_name}.npy"
    name_actions = f"sacn+{args.num_critics}+{args.critic_regularizer}+lag={args.critic_lagrangian}_seed={args.seed}_{args.dataset_name}_actions.npy"

    os.makedirs("figures", exist_ok=True)
    np.save(os.path.join("figures", name), np.array(q_values))
    np.save(os.path.join("figures", name_actions), np.array(matching_actions))


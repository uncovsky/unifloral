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

def visualize_hopper(args, agent_state, dataset, q_apply, actor_apply, rng):

    x_bounds = (-1, 1)
    y_bounds = (-1, 1)
    z_bounds = (-1, 1)


    # --- Step 2: Define grid resolution ---
    n_per_axis = 10
    x = jnp.linspace(*x_bounds, n_per_axis)
    y = jnp.linspace(*y_bounds, n_per_axis)
    z = jnp.linspace(*z_bounds, n_per_axis)

    # --- Step 3: Create 3D grid of actions ---
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")  # shape (n_per_axis, n_per_axis, n_per_axis)

    # --- Step 4: Flatten into (N, 3) action samples ---
    actions = jnp.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=-1)  # (N, 3)
    n_samples = actions.shape[0]
    state = dataset.obs[0].reshape(1, -1).repeat(n_samples, axis=0)

    distances = jnp.linalg.norm(dataset.obs - dataset.obs[0], axis=1)
    top20_idx = jnp.argsort(distances)[:20]
    matching_actions = dataset.action[top20_idx]
    q_values = q_apply(agent_state.vec_q.params, state, actions).min(axis=-1)

    name = f"sacn+{args.critic_regularizer}+lag={args.critic_lagrangian}_hopper_actions.npy"
    os.makedirs("figures", exist_ok=True)
    np.save(os.path.join("figures", name), np.array(q_values.reshape(n_per_axis, n_per_axis, n_per_axis)))



def visualize_q_vals(args, agent_state, dataset, q_apply):

    n_per_axis = 10
    action_dim = dataset.action.shape[1]
    print(action_dim)
    axes = [jnp.linspace(-1, 1, n_per_axis) for _ in range(action_dim)]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    actions = jnp.stack([m.flatten() for m in mesh], axis=-1)
    n_samples = actions.shape[0]

    close_actions = jnp.linalg.norm(dataset.action - dataset.action[0], axis=1)
    close_actions = jnp.argsort(close_actions)[:20]
    matching_actions = dataset.action[close_actions]

    state = dataset.obs[0].reshape(1, -1).repeat(n_samples, axis=0)
    q_values = q_apply(agent_state.vec_q.params, state, actions).min(axis=-1)
    q_grid = q_values.reshape((n_per_axis,) * action_dim)

    name = f"sacn+{args.critic_regularizer}+lag={args.critic_lagrangian}_seed={args.seed}.npy"
    name_actions = f"sacn+{args.critic_regularizer}+lag={args.critic_lagrangian}_actions_seed={args.seed}.npy"

    os.makedirs("figures", exist_ok=True)
    np.save(os.path.join("figures", name), np.array(q_grid))
    np.save(os.path.join("figures", name_actions), np.array(matching_actions))



def visualize_bandit(args, agent_state, dataset, q_apply, actor_apply, rng):
    actions = jnp.linspace(-args.action_scale, args.action_scale, 1000).reshape(-1, 1)

    state = dataset.obs[0].reshape(1, -1)
    states = jnp.repeat(state, repeats=1000, axis=0)

    q_values = q_apply(agent_state.vec_q.params, states, actions)

    rng, rng_sample = jax.random.split(rng)
    pi = actor_apply(agent_state.actor.params, state)
    samples = pi.sample(seed=rng_sample, sample_shape=(1000,))
    print("Sampled actions: ", samples.mean(), samples.std())
    samples = jnp.asarray(samples).reshape(-1)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    colors = plt.cm.tab10.colors  # 10 distinct colors
    num_critics = q_values.shape[-1]

    for critic in range(num_critics):
        color = colors[critic % len(colors)]  # wrap if > 10 critics
        ax1.plot(
            actions,
            q_values[:, critic],
            linestyle="dotted",
            color=color,
            alpha=0.8,
            linewidth=2.5,
            label=f"Critic {critic+1}"
        )
    ax1.set_xlabel("Action", fontsize=14)
    ax1.set_ylabel("Q-value", fontsize=14)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.hist(samples, bins=30, range=(-args.action_scale, args.action_scale),
             density=True, alpha=0.4, color="tab:green", label="Action samples")
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    true_values = jnp.where(jnp.abs(actions) > 0.5, 1.0, -1.0)
    ax1.plot(actions, true_values, linestyle='dashed', color='black', label='True Value')

    fig.tight_layout()
    ax1.legend(loc='upper left', fontsize=14)
    plt.title("Q-values", fontsize=16)
    plt.show()



def visualize_reach_bias(args, agent_state, q_apply, actor_apply, rng):


    # Define a 100x100 grid in [0,1]^2
    grid_size = 30
    x = jnp.linspace(0, 1, grid_size)
    y = jnp.linspace(0, 1, grid_size)
    xx, yy = jnp.meshgrid(x, y)

    # Flatten grid into (10000, 2)
    grid_points = jnp.stack([xx.ravel(), yy.ravel()], axis=-1)

    # Append goal information to each state
    goal = jnp.array([1.0, 1.0])
    goal_repeated = jnp.repeat(goal[None, :], grid_points.shape[0], axis=0)
    obs = jnp.concatenate([grid_points, goal_repeated], axis=1)  # shape (10000, 4)
    print(obs)
    # Sample actions from the actor
    rng, rng_actor = jax.random.split(rng)
    pi = actor_apply(agent_state.actor.params, obs)

    #actions = pi.sample(seed=rng_actor)

    # Evaluate Q-values for each (state, action)
    q_values = q_apply(agent_state.vec_q.params, obs, actions)

    # If q_values has shape (10000, num_actions), aggregate (e.g., max)
    q_values = jnp.min(q_values, axis=-1)  # shape (10000,)

    # Reshape back into grid for visualization
    q_grid = q_values.reshape(grid_size, grid_size)

    # Plot the heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(q_grid, origin="lower", extent=[0, 1, 0, 1], cmap="viridis")
    plt.colorbar(label="Q-value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q-value Heatmap with Sampled Actions")
    plt.show()

    return q_grid, actions.reshape(grid_size, grid_size, -1)

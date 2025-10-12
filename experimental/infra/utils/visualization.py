import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

def visualize_q_vals(args, agent_state, dataset, q_apply, actor_apply, rng):
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

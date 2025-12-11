from experiments.reachability_experiments.reach.reach_data_collection import collect_dataset, collect_stitch_dataset


def collect_data_reach():
    seeds = range(9)
    i = 0
    for horizon in [10, 50, 200]:
        for noise in [0.1, 0.2, 0.3]:
            seed = seeds[i]
            i += 1
            collect_stitch_dataset(H=horizon, noise_eps=noise, episodes=1000,
                                   seed=seed)
            #collect_dataset(H=horizon, noise_eps=noise, episodes=1000,
             #               seed=seed)

def print_args(args : object) -> None:
    """
        Prints out the training settings in human-friendly format.
    """
    parameter_semantics = "ood actions sampled" if args.critic_regularizer in ["msg", "pbrl", "filtered_pbrl"] else "temperature"
    print(50 * "=")
    print("Training with the following settings:")
    print("Dataset: ", args.dataset_name, " from ", args.dataset_source)
    print("Ensemble size: ", args.num_critics)
    print("PE operator: ",  "shared" if args.shared_targets else f"independent with {args.beta_id} std penalty")
    print("PI operator: ", args.pi_operator)
    if args.pi_operator == "lcb":
        print(f"\t with LCB penalty: {args.actor_lcb_penalty}")
    print(f"Critic regularizer: {args.critic_regularizer}, with lagrangian: {args.critic_lagrangian} and ", end="")
    print(f"{parameter_semantics}: {args.critic_regularizer_parameter}")
    print(f"Ensemble regularizer: {args.ensemble_regularizer} with lagrangian: {args.reg_lagrangian}")
    if args.critic_norm != "none":
        print(f"Using {args.critic_norm} normalization for critics")
    if args.prior:
        print(f"Using prior functions of depth {args.randomized_prior_depth} and scale {args.randomized_prior_scale}")
    if args.pretrain_updates > 0:
        print(f"Pretraining for {args.pretrain_updates} updates with {args.pretrain_loss} loss and lagrangian {args.pretrain_lagrangian}")
    if args.no_entropy_bonus:
        print("Not using entropy bonus for actor/critic updates")

    print(50 * "=")


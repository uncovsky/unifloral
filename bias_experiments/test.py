import minari

data = minari.load_dataset("square-reach/horizon-2-v0")

for ep in data.iterate_episodes():
    print(ep.rewards)

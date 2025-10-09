from experiments import collect_bandit_data, collect_cql_data

def collect_data():
    collect_cql_data(episodes=10000, seed=0)
    collect_bandit_data(episodes=1000, seed=0)

if __name__ == "__main__":
    collect_data()





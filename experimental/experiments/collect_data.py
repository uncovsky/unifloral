from experiments import collect_bandit_data, collect_cql_data, collect_uniform_data, collect_d_dim_data

def collect_data():
    collect_d_dim_data(episodes=1000, ds=[5,10,20], seeds=[0,1,2])
    collect_uniform_data(episodes=1000, seed=0)
    collect_cql_data(episodes=10000, seed=0)
    collect_bandit_data(episodes=1000, seed=0)

if __name__ == "__main__":
    collect_data()





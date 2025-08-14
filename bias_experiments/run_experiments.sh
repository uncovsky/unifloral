export JAX_PLATFORM_NAME=cpu

python3 bias_pendulum_test.py --num-critics 2 --dataset pendulum/uniform-v0 --cql-min-q-weight 0.0 --num-updates 20000

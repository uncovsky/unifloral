nice -n 19 podman run -d \
                      --gpus all \
                      --shm-size=1g \
                      --ulimit memlock=-1 \
                      --ulimit stack=67108864 \
                      --ulimit nproc=65535:65535 \
                      --memory=32g \
                      --pids-limit=5000 \
                      -v /home/xuncovsk/thesis/repo/unifloral/:/work/rl/ \
                      --name uni \
                       unifloral tail -f /dev/null

podman exec -it uni bash

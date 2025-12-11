build:
	git pull
	docker build . -t unifloral
up:
	nice -n 19 podman run -d \
			      --gpus all \
			      --name uni \
			       unifloral tail -f /dev/null
	podman exec -it uni bash

down:
	podman stop uni
	podman rm uni



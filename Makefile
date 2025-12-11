.PHONY: build up down up-gpu

build:
	git pull
	docker build . -t unifloral
up:
	nice -n 19 docker run -d \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

up-gpu:
	nice -n 19 docker run -d \
	    --gpus "device=0" \
	    --name uni \
	    unifloral tail -f /dev/null
	docker exec -it uni bash

down:
	docker stop uni
	docker rm uni



.PHONY: build up down

build:
	git pull
	docker build . -t unifloral
up:
	nice -n 19 docker run -d \
			      --gpus all \
			      --name uni \
			       unifloral tail -f /dev/null
	docker exec -it uni bash

down:
	docker stop uni
	docker rm uni



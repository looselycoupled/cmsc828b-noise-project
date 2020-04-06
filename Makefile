# Shell to use with Make
SHELL := /bin/bash

# Export targets not associated with files
.PHONY: clean build push

# Clean build files
clean:
	find . -name "*.pyc" -print0 | xargs -0 rm -rf
	find . -name "__pycache__" -print0 | xargs -0 rm -rf

# Build the docker image
build:
	docker build -t looselycoupled/cmsc828b-noise-simple -f docker/Dockerfile .

# Push the docker image
push:
	docker push looselycoupled/cmsc828b-noise-simple

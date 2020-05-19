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
	cd kubernetes/ && ./hide-keys.sh
	docker build -t looselycoupled/cmsc828b-tensor2tensor -f docker/Dockerfile .

# Push the docker image
push:
	docker push looselycoupled/cmsc828b-tensor2tensor


# Start the mini job
mini:
	cd kubernetes/ && ./add-keys.sh
	cd kubernetes/ &&  kubectl apply -f mini.t2t.job.yaml

baseline:
	cd kubernetes/ && ./add-keys.sh
	cd kubernetes/ &&  kubectl apply -f baseline.t2t.job.yaml

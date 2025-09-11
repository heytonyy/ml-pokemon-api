# Makefile for Pokemon API Docker operations

# Variables
IMAGE_NAME = pokemon-api
CONTAINER_NAME = ml-pokemon-api-1
VERSION = latest
HOST_PORT = 5050
CONTAINER_PORT = 5050

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME):$(VERSION) .

# Build with no cache
build-no-cache:
	docker build --no-cache -t $(IMAGE_NAME):$(VERSION) .

# Run container with docker run
run:
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/uploads:/app/uploads \
		-e FLASK_ENV=production \
		-e MODEL_PATH=/app/models/pokemon_v1.pth \
		-e CLASS_MAPPING_PATH=/app/models/class_mapping.json \
		$(IMAGE_NAME):$(VERSION)

# Run container interactively
run-interactive:
	docker run -it --rm \
		-p $(HOST_PORT):$(CONTAINER_PORT) \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/uploads:/app/uploads \
		-e FLASK_ENV=production \
		$(IMAGE_NAME):$(VERSION) /bin/bash

# Use docker-compose
up:
	docker-compose up -d

# Build and run with docker-compose
up-build:
	docker-compose up -d --build

# Stop docker-compose services
down:
	docker-compose down

# View logs
logs:
	docker-compose logs -f pokemon-api

# Stop and remove container
stop:
	docker stop $(CONTAINER_NAME) || true
	docker rm $(CONTAINER_NAME) || true

# Remove image
clean-image:
	docker rmi $(IMAGE_NAME):$(VERSION) || true

# Remove all related containers and images
clean-all: stop
	docker-compose down --rmi all --volumes || true
	docker rmi $(IMAGE_NAME):$(VERSION) || true
	docker system prune -f

# Show running containers
ps:
	docker ps

# Shell into running container
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# Check container health
health:
	curl -f http://localhost:$(HOST_PORT)/health || echo "Health check failed"

# Test API
test:
	curl -X GET http://localhost:$(HOST_PORT)/testing || echo "API test failed"

# Copy model files to container (if needed)
copy-models:
	docker cp ./models/. $(CONTAINER_NAME):/app/models/

# Backup models from container
backup-models:
	mkdir -p ./models_backup
	docker cp $(CONTAINER_NAME):/app/models/. ./models_backup/

# View container resource usage
stats:
	docker stats $(CONTAINER_NAME)

# Restart services
restart: down up

# Follow logs in real-time
logs-follow:
	docker-compose logs -f

# Quick development cycle: rebuild and restart
dev: down up-build logs-follow

help:
	@echo "Available commands:"
	@echo "  build          - Build Docker image"
	@echo "  build-no-cache - Build Docker image without cache"
	@echo "  run            - Run container with docker run"
	@echo "  run-interactive- Run container interactively"
	@echo "  up             - Start services with docker-compose"
	@echo "  up-build       - Build and start with docker-compose"
	@echo "  down           - Stop docker-compose services"
	@echo "  logs           - View service logs"
	@echo "  logs-follow    - Follow logs in real-time"
	@echo "  restart        - Restart docker-compose services"
	@echo "  dev            - Quick dev cycle: rebuild, restart, follow logs"
	@echo "  stop           - Stop and remove container"
	@echo "  clean-image    - Remove Docker image"
	@echo "  clean-all      - Remove all containers, images, and volumes"
	@echo "  ps             - Show running containers"
	@echo "  shell          - Shell into running container"
	@echo "  health         - Check API health"
	@echo "  test           - Test API endpoints"
	@echo "  copy-models    - Copy models to container"
	@echo "  backup-models  - Backup models from container"
	@echo "  stats          - View container resource usage"

.PHONY: build build-no-cache run run-interactive up up-build down logs logs-follow restart dev stop clean-image clean-all ps shell health test copy-models backup-models stats help
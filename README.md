# Pokemon Classifier API

TODO: Update this README later... general docker instructions

## 🐳 Quick Start With Docker

### Prerequisites
- Docker installed
- Docker Compose installed
- Trained model files (`pokemon_v1.pth` and `class_mapping.json`)

### Project Structure
```
whos-that-pokemon-api/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── Makefile
├── app.py 
├── model.py 
├── models/
│   ├── pokemon_v1.pth
│   └── class_mapping.json
└── uploads/
```

## 🚀 Deployment Options

### Option 1: Docker Compose (Recommended)

```bash
# Build and start the service
make up-build

# Or manually:
docker-compose up -d --build
```

### Option 2: Docker Run

```bash
# Build the image
make build

# Run the container
make run

# Or manually:
docker build -t pokemon-classifier .
docker run -d -p 5050:5050 --name pokemon-api pokemon-classifier
```

## 📝 Configuration

### Environment Variables

Set these in `docker-compose.yml` or when running the container:

```yaml
environment:
  - FLASK_ENV=production
  - MODEL_PATH=/app/models/pokemon_v1.pth
  - CLASS_MAPPING_PATH=/app/models/class_mapping.json
```

### Volume Mounts

The setup includes volume mounts for:
- `./models:/app/models` - Persist model files
- `./uploads:/app/uploads` - Temporary file uploads

## 🔧 Useful Commands

### Using Makefile (Recommended)

```bash
# Build and start
make up-build

# View logs
make logs

# Stop services
make down

# Check health
make health

# Test API
make test

# Clean everything
make clean-all
```

## 🚀 Production Deployment

### For Production Use:

1. **Use Multi-stage Build** (for smaller images):
```dockerfile
# Add to Dockerfile
FROM python:3.9-slim as builder
# ... build dependencies
FROM python:3.9-slim as runtime
COPY --from=builder /app /app
```

2. **Add Nginx Reverse Proxy**:
```yaml
# Uncomment nginx service in docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
```

3. **Use Docker Secrets** for sensitive data:
```yaml
secrets:
  model_path:
    file: ./secrets/model_path.txt
```

4. **Add Resource Limits**:
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

## 🔒 Security Considerations

1. **Don't include model files in image** - use volumes
2. **Use non-root user**:
```dockerfile
RUN useradd -m -u 1001 appuser
USER appuser
```

3. **Scan for vulnerabilities**:
```bash
docker scan pokemon-classifier
```

4. **Use specific base image versions**:
```dockerfile
FROM python:3.9.18-slim
```

## 🐛 Troubleshooting

### Common Issues:

**Container won't start:**
```bash
docker logs pokemon-api  # Check logs
make shell               # Get shell access
```

**Model files not found:**
```bash
# Ensure models are in the right location
ls -la models/
# Check volume mounts
docker inspect pokemon-api
```

**Out of memory:**
```bash
# Check resource usage
make stats
# Increase Docker memory limits
```

**Port already in use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8080:5050"  # Use different host port
```

## 📈 Scaling & Performance

### Horizontal Scaling:
```bash
# Scale with docker-compose
docker-compose up -d --scale pokemon-api=3
```

### Add Load Balancer:
```yaml
# Add to docker-compose.yml
nginx:
  image: nginx:alpine
  depends_on:
    - pokemon-api
```

### GPU Support:
```dockerfile
# For GPU inference
FROM nvidia/cuda:11.8-runtime-ubuntu20.04
```

```yaml
# In docker-compose.yml
runtime: nvidia
environment:
  - NVIDIA_VISIBLE_DEVICES=all
```

## 🎯 Next Steps

1. Set up CI/CD pipeline
2. Add monitoring (Prometheus/Grafana)
3. Implement caching (Redis)
4. Add rate limiting
5. Set up logging aggregation
6. Consider Kubernetes for production
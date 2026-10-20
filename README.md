---
title: ML Pokemon API
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: gradio_app.py
pinned: false
---

# Pokemon Classifier API

The classifier runs in two ways:
- **`gradio_app.py`**: the Gradio app deployed to a Hugging Face Space on ZeroGPU (free). The React site calls this.
- **`app.py`**: the Flask API, for local development or Docker hosting (see below).

## ☁️ Deploying to Hugging Face Spaces (ZeroGPU)

The model weights (`models/`) are too large for git, so they live in the Hugging Face model repo
[`heytony/pokemon-classifier`](https://huggingface.co/heytony/pokemon-classifier) and are downloaded when the Space starts.

Free accounts can host up to 2 ZeroGPU Spaces if the email is verified and the account is older than 30 days.

1. Upload the weights (one time, or whenever you retrain). `hf upload` creates the repo if it doesn't exist:
   ```bash
   hf auth login   # paste a token with "Write" access from https://huggingface.co/settings/tokens
   hf upload heytony/pokemon-classifier models/pokemon_v1.pth pokemon_v1.pth
   hf upload heytony/pokemon-classifier models/class_mapping.json class_mapping.json
   ```
2. Create a new Space at https://huggingface.co/new-space named `ml-pokemon-api`, with SDK **Gradio** → **Blank** template,
   hardware **ZeroGPU**, and visibility **Public**.
3. Upload this code to the Space (rerun to redeploy):
   ```bash
   hf upload heytony/ml-pokemon-api . . --repo-type=space \
     --exclude ".venv/*" --exclude "models/*" --exclude ".git/*" --exclude "*__pycache__*" \
     --exclude "*.DS_Store" --exclude "blueprints/pokemon_text_generation/*"
   ```
4. The app is at https://huggingface.co/spaces/heytony/ml-pokemon-api. The React site connects to it by Space id with `@gradio/client`.

Run the Gradio app locally with `python gradio_app.py` (http://127.0.0.1:7860). It uses `models/` when present.

| Env var | Default | Purpose |
|---|---|---|
| `HF_MODEL_REPO` | `heytony/pokemon-classifier` in `gradio_app.py`, unset in `app.py` | HF model repo to download weights from when they aren't on disk |
| `MODEL_PATH` (Flask only) | `models/pokemon_v1.pth` | Local weights path |
| `CLASS_MAPPING_PATH` (Flask only) | `models/class_mapping.json` | Local class mapping path |
| `CORS_ORIGINS` (Flask only) | `https://heytonyy.github.io,http://localhost:5173` | Origins allowed to call the API |
| `PORT` (Flask only) | `7860` in Docker, `5050` with `python app.py` | Port to listen on |

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
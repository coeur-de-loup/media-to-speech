# Docker Compose Setup - Media-to-Text Microservice

This document provides comprehensive guidance for running the Media-to-Text microservice using Docker Compose with a multi-container setup.

## Architecture Overview

The Docker Compose setup orchestrates three main services plus optional monitoring:

### Core Services
1. **API Service** - FastAPI application with uv dependency management
2. **Redis Service** - Job state management and pub/sub messaging  
3. **FFmpeg Service** - Media processing container

### Optional Services
4. **Prometheus** - Metrics collection and monitoring
5. **Grafana** - Metrics visualization and dashboards

## Quick Start

### Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+
- At least 4GB RAM available for containers
- OpenAI API key

### Basic Setup

1. **Clone and navigate to project:**
   ```bash
   git clone <repository-url>
   cd media-to-text
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

3. **Start the services:**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are healthy:**
   ```bash
   docker-compose ps
   docker-compose logs api
   ```

5. **Test the API:**
   ```bash
   curl http://localhost:8000/healthz
   curl http://localhost:8000/docs  # API documentation
   ```

## Environment Configuration

### Required Variables

Create a `.env` file with these essential variables:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional but recommended
DEBUG=false
LOG_LEVEL=INFO
REDIS_MAX_MEMORY=512mb
```

### Complete Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for transcription |
| `API_PORT` | `8000` | Port for API service |
| `DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_MAX_MEMORY` | `512mb` | Redis memory limit |
| `ENABLE_AXIOM` | `false` | Enable Axiom logging |
| `AXIOM_TOKEN` | - | Axiom API token (if using Axiom) |
| `DATA_DIR` | `./data` | Directory for input files |

## Service Details

### API Service (FastAPI)

- **Image:** Built from local Dockerfile
- **Ports:** 8000 (API), 9090 (Metrics)
- **Health Check:** `GET /healthz`
- **Dependencies:** Redis, FFmpeg
- **Resources:** 1GB memory limit, 1 CPU limit

**Key Features:**
- Structured logging with Axiom integration
- Request tracing and observability
- Comprehensive health and readiness checks
- Prometheus metrics endpoint
- Rate-limited OpenAI integration

### Redis Service

- **Image:** `redis:7.2-alpine`
- **Port:** 6379
- **Persistence:** AOF + RDB snapshots
- **Configuration:** Optimized for job management and pub/sub
- **Resources:** 768MB memory limit

**Persistence Strategy:**
- AOF (Append Only File) for durability
- RDB snapshots for backup
- Data stored in named volume `redis_data`

### FFmpeg Service

- **Image:** `linuxserver/ffmpeg:version-6.0-cli`
- **Purpose:** Media processing via `docker exec`
- **Volumes:** Shared media storage
- **Resources:** 2GB memory limit, 2 CPU limit

**Processing Method:**
```bash
# Example of how API uses FFmpeg
docker exec media-to-text-ffmpeg ffmpeg -i input.mp4 -f wav output.wav
```

## Usage Instructions

### Starting Services

```bash
# Start all core services
docker-compose up -d

# Start with monitoring stack
docker-compose --profile monitoring up -d

# Start specific services
docker-compose up -d api redis

# View logs
docker-compose logs -f api
```

### Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (⚠️ destroys data)
docker-compose down -v

# Stop services but keep containers
docker-compose stop
```

### Service Management

```bash
# Check service status
docker-compose ps

# View service logs
docker-compose logs [service_name]

# Restart a service
docker-compose restart api

# Scale API service (if needed)
docker-compose up -d --scale api=2
```

### Using the API

1. **Create a transcription job:**
   ```bash
   curl -X POST "http://localhost:8000/transcriptions/" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@your_audio_file.mp3" \
        -F "language=en"
   ```

2. **Check job status:**
   ```bash
   curl "http://localhost:8000/jobs/{job_id}"
   ```

3. **Get transcript:**
   ```bash
   curl "http://localhost:8000/jobs/{job_id}/transcript"
   ```

## Monitoring and Observability

### Built-in Monitoring

- **Health Checks:** `http://localhost:8000/healthz`
- **Readiness:** `http://localhost:8000/readyz`
- **Metrics:** `http://localhost:8000/metrics`
- **API Documentation:** `http://localhost:8000/docs`

### Optional Prometheus + Grafana

Start monitoring stack:
```bash
docker-compose --profile monitoring up -d
```

Access dashboards:
- **Prometheus:** http://localhost:9091
- **Grafana:** http://localhost:3000 (admin/admin)

### Structured Logging

The API service provides structured JSON logging with:
- Request tracing
- Performance metrics
- Error tracking
- Optional Axiom integration

## Troubleshooting

### Common Issues

#### 1. API Service Won't Start

**Symptoms:** API container exits immediately
```bash
# Check logs
docker-compose logs api

# Common causes:
# - Missing OpenAI API key
# - Port 8000 already in use
# - Insufficient memory
```

**Solutions:**
```bash
# Check environment variables
docker-compose config

# Verify API key is set
grep OPENAI_API_KEY .env

# Check port availability
netstat -tulpn | grep 8000
```

#### 2. Redis Connection Issues

**Symptoms:** API logs show Redis connection errors
```bash
# Check Redis status
docker-compose ps redis
docker-compose logs redis

# Test Redis connectivity
docker-compose exec redis redis-cli ping
```

**Solutions:**
```bash
# Restart Redis
docker-compose restart redis

# Check Redis configuration
docker-compose exec redis cat /usr/local/etc/redis/redis.conf
```

#### 3. FFmpeg Processing Failures

**Symptoms:** Transcription jobs fail during media processing
```bash
# Check FFmpeg availability
docker-compose exec ffmpeg ffmpeg -version

# Check shared storage
docker-compose exec api ls -la /tmp/media-to-text
docker-compose exec ffmpeg ls -la /tmp
```

**Solutions:**
```bash
# Restart FFmpeg service
docker-compose restart ffmpeg

# Check file permissions
docker-compose exec ffmpeg ls -la /workspace
```

#### 4. Out of Memory Issues

**Symptoms:** Services getting killed (OOMKilled)
```bash
# Check resource usage
docker stats

# Check container limits
docker-compose config
```

**Solutions:**
```bash
# Increase memory limits in docker-compose.yml
# Or reduce Redis memory usage
docker-compose exec redis redis-cli CONFIG SET maxmemory 256mb
```

### Health Check Commands

```bash
# Check all service health
docker-compose ps

# Detailed health check
curl -s http://localhost:8000/readyz | jq

# Check individual services
docker-compose exec api curl -f localhost:8000/healthz
docker-compose exec redis redis-cli ping
docker-compose exec ffmpeg ffmpeg -version
```

### Log Analysis

```bash
# View all logs
docker-compose logs

# Follow API logs
docker-compose logs -f api

# Search for errors
docker-compose logs api | grep ERROR

# View structured logs (if using JSON logging)
docker-compose logs api | jq -r '.message'
```

## Performance Optimization

### Resource Tuning

1. **API Service:**
   ```yaml
   # Adjust in docker-compose.yml
   deploy:
     resources:
       limits:
         memory: 2G      # Increase if handling large files
         cpus: '2.0'     # Increase for more parallel processing
   ```

2. **Redis Memory:**
   ```bash
   # Monitor Redis memory usage
   docker-compose exec redis redis-cli INFO memory
   
   # Adjust maxmemory
   REDIS_MAX_MEMORY=1g docker-compose up -d redis
   ```

3. **FFmpeg Processing:**
   ```yaml
   # Adjust FFmpeg resources for large media files
   deploy:
     resources:
       limits:
         memory: 4G
         cpus: '4.0'
   ```

### Volume Optimization

```bash
# Use SSD storage for better performance
# Mount specific host directories
volumes:
  - /fast/ssd/path:/tmp/media-to-text

# Clean up old data periodically
docker volume prune
```

## Security Considerations

### Production Deployment

1. **Environment Variables:**
   ```bash
   # Use Docker secrets for sensitive data
   echo "your_openai_key" | docker secret create openai_key -
   ```

2. **Network Security:**
   ```yaml
   # Restrict port exposure
   ports:
     - "127.0.0.1:8000:8000"  # Only localhost access
   ```

3. **Redis Security:**
   ```bash
   # Enable Redis authentication
   # Edit config/redis.conf
   requirepass your_secure_password
   ```

### File System Security

```yaml
# Run with non-root user
user: "1000:1000"

# Read-only root filesystem
read_only: true
tmpfs:
  - /tmp:noexec,nosuid,size=100m
```

## Development Workflow

### Local Development

```bash
# Development with live reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Mount source code for development
volumes:
  - ./src:/app/src:rw

# Use development environment variables
DEBUG=true
LOG_LEVEL=DEBUG
```

### Testing

```bash
# Run tests against running services
docker-compose exec api python -m pytest

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Building Images

```bash
# Build API image
docker-compose build api

# Build with no cache
docker-compose build --no-cache api

# Build for production
docker-compose -f docker-compose.prod.yml build
```

## Backup and Recovery

### Data Backup

```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp media-to-text-redis:/data/dump.rdb ./backup/

# Backup volumes
docker run --rm -v media-to-text-storage:/source -v $(pwd)/backup:/backup alpine tar czf /backup/media-storage.tar.gz -C /source .
```

### Data Recovery

```bash
# Restore Redis data
docker-compose stop redis
docker cp ./backup/dump.rdb media-to-text-redis:/data/
docker-compose start redis

# Restore volumes
docker run --rm -v media-to-text-storage:/target -v $(pwd)/backup:/backup alpine tar xzf /backup/media-storage.tar.gz -C /target
```

## Advanced Configuration

### Custom Network Configuration

```yaml
networks:
  media-to-text-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
```

### External Services Integration

```yaml
# Connect to external Redis
services:
  api:
    environment:
      - REDIS_URL=redis://external-redis:6379
```

### Load Balancing

```yaml
# Multiple API instances
services:
  api:
    deploy:
      replicas: 3
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
```

## Support and Troubleshooting

### Getting Help

1. **Check service logs first:**
   ```bash
   docker-compose logs --tail=100 api
   ```

2. **Verify configuration:**
   ```bash
   docker-compose config --quiet && echo "✅ Config is valid"
   ```

3. **Health checks:**
   ```bash
   curl -f http://localhost:8000/healthz
   curl -f http://localhost:8000/readyz
   ```

4. **Resource monitoring:**
   ```bash
   docker stats --no-stream
   ```

### Common Commands Reference

```bash
# Essential commands
docker-compose up -d                    # Start all services
docker-compose down                     # Stop all services
docker-compose logs -f api              # Follow API logs
docker-compose ps                       # Check service status
docker-compose restart api              # Restart API service
docker-compose pull                     # Update images
docker-compose build --no-cache         # Rebuild images

# Maintenance commands
docker system prune -f                  # Clean unused resources
docker volume ls                        # List volumes
docker network ls                       # List networks
docker-compose config                   # Validate configuration
```

For additional support, refer to:
- API documentation: http://localhost:8000/docs
- Health status: http://localhost:8000/healthz
- Metrics: http://localhost:8000/metrics
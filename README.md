# Media-to-Text Microservice

A containerized microservice that converts audio and video files to text using OpenAI's Speech-to-Text API. The service handles large files by automatically chunking them and processing in parallel.

## Architecture

The service consists of three Docker containers:

- **API Container**: FastAPI application that handles requests and orchestrates transcription
- **Redis Container**: Job state management and real-time progress updates via pub/sub
- **FFmpeg Container**: Media processing for format conversion and chunking

## Features

- ✅ Support for video (MP4, MKV, AVI) and audio (MP3, AAC, FLAC, WAV) formats
- ✅ Automatic chunking for files larger than 25MB (OpenAI limit)
- ✅ Parallel processing with configurable concurrency
- ✅ Real-time job progress via Redis pub/sub
- ✅ REST API with health checks
- ✅ Docker Compose orchestration
- ✅ Automatic cleanup of temporary files

## Prerequisites

- Docker and Docker Compose
- OpenAI API key

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd media-to-text
   ```

2. **Configure environment**:
   ```bash
   # Create .env file
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running**:
   ```bash
   curl http://localhost:8000/healthz
   ```

## API Endpoints

### POST /transcriptions
Submit a new transcription job.

```bash
curl -X POST "http://localhost:8000/transcriptions" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/app/data/my_video.mp4",
    "language": "en",
    "async_processing": true
  }'
```

Response:
```json
{
  "job_id": "abc123",
  "state": "QUEUED"
}
```

### GET /jobs/{job_id}
Get job status and progress.

```bash
curl "http://localhost:8000/jobs/abc123"
```

Response:
```json
{
  "job_id": "abc123",
  "state": "PROCESSING",
  "progress": 0.42,
  "chunks_done": 5,
  "chunks_total": 12
}
```

### GET /jobs/{job_id}?stream=true
Stream real-time job updates via Server-Sent Events.

### DELETE /jobs/{job_id}
Cancel a running job.

## File Upload

Place media files in the `./data` directory, which is mounted to `/app/data` in the containers:

```bash
mkdir -p data
cp your_video.mp4 data/
```

Then reference the file as `/app/data/your_video.mp4` in API requests.

## Development

### Local Development
```bash
# Install dependencies
uv install

# Run API locally (requires Redis)
uv run uvicorn src.media_to_text.main:app --reload
```

### Testing
```bash
# Run tests
uv run pytest

# With coverage
uv run pytest --cov=src
```

### Code Quality
```bash
# Format code
uv run black src/

# Lint code
uv run ruff check src/

# Type checking
uv run mypy src/
```

## Configuration

Environment variables can be set in `.env` file or Docker Compose environment:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `REDIS_URL` | `redis://redis:6379` | Redis connection URL |
| `OPENAI_MAX_PARALLEL_REQUESTS` | `8` | Parallel OpenAI requests |
| `TEMP_DIR` | `/tmp` | Temporary files directory |
| `DEBUG` | `false` | Enable debug logging |

## Monitoring

- **Health checks**: `GET /healthz` and `GET /readyz`
- **Metrics**: Prometheus metrics available (if enabled)
- **Logs**: Use `docker-compose logs -f api` to view logs

## Scaling

To scale the API service:
```bash
docker-compose up -d --scale api=3
```

## Troubleshooting

### Check service status:
```bash
docker-compose ps
```

### View logs:
```bash
docker-compose logs -f api
docker-compose logs -f redis
docker-compose logs -f ffmpeg
```

### Reset everything:
```bash
docker-compose down -v
docker-compose up -d
```

## License

MIT License
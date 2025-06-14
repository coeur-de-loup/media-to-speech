# Crash Recovery and Idempotency - Media-to-Text Microservice

This document provides comprehensive guidance on the crash recovery and idempotency mechanisms implemented in the Media-to-Text microservice job worker system.

## Architecture Overview

The Media-to-Text microservice implements a robust crash recovery system that ensures job processing can resume after worker crashes, system restarts, or unexpected failures. The system is built on four core pillars:

### Core Components

1. **Startup Recovery Scanner** - Detects interrupted jobs during worker initialization
2. **Job Resumption Engine** - Resumes processing from the last known state
3. **Idempotency Manager** - Prevents duplicate work and ensures consistent results
4. **Recovery Event Publisher** - Tracks and publishes recovery-related events

### Architecture Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Job Worker    │    │   Redis Store    │    │  Recovery       │
│   Startup       │    │                  │    │  Events         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │ 1. Scan for           │                       │
         │    PROCESSING jobs    │                       │
         ├──────────────────────▶│                       │
         │                       │                       │
         │ 2. Get job metadata   │                       │
         │◀──────────────────────┤                       │
         │                       │                       │
         │ 3. Check chunk info   │                       │
         ├──────────────────────▶│                       │
         │                       │                       │
         │ 4. Identify completed │                       │
         │    chunks (idempotency)│                       │
         │◀──────────────────────┤                       │
         │                       │                       │
         │ 5. Resume remaining   │                       │
         │    chunks             │                       │
         ├──────────────────────▶│                       │
         │                       │                       │
         │ 6. Publish recovery   │                       │
         │    events             ├──────────────────────▶│
         │                       │                       │
```

## Recovery Process

### Startup Recovery Sequence

When a job worker starts, it performs the following recovery sequence:

1. **Scan for Interrupted Jobs**
   - Queries Redis for jobs in `PROCESSING` state
   - These jobs were likely interrupted during processing

2. **Job Recovery Assessment**
   - Validates job metadata exists
   - Checks if original file still exists
   - Attempts to recover processing state

3. **Chunk Information Recovery**
   - Recovers chunk data from Redis events
   - Falls back to filesystem scan if needed
   - Reconstructs processing state

4. **Idempotency Check**
   - Identifies completed chunks
   - Calculates remaining work
   - Prevents duplicate processing

5. **Resume or Complete**
   - Resumes processing remaining chunks
   - Finalizes jobs with all chunks complete
   - Handles failures gracefully

### Detailed Recovery Flow

```
Job Worker Startup
├── _perform_crash_recovery()
│   ├── Scan Redis for PROCESSING jobs
│   ├── For each interrupted job:
│   │   ├── _resume_job()
│   │   │   ├── Validate job metadata
│   │   │   ├── Check file existence
│   │   │   ├── _recover_chunk_info()
│   │   │   │   ├── Try Redis events first
│   │   │   │   └── Fallback to filesystem scan
│   │   │   ├── _check_completed_chunks()
│   │   │   │   └── Scan Redis for completed chunks
│   │   │   ├── _resume_chunk_processing()
│   │   │   │   ├── Process remaining chunks
│   │   │   │   ├── Update progress
│   │   │   │   └── Publish chunk events
│   │   │   └── _finalize_recovered_job()
│   │   └── Publish recovery events
│   └── Continue normal processing
└── _worker_loop()
```

## Idempotency Mechanisms

The system implements multiple layers of idempotency to ensure consistent behavior:

### 1. Chunk-Level Idempotency (Primary)

**Mechanism:** Redis stream event tracking
- Each completed chunk publishes a `chunk_transcribed` event
- Recovery process scans events to identify completed chunks
- Only remaining chunks are processed during recovery

**Implementation:**
```python
async def _check_completed_chunks(self, job_id: str, chunk_infos: List[ChunkInfo]) -> List[int]:
    """Check which chunks have already been completed."""
    completed_chunks = []
    updates = await self.redis_service.get_job_updates(job_id, "0")
    
    for update in updates:
        if update.get("event") == "chunk_transcribed":
            chunk_index = update.get("chunk_index")
            if chunk_index is not None:
                completed_chunks.append(int(chunk_index))
    
    return sorted(list(set(completed_chunks)))
```

### 2. Job-Level Idempotency

**Mechanism:** Final result checking
- Checks if transcript already exists in Redis
- Prevents duplicate final processing
- Validates job completion state

**Implementation:**
```python
async def _finalize_recovered_job(self, job_id: str, chunk_infos: List[ChunkInfo], 
                                completed_chunks: List[int]) -> bool:
    """Finalize a job that had all chunks completed."""
    result_key = f"transcript:{job_id}"
    existing_result = await self.redis_service.redis.get(result_key)
    
    if existing_result:
        # Job already finalized
        await self.redis_service.update_job_state(job_id, JobState.COMPLETED)
        return True
    # ... continue with finalization
```

### 3. Event-Based Idempotency

**Mechanism:** Redis stream metadata tracking
- Each event includes unique identifiers
- Prevents duplicate event processing
- Maintains processing history

**Event Structure:**
```json
{
  "event": "chunk_transcribed",
  "chunk_index": 2,
  "chunk_text": "transcribed text",
  "processing_time": 3.45,
  "retry_count": 0,
  "recovered": true,
  "timestamp": 1640995200.0
}
```

### 4. State Management Idempotency

**Mechanism:** Job state validation
- Validates current job state before processing
- Prevents duplicate state transitions
- Ensures consistent job lifecycle

**State Transitions:**
```
QUEUED → PROCESSING → COMPLETED
   ↓         ↓            ↓
FAILED ← PROCESSING → FAILED
```

### 5. Progress Tracking Idempotency

**Mechanism:** Combined progress calculation
- Accounts for previously completed chunks
- Prevents double-counting progress
- Maintains accurate completion percentage

**Progress Calculation:**
```python
total_completed = len(completed_chunks) + sum(1 for r in new_results if r.success)
total_chunks = len(completed_chunks) + len(remaining_chunks)
progress = (total_completed / total_chunks) * 100
```

### 6. Storage Idempotency

**Mechanism:** Unique Redis keys with TTL
- Each result stored with unique key
- TTL prevents indefinite storage
- Consistent key naming convention

**Key Structure:**
```
transcript:{job_id}        # Final transcript
job:{job_id}              # Job metadata
job_updates:{job_id}      # Event stream
```

## Recovery Event Publishing

The system publishes detailed recovery events for monitoring and debugging:

### Event Types

1. **recovery_started** - Recovery process begins
2. **recovery_completed** - Job successfully recovered
3. **recovery_failed** - Recovery attempt failed
4. **processing_resumed** - Chunk processing resumed

### Event Structure

```json
{
  "event": "recovery_started",
  "job_id": "abc123",
  "timestamp": 1640995200.0,
  "recovery": true,
  "message": "Starting crash recovery for interrupted job",
  "recovery_type": "startup_scan"
}
```

### Recovery Event Flow

```
Recovery Process
├── recovery_started
│   ├── message: "Starting crash recovery for interrupted job"
│   ├── recovery_type: "startup_scan"
│   └── job_id: "abc123"
├── processing_resumed (if partial recovery)
│   ├── message: "Resuming processing with X remaining chunks"
│   ├── remaining_chunks: 3
│   └── completed_chunks: 5
└── recovery_completed OR recovery_failed
    ├── message: "Job successfully recovered and resumed"
    ├── recovery_result: "success"
    └── final_state: "COMPLETED"
```

## Configuration and Setup

### Environment Variables

No specific environment variables are required for crash recovery - it's enabled by default.

### Redis Configuration

Ensure Redis persistence is enabled for recovery to work:

```bash
# In redis.conf
appendonly yes
appendfsync everysec
save 900 1
save 300 10
save 60 10000
```

### Worker Configuration

The job worker automatically performs crash recovery on startup:

```python
# In main.py
async def startup_event():
    # Worker automatically performs crash recovery
    await init_job_worker(settings, redis_service)
```

## Monitoring and Debugging

### Recovery Monitoring

1. **Check Recovery Status**
   ```bash
   # Check if recovery completed
   docker-compose logs api | grep "crash recovery"
   docker-compose logs api | grep "recovery_completed"
   ```

2. **Monitor Recovery Events**
   ```bash
   # Monitor Redis streams for recovery events
   docker-compose exec redis redis-cli XREAD STREAMS job_updates:* 0
   ```

3. **Check Job States**
   ```bash
   # Find jobs in PROCESSING state
   curl "http://localhost:8000/jobs?status=PROCESSING"
   ```

### Debug Recovery Issues

1. **Inspect Job Metadata**
   ```bash
   # Check job data in Redis
   docker-compose exec redis redis-cli HGETALL "job:abc123"
   ```

2. **Review Event History**
   ```bash
   # Get job update events
   curl "http://localhost:8000/jobs/abc123/events"
   ```

3. **Check File System State**
   ```bash
   # Inspect job directory
   docker-compose exec api ls -la /tmp/media-to-text/job_abc123/
   ```

### Recovery Metrics

Monitor these metrics to track recovery performance:

- **Recovery Success Rate:** `recovered_jobs / total_interrupted_jobs`
- **Recovery Time:** Time from startup to recovery completion
- **Chunk Recovery Rate:** Successfully recovered chunks vs. total chunks
- **Job Completion Rate:** Jobs completed after recovery

## Troubleshooting

### Common Recovery Issues

#### 1. Jobs Stuck in PROCESSING State

**Symptoms:** Jobs remain in PROCESSING state after worker restart

**Diagnosis:**
```bash
# Check for PROCESSING jobs
curl "http://localhost:8000/jobs?status=PROCESSING"

# Check worker logs
docker-compose logs api | grep "recovery"
```

**Solutions:**
1. Restart the worker to trigger recovery
2. Manually reset job state if recovery fails
3. Check Redis persistence configuration

#### 2. Duplicate Chunk Processing

**Symptoms:** Same chunks processed multiple times

**Diagnosis:**
```bash
# Check Redis events for duplicate entries
docker-compose exec redis redis-cli XRANGE job_updates:abc123 - +
```

**Solutions:**
1. Verify idempotency checks are working
2. Check Redis stream integrity
3. Review chunk indexing logic

#### 3. Missing Chunk Information

**Symptoms:** Recovery fails due to no chunk information

**Diagnosis:**
```bash
# Check if chunks directory exists
docker-compose exec api ls -la /tmp/media-to-text/job_abc123/chunks/

# Check Redis events
docker-compose exec redis redis-cli XRANGE job_updates:abc123 - +
```

**Solutions:**
1. Verify Redis event publishing is working
2. Check filesystem permissions
3. Ensure proper cleanup timing

#### 4. Recovery Event Publishing Failures

**Symptoms:** No recovery events in Redis streams

**Diagnosis:**
```bash
# Check Redis connection
docker-compose exec api python -c "import redis; r=redis.Redis(host='redis'); print(r.ping())"

# Check stream existence
docker-compose exec redis redis-cli EXISTS job_updates:abc123
```

**Solutions:**
1. Verify Redis connectivity
2. Check Redis stream configuration
3. Review event publishing logic

### Recovery Debugging Commands

```bash
# Check recovery completion
docker-compose logs api | grep "recovery_completed"

# List interrupted jobs
docker-compose exec redis redis-cli KEYS "job:*" | xargs -I {} docker-compose exec redis redis-cli HGET {} state | grep PROCESSING

# Check chunk recovery
docker-compose exec redis redis-cli XRANGE job_updates:abc123 - + | grep chunk_transcribed

# Monitor recovery in real-time
docker-compose logs -f api | grep recovery
```

## Operational Procedures

### Manual Recovery Triggers

1. **Force Recovery on Startup**
   ```bash
   # Restart worker to trigger recovery
   docker-compose restart api
   ```

2. **Manual Job State Reset**
   ```bash
   # Reset job to QUEUED state
   curl -X PATCH "http://localhost:8000/jobs/abc123" \
        -H "Content-Type: application/json" \
        -d '{"state": "QUEUED"}'
   ```

3. **Clear Corrupted Job Data**
   ```bash
   # Remove job data (use with caution)
   docker-compose exec redis redis-cli DEL "job:abc123"
   docker-compose exec redis redis-cli DEL "transcript:abc123"
   ```

### Recovery Health Checks

1. **Verify Recovery Capability**
   ```bash
   # Check if recovery process is working
   docker-compose logs api | grep "Starting crash recovery scan"
   docker-compose logs api | grep "Crash recovery completed"
   ```

2. **Test Recovery Functionality**
   ```bash
   # Create test job and simulate crash
   # 1. Start transcription job
   curl -X POST "http://localhost:8000/transcriptions/" \
        -F "file=@test.mp3"
   
   # 2. Stop worker during processing
   docker-compose stop api
   
   # 3. Restart and verify recovery
   docker-compose start api
   docker-compose logs api | grep recovery
   ```

### Best Practices

1. **Monitoring Setup**
   - Monitor recovery events in production
   - Set up alerts for failed recoveries
   - Track recovery success rates

2. **Data Integrity**
   - Ensure Redis persistence is properly configured
   - Regular backups of Redis data
   - Monitor disk space for temporary files

3. **Performance Optimization**
   - Tune Redis memory settings for job storage
   - Monitor recovery time for large jobs
   - Optimize chunk size for faster recovery

4. **Operational Procedures**
   - Document recovery procedures for operations team
   - Regular testing of recovery functionality
   - Maintain runbooks for common recovery scenarios

## Advanced Recovery Scenarios

### Large File Recovery

For large files with many chunks:
- Recovery time increases with chunk count
- Memory usage scales with concurrent chunk processing
- Consider chunk size optimization

### Network Partition Recovery

During Redis connectivity issues:
- Workers wait for Redis connection
- Recovery triggers after reconnection
- Event publishing resumes automatically

### Disk Space Recovery

When temporary storage is full:
- Cleanup service removes old job directories
- Recovery checks file existence before processing
- Graceful degradation for missing files

### Multi-Worker Recovery

With multiple workers:
- Each worker performs independent recovery
- Redis locking prevents duplicate recovery attempts
- Coordination through Redis state management

---

*This documentation covers the complete crash recovery and idempotency system. For additional technical details, refer to the source code in `src/media_to_text/services/job_worker.py`.*
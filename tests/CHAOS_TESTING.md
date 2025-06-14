# Chaos Testing for Crash Recovery

This document describes the comprehensive chaos testing suite for validating crash recovery functionality in the media-to-text microservice.

## Overview

The chaos testing suite simulates various crash scenarios to ensure that the worker crash recovery system is robust, reliable, and maintains data consistency. The tests validate three critical aspects:

1. **Recovery Functionality**: Jobs are properly resumed after crashes
2. **Idempotency**: No duplicate processing occurs during recovery
3. **Event Publishing**: Recovery events are published correctly for monitoring

## Test Architecture

### ChaosTestFramework

The `ChaosTestFramework` class provides the core infrastructure for simulating crashes and validating recovery:

- **Test Job Creation**: Creates jobs with configurable file sizes
- **Crash Simulation**: Simulates worker crashes at different processing stages
- **Recovery Validation**: Validates that recovery works correctly
- **Idempotency Checking**: Ensures no duplicate chunk processing
- **Event Validation**: Verifies recovery events are published

### Test Scenarios

#### Basic Crash Recovery Tests

1. **Recovery After Transcription Crash**
   - Crashes worker during chunk transcription
   - Validates job resumes from correct state
   - Ensures no chunks are re-processed

2. **Recovery with No Previous Chunks**
   - Crashes before any chunks complete
   - Validates job restarts from beginning
   - Ensures proper initialization

3. **Recovery with All Chunks Completed**
   - Crashes after all chunks are done
   - Validates job finalization
   - Ensures no additional processing

4. **Multiple Recovery Attempts**
   - Tests multiple recovery attempts on same job
   - Validates idempotency across attempts
   - Ensures system stability

5. **Recovery Event Structure**
   - Validates event data structure
   - Ensures all required fields present
   - Validates event-specific data

6. **Crash During Different Phases**
   - Tests crashes at initialization, early/mid/late transcription
   - Ensures recovery works at all phases
   - Validates phase-specific behavior

#### Extended Chaos Scenarios

1. **Redis Connection Loss**
   - Simulates Redis failures during recovery
   - Tests resilience to infrastructure issues
   - Validates graceful degradation

2. **Concurrent Recovery Attempts**
   - Simulates multiple workers attempting recovery
   - Tests race condition handling
   - Validates concurrent safety

3. **Corrupted Chunk Files**
   - Simulates missing/corrupted chunk files
   - Tests fallback mechanisms
   - Validates graceful error handling

## Test Implementation Details

### Mock Services

The framework uses comprehensive mocking to isolate crash recovery logic:

- **Redis Service**: Mock implementation with in-memory state
- **FFmpeg Service**: Mocked media processing with configurable responses
- **OpenAI Service**: Mocked transcription with controllable failures
- **Cleanup Service**: Mocked cleanup integration

### Crash Simulation

Crashes are simulated by:
1. Starting normal job processing
2. Injecting exceptions at specific points
3. Capturing job state at crash
4. Creating new worker for recovery
5. Validating recovery behavior

### Validation Methods

Each test validates multiple aspects:

- **State Consistency**: Job state before/after recovery
- **Event Sequence**: Correct recovery events published
- **Chunk Tracking**: No duplicate chunk processing
- **Progress Accuracy**: Correct progress calculations
- **Error Handling**: Graceful failure management

## Running the Tests

### Prerequisites

Install testing dependencies:
```bash
pip install -r tests/requirements-test.txt
```

### Test Execution Options

#### Run All Tests
```bash
python tests/run_chaos_tests.py --verbose
```

#### Quick Test Suite
```bash
python tests/run_chaos_tests.py --quick
```

#### Full Comprehensive Suite
```bash
python tests/run_chaos_tests.py --full
```

#### With Coverage
```bash
python tests/run_chaos_tests.py --coverage
```

#### Parallel Execution
```bash
python tests/run_chaos_tests.py --parallel 4
```

### Direct Pytest Execution

```bash
# Run all chaos tests
pytest tests/test_chaos_recovery.py -v

# Run specific test
pytest tests/test_chaos_recovery.py::TestCrashRecovery::test_recovery_after_transcription_crash -v

# Run comprehensive suite
pytest tests/test_chaos_recovery.py::test_comprehensive_chaos_suite -v
```

## Test Scenarios Coverage

### Crash Points Tested

1. **Before Chunk Creation**: Worker crashes during initialization
2. **During Chunking**: Worker crashes while creating audio chunks
3. **Early Transcription**: Worker crashes after 1-2 chunks
4. **Mid Transcription**: Worker crashes in middle of processing
5. **Late Transcription**: Worker crashes near completion
6. **After Completion**: Worker crashes during finalization

### Recovery Scenarios

1. **Clean Recovery**: All data intact, normal resumption
2. **Partial Data Loss**: Some chunks missing, partial recovery
3. **Complete Data Loss**: No previous data, restart from beginning
4. **State Corruption**: Inconsistent state, error handling
5. **Infrastructure Failure**: Redis/filesystem issues during recovery

### Validation Checks

For each scenario, tests validate:

- ✅ Job state transitions correctly
- ✅ No duplicate chunk processing (idempotency)
- ✅ Recovery events published with correct structure
- ✅ Progress tracking remains accurate
- ✅ Final results are consistent
- ✅ Cleanup integration works properly
- ✅ Error handling is graceful

## Integration with CI/CD

The chaos tests can be integrated into CI/CD pipelines:

### GitHub Actions Example
```yaml
- name: Run Chaos Tests
  run: |
    pip install -r tests/requirements-test.txt
    python tests/run_chaos_tests.py --coverage
```

### Docker Integration
```dockerfile
# Add to Dockerfile for testing
COPY tests/ /app/tests/
RUN pip install -r tests/requirements-test.txt
RUN python tests/run_chaos_tests.py --quick
```

## Performance Considerations

### Test Execution Time

- **Quick Suite**: ~30 seconds (basic scenarios)
- **Full Suite**: ~2-3 minutes (comprehensive scenarios)
- **Individual Tests**: ~5-10 seconds each

### Resource Usage

- **Memory**: ~50MB per test (isolated environments)
- **Disk**: Temporary files cleaned up automatically
- **CPU**: Minimal (mostly I/O mocking)

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure src/ is in Python path
2. **Async Issues**: Check pytest-asyncio configuration
3. **Mock Failures**: Verify mock service setup
4. **Timeout Issues**: Increase timeout for slow systems

### Debug Mode

Run with maximum verbosity:
```bash
pytest tests/test_chaos_recovery.py -v -s --tb=long
```

### Test Isolation

Each test runs in isolation with:
- Separate temporary directories
- Independent mock services
- Automatic cleanup

## Expected Outcomes

### Success Criteria

All tests should pass, validating:

1. **100% Recovery Success Rate**: All crashes recovered successfully
2. **Zero Idempotency Violations**: No duplicate chunk processing
3. **Complete Event Coverage**: All recovery events published
4. **Consistent State Management**: Job states remain consistent
5. **Graceful Error Handling**: System handles errors without corruption

### Performance Benchmarks

- **Recovery Time**: < 5 seconds for typical jobs
- **Memory Overhead**: < 10% during recovery
- **Event Latency**: < 100ms for recovery events
- **State Consistency**: 100% across all scenarios

## Maintenance

### Adding New Scenarios

1. Create new test method in appropriate test class
2. Use `ChaosTestFramework` for setup/teardown
3. Follow existing validation patterns
4. Add to comprehensive suite if needed

### Updating Mocks

1. Keep mocks aligned with actual service interfaces
2. Update when adding new recovery features
3. Maintain backward compatibility

### Performance Tuning

1. Monitor test execution times
2. Optimize mock service responses
3. Parallelize independent test scenarios
4. Use appropriate timeouts

---

## Conclusion

The chaos testing suite provides comprehensive validation of crash recovery functionality, ensuring the media-to-text microservice maintains reliability and data consistency even in the face of unexpected failures. The tests cover all critical scenarios and provide confidence in the system's robustness.
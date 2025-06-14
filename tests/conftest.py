"""Pytest configuration for chaos testing."""

import asyncio
import pytest
import pytest_asyncio


def pytest_configure(config):
    """Configure pytest for chaos testing."""
    config.addinivalue_line(
        "markers", "chaos: mark test as chaos testing for crash recovery"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
async def cleanup_test_environment():
    """Clean up test environment after each test."""
    yield
    # Cleanup is handled by individual test frameworks
    pass
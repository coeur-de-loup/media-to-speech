#!/usr/bin/env python3
"""Test runner for chaos testing of crash recovery functionality.

This script runs comprehensive chaos tests that validate:
1. Worker crash recovery and job resumption
2. Idempotency of chunk processing
3. Recovery event publishing
4. State consistency after crashes
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


def main():
    """Run chaos testing with various options."""
    parser = argparse.ArgumentParser(description="Run chaos tests for crash recovery")
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quick',
        action='store_true', 
        help='Run quick test subset only'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Run full comprehensive test suite'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run with coverage reporting'
    )
    parser.add_argument(
        '--parallel', '-n',
        type=int,
        default=1,
        help='Number of parallel test workers'
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Add test file
    test_file = os.path.join(os.path.dirname(__file__), 'test_chaos_recovery.py')
    pytest_args.append(test_file)
    
    # Add verbosity
    if args.verbose:
        pytest_args.extend(['-v', '-s'])
    
    # Add coverage if requested
    if args.coverage:
        pytest_args.extend([
            '--cov=media_to_text',
            '--cov-report=html',
            '--cov-report=term-missing'
        ])
    
    # Add parallel execution
    if args.parallel > 1:
        pytest_args.extend(['-n', str(args.parallel)])
    
    # Test selection
    if args.quick:
        # Run only basic recovery tests
        pytest_args.extend(['-k', 'test_recovery_after_transcription_crash or test_recovery_with_no_previous_chunks'])
    elif args.full:
        # Run comprehensive suite
        pytest_args.extend(['-k', 'test_comprehensive_chaos_suite'])
    
    # Add timeout for long-running tests
    pytest_args.extend(['--timeout=300'])
    
    # Add markers
    pytest_args.extend(['-m', 'not slow or chaos'])
    
    print("🧪 Starting Chaos Testing for Crash Recovery")
    print("=" * 60)
    print(f"Running pytest with args: {' '.join(pytest_args)}")
    print("=" * 60)
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("✅ All chaos tests passed! Crash recovery is robust.")
    else:
        print("❌ Some chaos tests failed. Check output above.")
    print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
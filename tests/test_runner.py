"""
Test runner script for TriPoD unit tests.

This script can be used to run the unit tests without requiring 
the full dustpy installation.

Usage:
    python test_runner.py

The script will run all unit tests and display the results.
"""

import subprocess
import sys
import os

def main():
    """Run the unit tests."""
    print("=" * 60)
    print("Running TriPoD Unit Tests")
    print("=" * 60)
    
    # Get the directory this script is in
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the standalone tests
    cmd = [
        sys.executable, "-m", "pytest", 
        os.path.join(test_dir, "unit", "test_size_distribution_standalone.py"),
        "-v", "--tb=short"
    ]
    
    try:
        result = subprocess.run(cmd, cwd=test_dir)
        return result.returncode
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest:")
        print("pip install pytest")
        return 1

if __name__ == "__main__":
    sys.exit(main())
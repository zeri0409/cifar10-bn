"""
Compatibility wrapper for the original sample entrypoint.

This script now delegates to the full Project 2 runner so that the
loss-landscape analysis is generated together with the other experiments.
"""

from run_project2 import main


if __name__ == "__main__":
    main()

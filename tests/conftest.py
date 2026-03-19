"""Shared pytest utilities available to all test modules."""
import glob


def find_data_file() -> str:
    """Return the first JSON file found in the data/ directory.

    Raises FileNotFoundError if no data file is present, which will cause
    the calling test to fail with a clear message rather than a skip.
    Tests that should skip when data is absent should catch FileNotFoundError
    and call pytest.skip() themselves.
    """
    files = glob.glob("data/*.json")
    if not files:
        raise FileNotFoundError("No JSON files found in data/")
    return files[0]

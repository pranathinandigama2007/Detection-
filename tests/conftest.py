# tests/conftest.py
import sys
from pathlib import Path

# Force the project root directory onto sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import pytest
import asyncio
from database_module.lite_db import init_sqlite_db

@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Initializes the SQLite embedded DB for tests."""
    asyncio.run(init_sqlite_db())
    yield
    if os.path.exists("decs_local.db"):
        try:
            os.remove("decs_local.db")
        except PermissionError:
            pass
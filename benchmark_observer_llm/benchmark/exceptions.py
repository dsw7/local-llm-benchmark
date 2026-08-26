from typing import Any


class ConfigError(Exception):
    def __init__(self, message: str, *args: Any):
        self.message = message
        self.args = args

    def __str__(self) -> str:
        return f"ConfigError: {self.message}"


class BenchmarkError(Exception):
    def __init__(self, message: str, *args: Any):
        self.message = message
        self.args = args

    def __str__(self) -> str:
        return f"BenchmarkError: {self.message}"

import functools
from ollama import Client


@functools.cache
def get_client(host: str) -> Client:
    return Client(host)

from functools import cache

from ollama import Client
from requests import get

from .exceptions import BenchmarkError


@cache
def get_client(host: str) -> Client:
    return Client(host)


def check_server_up(server: str) -> None:
    get(f"http://{server}", timeout=5)


def check_model_exists(server: str, model: str) -> None:
    client = get_client(server)
    response = client.list()

    for list_model in response.models:
        if list_model.model == model:
            break
    else:
        raise BenchmarkError(
            f"Model '{model}' not found on server '{server.split(':')[0]}'"
        )


def preload_model(server: str, model: str) -> None:
    client = get_client(server)
    client.generate(model=model, prompt="What is 3 + 5?", keep_alive="30m")

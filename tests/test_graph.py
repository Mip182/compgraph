import json
import pytest

from pathlib import Path

from compgraph import Graph
from compgraph import operations as ops


def put_test_to_file(input_path: Path, test) -> Path:  # type: ignore
    with open(input_path, "w") as file:
        for val in test:
            print(json.dumps(val), file=file)
    return input_path


@pytest.fixture(scope="session")
def data_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp("data") / "input.txt"

    return put_test_to_file(
        input_path,
        [
            {"count": 1, "text": "hell"},
            {"count": 1, "text": "world"},
            {"count": 2, "text": "hello"},
            {"count": 2, "text": "my"},
            {"count": 3, "text": "little"},
        ],
    )


def test_graph_from_file(data_file: Path) -> None:
    graph = Graph.graph_from_file(data_file.as_posix(), ops.json_parser)

    result = graph.run()

    assert list(result) == [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]


def test_graph_from_iter() -> None:
    tests = [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]

    expected = [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]

    graph = Graph.graph_from_iter("test")

    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_map() -> None:
    tests = [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]

    expected = [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]

    graph = Graph.graph_from_iter("test").map(ops.Split(column="text"))

    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_reduce() -> None:
    tests = [
        {"test_id": 1, "text": "hello, world"},
        {"test_id": 1, "text": "zzzz"},
        {"test_id": 2, "text": "bye!"},
    ]

    expected = [{"test_id": 1, "text": "hello, world"}, {"test_id": 2, "text": "bye!"}]

    graph = Graph.graph_from_iter("test").reduce(ops.FirstReducer(), keys=("test_id",))

    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected


def test_graph_sort() -> None:
    tests = [
        {"test_id": 1, "text": "b"},
        {"test_id": 2, "text": "c"},
        {"test_id": 3, "text": "a"},
    ]

    expected = [
        {"test_id": 3, "text": "a"},
        {"test_id": 1, "text": "b"},
        {"test_id": 2, "text": "c"},
    ]

    graph = Graph.graph_from_iter("test").sort(["text"])

    result = graph.run(test=lambda: iter(tests))

    assert list(result) == expected

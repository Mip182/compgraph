import json

from compgraph import operations as ops

from examples import (
    run_word_count,
    run_pmi_graph,
    run_yandex_maps_graph,
    run_inverted_index_graph,
)

import pytest
from click.testing import CliRunner
from operator import itemgetter
from pathlib import Path


def put_test_to_file(input_path: Path, test) -> Path:  # type: ignore
    with open(input_path, "w") as file:
        for val in test:
            print(json.dumps(val), file=file)
    return input_path


@pytest.fixture(scope="session")
def word_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp("data") / "input.txt"

    docs = [
        {"doc_id": 1, "text": "hello, my little WORLD"},
        {"doc_id": 2, "text": "Hello, my little little hell"},
    ]

    return put_test_to_file(input_path, docs)


@pytest.fixture(scope="session")
def out_file(tmp_path_factory) -> Path:  # type: ignore
    output_path = tmp_path_factory.mktemp("data") / "output.txt"

    return output_path


def test_word_count_script(word_file: Path, out_file: Path) -> None:
    CliRunner().invoke(run_word_count.main, [word_file.as_posix(), out_file.as_posix()])

    expected = [
        {"count": 1, "text": "hell"},
        {"count": 1, "text": "world"},
        {"count": 2, "text": "hello"},
        {"count": 2, "text": "my"},
        {"count": 3, "text": "little"},
    ]

    result = ops.Read(out_file.as_posix(), ops.json_parser)()

    assert list(result) == expected


@pytest.fixture(scope="session")
def pmi_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp("data") / "input.txt"

    rows = [
        {"doc_id": 1, "text": "hello, little world"},
        {"doc_id": 2, "text": "little"},
        {"doc_id": 3, "text": "little little little"},
        {"doc_id": 4, "text": "little? hello little world"},
        {"doc_id": 5, "text": "HELLO HELLO! WORLD..."},
        {
            "doc_id": 6,
            "text": "world? world... world!!! WORLD!!! HELLO!!! HELLO!!!!!!!",
        },
    ]

    return put_test_to_file(input_path, rows)


def test_pmi_script(pmi_file: Path, out_file: Path) -> None:
    CliRunner().invoke(run_pmi_graph.main, [pmi_file.as_posix(), out_file.as_posix()])

    expected = [
        {"doc_id": 3, "text": "little", "pmi": pytest.approx(0.9555, 0.001)},
        {"doc_id": 4, "text": "little", "pmi": pytest.approx(0.9555, 0.001)},
        {"doc_id": 5, "text": "hello", "pmi": pytest.approx(1.1786, 0.001)},
        {"doc_id": 6, "text": "world", "pmi": pytest.approx(0.7731, 0.001)},
        {"doc_id": 6, "text": "hello", "pmi": pytest.approx(0.0800, 0.001)},
    ]

    result = ops.Read(out_file.as_posix(), ops.json_parser)()

    assert list(result) == expected


@pytest.fixture(scope="session")
def tf_idf_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp("data") / "input.txt"

    rows = [
        {"doc_id": 1, "text": "hello, little world"},
        {"doc_id": 2, "text": "little"},
        {"doc_id": 3, "text": "little little little"},
        {"doc_id": 4, "text": "little? hello little world"},
        {"doc_id": 5, "text": "HELLO HELLO! WORLD..."},
        {"doc_id": 6, "text": "world? world... world!!! WORLD!!! HELLO!!!"},
    ]

    return put_test_to_file(input_path, rows)


def test_tf_idf_script(tf_idf_file: Path, out_file: Path) -> None:
    CliRunner().invoke(
        run_inverted_index_graph.main, [tf_idf_file.as_posix(), out_file.as_posix()]
    )

    expected = [
        {"doc_id": 1, "text": "hello", "tf_idf": pytest.approx(0.1351, 0.001)},
        {"doc_id": 1, "text": "world", "tf_idf": pytest.approx(0.1351, 0.001)},
        {"doc_id": 2, "text": "little", "tf_idf": pytest.approx(0.4054, 0.001)},
        {"doc_id": 3, "text": "little", "tf_idf": pytest.approx(0.4054, 0.001)},
        {"doc_id": 4, "text": "hello", "tf_idf": pytest.approx(0.1013, 0.001)},
        {"doc_id": 4, "text": "little", "tf_idf": pytest.approx(0.2027, 0.001)},
        {"doc_id": 5, "text": "hello", "tf_idf": pytest.approx(0.2703, 0.001)},
        {"doc_id": 5, "text": "world", "tf_idf": pytest.approx(0.1351, 0.001)},
        {"doc_id": 6, "text": "world", "tf_idf": pytest.approx(0.3243, 0.001)},
    ]

    result = ops.Read(out_file.as_posix(), ops.json_parser)()

    assert sorted(result, key=itemgetter("doc_id", "text")) == expected


@pytest.fixture(scope="session")
def time_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp("data") / "input.txt"

    times = [
        {
            "leave_time": "20171020T112238.723000",
            "enter_time": "20171020T112237.427000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171011T145553.040000",
            "enter_time": "20171011T145551.957000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171020T090548.939000",
            "enter_time": "20171020T090547.463000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171024T144101.879000",
            "enter_time": "20171024T144059.102000",
            "edge_id": 8414926848168493057,
        },
        {
            "leave_time": "20171022T131828.330000",
            "enter_time": "20171022T131820.842000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171014T134826.836000",
            "enter_time": "20171014T134825.215000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171010T060609.897000",
            "enter_time": "20171010T060608.344000",
            "edge_id": 5342768494149337085,
        },
        {
            "leave_time": "20171027T082600.201000",
            "enter_time": "20171027T082557.571000",
            "edge_id": 5342768494149337085,
        },
    ]

    return put_test_to_file(input_path, times)


@pytest.fixture(scope="session")
def length_file(tmp_path_factory) -> Path:  # type: ignore
    input_path = tmp_path_factory.mktemp("data") / "input.txt"

    lengths = [
        {
            "start": [37.84870228730142, 55.73853974696249],
            "end": [37.8490418381989, 55.73832445777953],
            "edge_id": 8414926848168493057,
        },
        {
            "start": [37.524768467992544, 55.88785375468433],
            "end": [37.52415172755718, 55.88807155843824],
            "edge_id": 5342768494149337085,
        },
        {
            "start": [37.56963176652789, 55.846845586784184],
            "end": [37.57018438540399, 55.8469259692356],
            "edge_id": 5123042926973124604,
        },
        {
            "start": [37.41463478654623, 55.654487907886505],
            "end": [37.41442892700434, 55.654839486815035],
            "edge_id": 5726148664276615162,
        },
        {
            "start": [37.584684155881405, 55.78285809606314],
            "end": [37.58415022864938, 55.78177368734032],
            "edge_id": 451916977441439743,
        },
        {
            "start": [37.736429711803794, 55.62696328852326],
            "end": [37.736344216391444, 55.626937723718584],
            "edge_id": 7639557040160407543,
        },
        {
            "start": [37.83196756616235, 55.76662947423756],
            "end": [37.83191015012562, 55.766647034324706],
            "edge_id": 1293255682152955894,
        },
    ]

    return put_test_to_file(input_path, lengths)


def test_yandex_maps_script(time_file: Path, length_file: Path, out_file: Path) -> None:
    CliRunner().invoke(
        run_yandex_maps_graph.main,
        [time_file.as_posix(), length_file.as_posix(), out_file.as_posix()],
    )

    expected = [
        {"weekday": "Fri", "hour": 8, "speed": pytest.approx(62.2322, 0.001)},
        {"weekday": "Fri", "hour": 9, "speed": pytest.approx(78.1070, 0.001)},
        {"weekday": "Fri", "hour": 11, "speed": pytest.approx(88.9552, 0.001)},
        {"weekday": "Sat", "hour": 13, "speed": pytest.approx(100.9690, 0.001)},
        {"weekday": "Sun", "hour": 13, "speed": pytest.approx(21.8577, 0.001)},
        {"weekday": "Tue", "hour": 6, "speed": pytest.approx(105.3901, 0.001)},
        {"weekday": "Tue", "hour": 14, "speed": pytest.approx(41.5145, 0.001)},
        {"weekday": "Wed", "hour": 14, "speed": pytest.approx(106.4505, 0.001)},
    ]

    result = ops.Read(out_file.as_posix(), ops.json_parser)()

    assert sorted(result, key=itemgetter("weekday", "hour")) == expected

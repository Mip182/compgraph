import click
import json

from compgraph.algorithms import yandex_maps_graph
from compgraph.operations import json_parser


@click.command()
@click.argument("input_time_filepath", type=str)
@click.argument("input_length_filepath", type=str)
@click.argument("output_filepath", type=str)
def main(
    input_time_filepath: str, input_length_filepath: str, output_filepath: str
) -> None:
    graph = yandex_maps_graph(
        input_stream_name_time=input_time_filepath,
        input_stream_name_length=input_length_filepath,
        parser=json_parser,
    )

    result = graph.run()
    with open(output_filepath, "w") as out:
        for row in result:
            print(json.dumps(row), file=out)


if __name__ == "__main__":
    main()

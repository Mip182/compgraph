from . import Graph, operations
import typing as tp
import math


def word_count_graph(
    input_stream_name: str,
    text_column: str = "text",
    count_column: str = "count",
    parser: tp.Callable[[str], operations.TRow] | None = None,
) -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    if parser is None:
        return (
            Graph.graph_from_iter(input_stream_name)
            .map(operations.FilterPunctuation(text_column))
            .map(operations.LowerCase(text_column))
            .map(operations.Split(text_column))
            .sort([text_column])
            .reduce(operations.Count(count_column), [text_column])
            .sort([count_column, text_column])
        )
    else:
        return (
            Graph.graph_from_file(input_stream_name, parser)
            .map(operations.FilterPunctuation(text_column))
            .map(operations.LowerCase(text_column))
            .map(operations.Split(text_column))
            .sort([text_column])
            .reduce(operations.Count(count_column), [text_column])
            .sort([count_column, text_column])
        )


def inverted_index_graph(
    input_stream_name: str,
    doc_column: str = "doc_id",
    text_column: str = "text",
    result_column: str = "tf_idf",
    parser: tp.Callable[[str], operations.TRow] | None = None,
) -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    count = "count"
    doc_count = "doc_count"
    idf = "idf"
    tf = "tf"

    if parser is None:
        read_graph = Graph.graph_from_iter(input_stream_name)
    else:
        read_graph = Graph.graph_from_file(input_stream_name, parser)

    preprocess_graph = read_graph.map(operations.FilterPunctuation(text_column)).map(
        operations.LowerCase(text_column)
    )
    split_words_graph = preprocess_graph.map(operations.Split(text_column))
    count_docs_graph = (
        read_graph.sort([doc_column])
        .reduce(operations.FirstReducer(), [doc_column])
        .reduce(operations.Count(count), [])
    )
    count_idf_graph = (
        split_words_graph.sort([doc_column, text_column])
        .reduce(operations.FirstReducer(), keys=[doc_column, text_column])
        .sort([text_column])
        .reduce(operations.Count(doc_count), keys=[text_column])
        .join(operations.InnerJoiner(), count_docs_graph, keys=[])
        .map(
            operations.BinaryOperation(
                lambda row: math.log(row[count] / row[doc_count]), idf
            )
        )
    )
    tf_graph = split_words_graph.sort([doc_column]).reduce(
        operations.TermFrequency(text_column), [doc_column]
    )
    tf_idf_graph = (
        tf_graph.sort([text_column])
        .join(operations.InnerJoiner(), count_idf_graph, keys=[text_column])
        .map(operations.Product([idf, tf], result_column))
        .map(operations.Project([doc_column, text_column, result_column]))
        .reduce(operations.TopN(result_column, 3), [text_column])
    )
    return tf_idf_graph


def pmi_graph(
    input_stream_name: str,
    doc_column: str = "doc_id",
    text_column: str = "text",
    result_column: str = "pmi",
    parser: tp.Callable[[str], operations.TRow] | None = None,
) -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    doc_tf = "doc_tf"
    total_tf = "total_tf"

    if parser is None:
        read_graph = Graph.graph_from_iter(input_stream_name)
    else:
        read_graph = Graph.graph_from_file(input_stream_name, parser)
    split_graph = (
        read_graph.map(operations.FilterPunctuation(text_column))
        .map(operations.LowerCase(text_column))
        .map(operations.Split(text_column))
        .map(operations.Filter(lambda row: len(row[text_column]) > 4))
    )
    freq_graph = (
        split_graph.sort([doc_column, text_column])
        .reduce(operations.Count(doc_tf), [doc_column, text_column])
        .map(operations.Filter(lambda row: row[doc_tf] > 1))
    )
    filtered_graph = split_graph.sort([doc_column, text_column]).join(
        operations.InnerJoiner(), freq_graph, [doc_column, text_column]
    )
    doc_tf_graph = filtered_graph.reduce(
        operations.TermFrequency(text_column, doc_tf), [doc_column]
    )
    total_tf_graph = filtered_graph.reduce(
        operations.TermFrequency(text_column, total_tf), []
    )
    calc_pmi_graph = (
        doc_tf_graph.sort([text_column])
        .join(
            operations.InnerJoiner(), total_tf_graph.sort([text_column]), [text_column]
        )
        .map(
            operations.BinaryOperation(
                lambda row: math.log(row[doc_tf] / row[total_tf]), result_column
            )
        )
        .map(operations.Project([doc_column, text_column, result_column]))
        .sort([doc_column])
        .reduce(operations.TopN(result_column, 10), [doc_column])
    )

    return calc_pmi_graph


def yandex_maps_graph(
    input_stream_name_time: str,
    input_stream_name_length: str,
    enter_time_column: str = "enter_time",
    leave_time_column: str = "leave_time",
    edge_id_column: str = "edge_id",
    start_coord_column: str = "start",
    end_coord_column: str = "end",
    weekday_result_column: str = "weekday",
    hour_result_column: str = "hour",
    speed_result_column: str = "speed",
    parser: tp.Callable[[str], operations.TRow] | None = None,
) -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""
    distance_col = "distance"
    duration_column = "duration"

    if parser is None:
        graph_travel_times = Graph.graph_from_iter(input_stream_name_time)
        graph_road_graph = Graph.graph_from_iter(input_stream_name_length)
    else:
        graph_travel_times = Graph.graph_from_file(input_stream_name_time, parser)
        graph_road_graph = Graph.graph_from_file(input_stream_name_length, parser)

    graph_distance = (
        graph_road_graph.map(
            operations.RoadGraphProcessor(
                edge_id_column, start_coord_column, end_coord_column, distance_col
            )
        )
        .map(operations.Project([edge_id_column, distance_col]))
        .sort([edge_id_column])
    )
    graph_duration = (
        graph_travel_times.map(
            operations.TravelTimeProcessor(
                edge_id_column,
                enter_time_column,
                leave_time_column,
                weekday_result_column,
                hour_result_column,
                duration_column,
            )
        )
        .map(
            operations.Project(
                [
                    edge_id_column,
                    weekday_result_column,
                    hour_result_column,
                    duration_column,
                ]
            )
        )
        .sort([edge_id_column])
    )

    joint_graph = graph_duration.join(
        operations.InnerJoiner(), graph_distance, [edge_id_column]
    ).sort([weekday_result_column, hour_result_column])

    duration_graph = joint_graph.reduce(
        operations.Sum(duration_column),
        [edge_id_column, weekday_result_column, hour_result_column],
    ).sort([edge_id_column, weekday_result_column, hour_result_column])

    distance_graph = joint_graph.reduce(
        operations.Sum(distance_col),
        [edge_id_column, weekday_result_column, hour_result_column],
    ).sort([edge_id_column, weekday_result_column, hour_result_column])

    speed_graph = (
        duration_graph.join(
            operations.InnerJoiner(),
            distance_graph,
            [edge_id_column, weekday_result_column, hour_result_column],
        )
        .map(
            operations.BinaryOperation(
                lambda row: row[distance_col] / row[duration_column],
                speed_result_column,
            )
        )
        .map(
            operations.Project(
                [weekday_result_column, hour_result_column, speed_result_column]
            )
        )
    )

    return speed_graph

import typing as tp

from . import operations as ops
from . import external_sort as sort


class Graph:
    """Computational graph implementation"""

    def __init__(self, *args: tp.Any):
        self.operation: ops.Operation | None = None
        self.graphs: tp.Any = args

    @staticmethod
    def graph_from_iter(name: str) -> "Graph":
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        new_graph = Graph()
        new_graph.operation = ops.ReadIterFactory(name)
        return new_graph

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> "Graph":
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        new_graph = Graph()
        new_graph.operation = ops.Read(filename, parser)
        return new_graph

    def map(self, mapper: ops.Mapper) -> "Graph":
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        new_graph = Graph(self)
        new_graph.operation = ops.Map(mapper)
        return new_graph

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> "Graph":
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        new_graph = Graph(self)
        new_graph.operation = ops.Reduce(reducer, keys)
        return new_graph

    def sort(self, keys: tp.Sequence[str]) -> "Graph":
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        new_graph = Graph(self)
        new_graph.operation = sort.ExternalSort(keys)
        return new_graph

    def join(
        self, joiner: ops.Joiner, join_graph: "Graph", keys: tp.Sequence[str]
    ) -> "Graph":
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        new_graph = Graph(self, join_graph)
        new_graph.operation = ops.Join(joiner, keys)
        return new_graph

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        op = self.operation
        if op is None:
            raise TypeError
        if len(self.graphs) == 0:
            yield from op(**kwargs)
        elif len(self.graphs) == 1:
            yield from op(self.graphs[0].run(**kwargs))
        else:
            yield from op(self.graphs[0].run(**kwargs), self.graphs[1].run(**kwargs))

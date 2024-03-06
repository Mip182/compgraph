import math
from abc import abstractmethod, ABC
import string
import typing as tp
from datetime import datetime
from itertools import groupby
from functools import reduce
import operator
import heapq
from collections import Counter
from typing import Any
import re
import json


TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]


def json_parser(line: str) -> TRow:
    print(line)
    return json.loads(line)


class Operation(ABC):
    @abstractmethod
    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self.filename = filename
        self.parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self.filename) as f:
            for line in f:
                yield self.parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self.name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""

    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self.mapper = mapper

    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        for row in rows:
            yield from self.mapper(row)


class Reducer(ABC):
    """Base class for reducers"""

    @abstractmethod
    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self.reducer = reducer
        self.keys = keys

    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        for _, group_rows in groupby(rows, key=lambda row: [row[k] for k in self.keys]):
            yield from self.reducer(tuple(self.keys), group_rows)


class Joiner(ABC):
    """Base class for joiners"""

    def __init__(self, suffix_a: str = "_1", suffix_b: str = "_2") -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self.keys = keys
        self.joiner = joiner

    def __call__(
        self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any
    ) -> TRowsGenerator:
        yield from self.joiner(self.keys, rows, *args)


# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        filtered = row[self.column].translate(str.maketrans("", "", string.punctuation))
        row[self.column] = filtered
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""

    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self.column = column

    @staticmethod
    def _lower_case(txt: str) -> str:
        return txt.lower()

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = row[self.column].lower()
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""

    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self.column = column
        self.separator = separator if separator is not None else r"\s"

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.column not in row:
            yield row
            return

        pattern = re.compile(self.separator)
        last_end = 0
        for match in pattern.finditer(row[self.column]):
            if match.start() != 0:
                yield {
                    **row,
                    self.column: row[self.column][last_end: match.start()].strip(),
                }
            last_end = match.end()

        if last_end < len(row[self.column]):
            yield {**row, self.column: row[self.column][last_end:].strip()}


class Product(Mapper):
    """Calculates product of multiple columns"""

    def __init__(
        self, columns: tp.Sequence[str], result_column: str = "product"
    ) -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self.columns = columns
        self.result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        product = reduce(operator.mul, (row[col] for col in self.columns), 1)
        yield {**row, self.result_column: product}


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""

    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self.condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self.condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""

    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self.columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {col: row[col] for col in self.columns if col in row}


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""

    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self.column_max = column
        self.n = n

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        top_rows = heapq.nlargest(self.n, rows, key=lambda x: x[self.column_max])

        for row in top_rows:
            yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""

    def __init__(self, words_column: str, result_column: str = "tf") -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self.words_column = words_column
        self.result_column = result_column

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for group_val, group in groupby(rows, lambda x: [x[y] for y in group_key]):
            word_count = Counter(row[self.words_column] for row in group)
            total_words = sum(word_count.values())

            for word, count in word_count.items():
                tf = count / total_words
                yield {
                    **dict(zip(group_key, group_val)),
                    self.words_column: word,
                    self.result_column: tf,
                }


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self.column = column

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for key, group in groupby(rows, lambda x: [x.get(val) for val in group_key]):
            if all(key):
                yield {**dict(zip(group_key, key)), self.column: sum(1 for _ in group)}


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """

    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self.column = column

    def __call__(
        self, group_key: tuple[str, ...], rows: TRowsIterable
    ) -> TRowsGenerator:
        for group_val, group in groupby(rows, lambda x: [x[y] for y in group_key]):
            total = sum(row[self.column] for row in group)

            yield {
                **dict(zip(group_key, group_val)),
                self.column: total,
            }


# Joiners


class InnerJoiner(Joiner):
    """Join with inner strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        iter_a = iter(rows_a)
        iter_b = iter(rows_b)

        try:
            row_a = next(iter_a)
            row_b = next(iter_b)
        except StopIteration:
            return

        overlapping_columns = set(row_a.keys()).intersection(set(row_b.keys()))
        overlapping_columns.difference_update(set(keys))

        matched_rows_b: list[dict[str, Any]] = []

        while True:
            key_a = tuple(row_a[k] for k in keys)
            key_b = tuple(row_b[k] for k in keys)

            if key_a < key_b:
                if (
                    len(matched_rows_b) > 0
                    and tuple(matched_rows_b[0][k] for k in keys) == key_a
                ):
                    for row_b_matched in matched_rows_b:
                        row_a_renamed = {
                            f"{k}{self._a_suffix}" if k in overlapping_columns else k: v
                            for k, v in row_a.items()
                        }
                        row_b_renamed = {
                            f"{k}{self._b_suffix}" if k in overlapping_columns else k: v
                            for k, v in row_b_matched.items()
                        }
                        yield {**row_a_renamed, **row_b_renamed}

                try:
                    row_a = next(iter_a)
                except StopIteration:
                    return
            elif key_a >= key_b:
                if (
                    len(matched_rows_b) == 0
                    or tuple(matched_rows_b[0][k] for k in keys) == key_b
                ):
                    matched_rows_b.append(row_b)
                else:
                    matched_rows_b = [row_b]

                try:
                    row_b = next(iter_b)
                except StopIteration:
                    break

        while True:
            key_a = tuple(row_a[k] for k in keys)

            if (
                len(matched_rows_b) > 0
                and tuple(matched_rows_b[0][k] for k in keys) == key_a
            ):
                for row_b_matched in matched_rows_b:
                    row_a_renamed = {
                        f"{k}{self._a_suffix}" if k in overlapping_columns else k: v
                        for k, v in row_a.items()
                    }
                    row_b_renamed = {
                        f"{k}{self._b_suffix}" if k in overlapping_columns else k: v
                        for k, v in row_b_matched.items()
                    }
                    yield {**row_a_renamed, **row_b_renamed}
            try:
                row_a = next(iter_a)
            except StopIteration:
                return


class OuterJoiner(Joiner):
    """Join with outer strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        dict_a = {tuple(row[k] for k in keys): row for row in rows_a}
        dict_b = {tuple(row[k] for k in keys): row for row in rows_b}

        all_keys = set(dict_a.keys()) | set(dict_b.keys())

        for key in all_keys:
            row_a = dict_a.get(key, {})
            row_b = dict_b.get(key, {})

            merged_row = {
                k: row_a.get(k, row_b.get(k))
                for k in set(row_a.keys()) | set(row_b.keys())
            }
            yield merged_row


class LeftJoiner(Joiner):
    """Join with left strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        iter_a = iter(rows_a)
        iter_b = iter(rows_b)

        try:
            row_a = next(iter_a)
            row_b = next(iter_b)
        except StopIteration:
            return

        overlapping_columns = set(row_a.keys()).intersection(set(row_b.keys()))
        overlapping_columns.difference_update(set(keys))

        matched_rows_b: list[dict[str, Any]] = []

        while True:
            key_a = tuple(row_a[k] for k in keys)
            key_b = tuple(row_b[k] for k in keys)

            while key_a >= key_b:
                if key_a == key_b:
                    matched_rows_b.append(row_b)

                try:
                    row_b = next(iter_b)
                    key_b = tuple(row_b[k] for k in keys)
                except StopIteration:
                    break

            if matched_rows_b:
                for row_b_matched in matched_rows_b:
                    yield {**row_a, **row_b_matched}
                matched_rows_b = []
            else:
                yield row_a

            try:
                row_a = next(iter_a)
            except StopIteration:
                break


class RightJoiner(Joiner):
    """Join with right strategy"""

    def __call__(
        self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable
    ) -> TRowsGenerator:
        iter_a = iter(rows_a)
        iter_b = iter(rows_b)

        try:
            row_a = next(iter_a)
            row_b = next(iter_b)
        except StopIteration:
            return

        overlapping_columns = set(row_a.keys()).intersection(set(row_b.keys()))
        overlapping_columns.difference_update(set(keys))

        matched_rows_a: list[dict[str, Any]] = []
        key_a = tuple(row_a[k] for k in keys)

        while True:
            key_b = tuple(row_b[k] for k in keys)

            while key_b >= key_a:
                if key_a == key_b:
                    matched_rows_a.append(row_a)

                try:
                    row_a = next(iter_a)
                    key_a = tuple(row_a[k] for k in keys)
                except StopIteration:
                    break

            if matched_rows_a:
                for row_a_matched in matched_rows_a:
                    yield {**row_a_matched, **row_b}
                matched_rows_a = []
            else:
                yield row_b

            try:
                row_b = next(iter_b)
            except StopIteration:
                break


class BinaryOperation(Mapper):
    """Result of operation between two columns"""

    def __init__(self, operation: tp.Callable[[TRow], float], column: str) -> None:
        """
        :param operation: binary operation
        :param column: name of result column
        """
        self.operation = operation
        self.column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self.column] = self.operation(row)
        yield row


class RoadGraphProcessor(Mapper):
    """Get the haversine distance for roads"""

    def __init__(
        self, edge_id_col: str, start_col: str, end_col: str, distance_col: str
    ) -> None:
        self.edge_id_col = edge_id_col
        self.start_col = start_col
        self.end_col = end_col
        self.distance_col = distance_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        lon1, lat1 = map(math.radians, row[self.start_col])
        lon2, lat2 = map(math.radians, row[self.end_col])

        lat_sin = math.sin((lat2 - lat1) / 2) ** 2
        long_sin = math.sin((lon2 - lon1) / 2) ** 2

        angle = math.sqrt(lat_sin + math.cos(lat1) * math.cos(lat2) * long_sin)
        earth_radius = 6373
        row[self.distance_col] = 2 * earth_radius * math.asin(angle)

        yield row


class TravelTimeProcessor(Mapper):
    """Get the duration, weekday and hour from time"""

    def __init__(
        self,
        edge_id_col: str,
        enter_time_col: str,
        leave_time_col: str,
        weekday_col: str,
        hour_col: str,
        duration_col: str,
    ) -> None:
        self.edge_id_col = edge_id_col
        self.enter_time_col = enter_time_col
        self.leave_time_col = leave_time_col
        self.weekday_col = weekday_col
        self.hour_col = hour_col
        self.duration_col = duration_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        enter_time = datetime.strptime(row[self.enter_time_col], "%Y%m%dT%H%M%S.%f")
        leave_time = datetime.strptime(row[self.leave_time_col], "%Y%m%dT%H%M%S.%f")
        duration = (leave_time - enter_time).total_seconds() / 3600

        row[self.weekday_col] = enter_time.strftime("%a")
        row[self.hour_col] = enter_time.hour
        row[self.duration_col] = duration

        yield row

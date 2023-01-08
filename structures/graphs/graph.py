from typing import Optional


class Vertex:
    def __init__(self, name: str):
        self.name = name


class Edge:
    def __init__(self, bidirectional: bool, start: Vertex, end: Vertex, cost: Optional[int]):
        self.end = end
        self.cost = cost
        self.start = start
        self.bidirectional = bidirectional

    @property
    def is_loop(self):
        return self.start == self.end


class Graph:
    def __init__(self, peaks: list[Vertex], edges: list[Edge]):
        self.edges = edges
        self.peaks = peaks

    def adjacency_matrix(self) -> list[list[bool]]:
        matrix = []
        for i, _ in enumerate(self.peaks):
            matrix.append([])
            for j, _ in enumerate(self.peaks):
                matrix[i].append(0)
        for edge in self.edges:
            matrix[edge.start][edge.end] = 1
            if edge.bidirectional:
                matrix[][] = 1
        return matrix

    def breadth_first_search(self):
        pass

    def depth_first_search(self):
        pass
"""Small fallback for restricted environments without LangGraph installed."""

from collections.abc import Callable, Mapping

START = "__start__"
END = "__end__"


class _CompiledGraph:
    def __init__(self, nodes, edges, conditionals):
        self.nodes = nodes
        self.edges = edges
        self.conditionals = conditionals

    def invoke(self, state):
        current = self.edges[START]
        while current != END:
            result = self.nodes[current](state)
            if result:
                state.update(result)
            if current in self.conditionals:
                chooser, destinations = self.conditionals[current]
                current = destinations[chooser(state)]
            else:
                current = self.edges.get(current, END)
        return state


class StateGraph:
    """Subset of StateGraph needed by the orchestrator's tests."""

    def __init__(self, state_schema):
        self.nodes = {}
        self.edges = {}
        self.conditionals = {}

    def add_node(self, name: str, node: Callable):
        self.nodes[name] = node
        return self

    def add_edge(self, source: str, target: str):
        self.edges[source] = target
        return self

    def add_conditional_edges(self, source: str, path, path_map: Mapping[str, str]):
        self.conditionals[source] = (path, dict(path_map))
        return self

    def compile(self):
        return _CompiledGraph(self.nodes, self.edges, self.conditionals)

from collections import defaultdict

def build_join_graph(tables, joins):
    graph = defaultdict(set)
    for ltab, _, rtab, _ in joins:
        graph[ltab].add(rtab)
        graph[rtab].add(ltab)
    for t in tables:
        graph[t]
    return graph

def find_connected_components(graph):
    visited = set()
    components = []

    def dfs(node, comp):
        for n in graph[node]:
            if n not in visited:
                visited.add(n)
                comp.add(n)
                dfs(n, comp)

    for node in graph:
        if node not in visited:
            visited.add(node)
            comp = {node}
            dfs(node, comp)
            components.append(comp)

    return components
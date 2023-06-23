from graphviz import Digraph

def trace(root):
    """
    Trace the computation graph starting from the root node.

    Args:
        root: The root Value object representing the starting point of the graph.

    Returns:
        A tuple containing sets of nodes and edges in the graph.
    """
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges

def draw_dot(root, graph_name):
    """
    Draw the computation graph using graphviz.

    Args:
        root: The root Value object representing the starting point of the graph.
        graph_name: The name of the graph (used as the filename).

    Returns:
        A Digraph object representing the computation graph.
    """
    dot = Digraph(format="png", graph_attr={'rankdir': "RL"})

    nodes, edges = trace(root)
    for n in nodes:
        # For any value in the graph, create a rectangular ('record') node for it
        uid = str(id(n))
        dot.node(name=uid, label="%s | data %.4f | grad %.4f" % (n.label, n.data, n.grad), shape="record")
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    out_dirview = 'assets/'
    dot.render(graph_name, directory=out_dirview, view=False)
    return dot

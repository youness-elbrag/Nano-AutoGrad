
from graphviz import Digraph

def trace(root):
    nodes , edges= set(),set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for chlid in v._prev:
                edges.add((chlid,v))
                build(chlid)
    build(root)
    return nodes,edges
# filename = g1.render(filename = 'img/g1')

# pylab.savefig('filename.png')
def draw_dot(root,graph_name):
    dot=Digraph(format="png",graph_attr={'rankdir':"RL"})
    
    nodes,edges=trace(root)
    for n in nodes:
        # for any value in the graph, create a rectangular ('record') node for it
        uid=str(id(n))
        dot.node(name=uid,label="%s | data %.4f | grad %.4f"  % (n.label,n.data,n.grad),shape="record")
        if n._op:
            dot.node(name=uid + n._op,label=n._op)
            dot.edge(uid+n._op,uid)
    for n1 , n2 in edges:
        dot.edge(str(id(n1)),str(id(n2))+n2._op)
        
        #pylab.savefig()
        
    out_dirview='assets/'
    dot.render(graph_name, directory=out_dirview,view=False)
    return dot      
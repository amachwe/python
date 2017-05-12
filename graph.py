

class Graph:


    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, n1):
        self.nodes[n1] = True

    def add_and_connect_node_pair(self, n1, n2, directed=False):
        self.nodes[n1] = True
        self.nodes[n2] = True
        self.link_nodes(n1,n2,directed=directed)


    def link_nodes(self,n1,n2, directed=False):
        if self.edges.get(n1) is not None:
            self.edges[n1][n2] = True
        else:
            self.edges[n1] = {}
            self.edges[n1][n2] =  True

        if not directed:
            if self.edges.get(n2) is not None:
                self.edges[n2][n1] = True
            else:
                self.edges[n2] = {}
                self.edges[n2][n1] = True


    def get_nodes(self):
        return self.nodes

    def does_node_exist(self, n1):
        return self.nodes.get(n1) is not None

    def get_edges_from_node(self, n1):
        return self.edges[n1]


    def is_hanging_node(self,n1):
        edges = self.edges.get(n1)

        if edges is not None:
            if len(edges) is not 0:
                return False

        return True

g = Graph()

g.add_and_connect_node_pair("a","b")


g.add_node("h")

print(g.get_nodes())
print(g.is_hanging_node("h"))
print(g.does_node_exist("b"))
print(g.does_node_exist("c"))
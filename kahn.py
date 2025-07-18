## DAG EXTRACTION FROM ONTHOLOGY ##

# MATCH path = (n {name: 'deep_learning'})-[:`cso#superTopicOf`*]->(m)
# RETURN path


## DFS TRAVERSAL OF THE DAG ##

# MATCH (n {name: 'machine_learning'})
# CALL apoc.path.expandConfig(n, {
#   relationshipFilter: 'cso#superTopicOf>',
#   bfs: false
# }) YIELD path
# RETURN [node IN nodes(path) | node.name] AS TopologicalOrder

#!pip install neo4j

## KAHN ALGO FOR TOPOLOGICAL SORT ##

from neo4j import GraphDatabase
import networkx as nx

uri = "neo4j+s://3ac1623d.databases.neo4j.io"
username = "neo4j"
password = "klKPPX9dZioY_nXpH6MDPC1EvdbsGc6-To18mP0XsJ8"

driver = GraphDatabase.driver(uri, auth=(username, password))

def fetch_graph_from_neo4j(start_node_name: str) -> nx.DiGraph:
    query = """
    MATCH path = (n {name: $start_node_name})-[:`cso#superTopicOf`*]->(m)
    RETURN path
    """
    
    graph = nx.DiGraph()
    
    with driver.session() as session:
        result = session.run(query, start_node_name=start_node_name)
        
        for record in result:
            path = record["path"]
            nodes = path.nodes
            relationships = path.relationships

            for node in nodes:
                graph.add_node(node.element_id, name=node["name"])

            for rel in relationships:
                start_node = rel.start_node.element_id
                end_node = rel.end_node.element_id
                graph.add_edge(start_node, end_node)
    
    return graph

def kahn_topological_sort(graph: nx.DiGraph) -> list:
    in_degree = {node: 0 for node in graph.nodes()}
    
    for u, v in graph.edges():
        in_degree[v] += 1

    zero_in_degree_nodes = [node for node, degree in in_degree.items() if degree == 0]
    
    topological_order = []
    
    while zero_in_degree_nodes:
        current_node = zero_in_degree_nodes.pop(0)
        topological_order.append(current_node)
        
        for neighbor in list(graph.neighbors(current_node)):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree_nodes.append(neighbor)
    
    if len(topological_order) == len(graph.nodes()):
        return topological_order
    else:
        raise ValueError("The graph contains a cycle!")

if __name__ == "__main__":

    graph = fetch_graph_from_neo4j("deep_learning")
    
    try:
        sorted_nodes = kahn_topological_sort(graph)
        sorted_names = [graph.nodes[node]['name'] for node in sorted_nodes]
        
        print("Topological Order:")
        # print(" -> ".join(sorted_names))
        for s in sorted_names:
          print(s)

    except ValueError as e:
        print(e)

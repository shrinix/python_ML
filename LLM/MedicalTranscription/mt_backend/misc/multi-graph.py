from neo4j import GraphDatabase

class Neo4jGraphManager:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_node_with_label(self, label, properties):
        with self.driver.session() as session:
            session.write_transaction(self._create_node_with_label, label, properties)

    @staticmethod
    def _create_node_with_label(tx, label, properties):
        query = f"CREATE (n:{label} $properties)"
        tx.run(query, properties=properties)

    def create_node_with_property(self, properties):
        with self.driver.session() as session:
            session.write_transaction(self._create_node_with_property, properties)

    @staticmethod
    def _create_node_with_property(tx, properties):
        query = "CREATE (n $properties)"
        tx.run(query, properties=properties)

    def query_nodes_by_label(self, label):
        with self.driver.session() as session:
            result = session.read_transaction(self._query_nodes_by_label, label)
            return [record["n"] for record in result]

    @staticmethod
    def _query_nodes_by_label(tx, label):
        query = f"MATCH (n:{label}) RETURN n"
        result = tx.run(query)
        return list(result)  # Convert the result to a list within the transaction

    def query_nodes_by_property(self, key, value):
        with self.driver.session() as session:
            result = session.read_transaction(self._query_nodes_by_property, key, value)
            return [record["n"] for record in result]

    @staticmethod
    def _query_nodes_by_property(tx, key, value):
        query = f"MATCH (n) WHERE n.{key} = $value RETURN n"
        result = tx.run(query, value=value)
        return list(result)  # Convert the result to a list within the transaction

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "MyNeo4J@2024"

    graph_manager = Neo4jGraphManager(uri, user, password)

    # Create nodes with labels
    graph_manager.create_node_with_label("Graph1_Node", {"name": "Alice"})
    graph_manager.create_node_with_label("Graph2_Node", {"name": "Bob"})

    # Create nodes with properties
    graph_manager.create_node_with_property({"name": "Charlie", "graph": "Graph1"})
    graph_manager.create_node_with_property({"name": "Dave", "graph": "Graph2"})

    # Query nodes by labels
    graph1_nodes = graph_manager.query_nodes_by_label("Graph1_Node")
    graph2_nodes = graph_manager.query_nodes_by_label("Graph2_Node")
    print("Graph1 Nodes:", graph1_nodes)
    print("Graph2 Nodes:", graph2_nodes)

    # Query nodes by properties
    graph1_nodes_by_property = graph_manager.query_nodes_by_property("graph", "Graph1")
    graph2_nodes_by_property = graph_manager.query_nodes_by_property("graph", "Graph2")
    print("Graph1 Nodes by Property:", graph1_nodes_by_property)
    print("Graph2 Nodes by Property:", graph2_nodes_by_property)

    graph_manager.close()
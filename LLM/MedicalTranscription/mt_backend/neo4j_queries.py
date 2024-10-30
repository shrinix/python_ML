
def get_all_nodes(driver):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN n")
        return [record["n"] for record in result]

def get_all_relationships(driver):
    with driver.session() as session:
        result = session.run("MATCH ()-[r]->() RETURN r")
        return [record["r"] for record in result]

def get_node_labels(driver):
        with driver.session() as session:
            result = session.run("MATCH (n) RETURN DISTINCT labels(n) AS labels")
            return [record["labels"] for record in result]

def get_relationship_types(driver):
    with driver.session() as session:
        result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS type")
        return [record["type"] for record in result]

def get_node_details(driver):
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN id(n) AS id, labels(n) AS labels, n AS node")
        return [{"id": record["id"], "labels": record["labels"], "properties": dict(record["node"])} for record in result]


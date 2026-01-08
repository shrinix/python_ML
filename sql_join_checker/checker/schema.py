def load_schema(schema_json):
    return schema_json["tables"], schema_json.get("foreign_keys", [])

def is_foreign_key(left, right, fks):
    for fk in fks:
        if (left == (fk["from_table"], fk["from_column"]) and
            right == (fk["to_table"], fk["to_column"])) or \
           (right == (fk["from_table"], fk["from_column"]) and
            left == (fk["to_table"], fk["to_column"])):
            return True
    return False
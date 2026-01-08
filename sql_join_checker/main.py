import json
import importlib.util
import glob
from checker.parser import extract_tables_and_joins
from checker.schema import load_schema
from checker.schema_analysis import analyze_schema
from checker.sample_analysis import analyze_samples
from checker.confidence import aggregate_confidence
from checker.fixes import suggest_fixes, apply_suggested_fixes

def load_sample_data(file_path):
    spec = importlib.util.spec_from_file_location("sample_data", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.samples

def check_query(sql, schema_json, samples=None):
    schema_tables, fks = load_schema(schema_json)
    tables, joins = extract_tables_and_joins(sql)
    schema_findings = analyze_schema(tables, joins, schema_tables, fks)
    sample_findings = analyze_samples(joins, samples)
    findings = schema_findings + sample_findings
    return {
        "tables": list(tables),
        "joins": joins,
        "findings": findings,
        "overall_risk_confidence": aggregate_confidence(findings),
        "suggested_fixes": suggest_fixes(findings, fks)
    }

def run_config(config_file):
    with open(config_file) as f:
        config = json.load(f)

    # Load SQL from external file
    with open(config["sql_file"], "r") as f:
        sql = f.read()

    # Load schema
    with open(config["schema_file"], "r") as f:
        schema_json = json.load(f)

    # Load optional sample data
    samples = None
    if "sample_data_file" in config:
        samples = load_sample_data(config["sample_data_file"])

    # Run checker
    result = check_query(sql, schema_json, samples)

    # Generate fixed SQL
    fixed_sql = apply_suggested_fixes(sql, result["findings"], result["suggested_fixes"])
    result["fixed_sql"] = fixed_sql
    result["name"] = config.get("name", "unnamed_test")
    return result

if __name__ == "__main__":
    results = []
    for cfg_file in glob.glob("configs/*.json"):
        print(f"\n=== Running config: {cfg_file} ===")
        result = run_config(cfg_file)
        from pprint import pprint
        pprint(result)
        results.append(result)
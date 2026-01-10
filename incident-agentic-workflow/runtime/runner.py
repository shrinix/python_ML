# runtime/runner.py
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import CONTROL_MODE
from domain.models import IncidentState

def load_incidents():
    with open("data/incidents.json") as f:
        return json.load(f)

def run():
    incidents = load_incidents()

    for inc in incidents:
        print(f"\n--- Processing {inc['id']} ---")

        if CONTROL_MODE == "langgraph":
            from orchestration.langgraph_flow import build_langgraph
            app = build_langgraph()
            state = IncidentState(description=inc["description"])
            result = app.invoke(state)
            print(result)

        elif CONTROL_MODE == "autogen":
            from orchestration.autogen_flow import build_autogen
            user, classifier, diagnoser, recommender = build_autogen()
            user.initiate_chat(
                classifier,
                message=inc["description"],
                recipients=[diagnoser, recommender]
            )

if __name__ == "__main__":
    run()
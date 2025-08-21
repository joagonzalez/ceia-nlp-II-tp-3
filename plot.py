from src.agent import build_app

app = build_app()
print(app.get_graph().draw_mermaid())
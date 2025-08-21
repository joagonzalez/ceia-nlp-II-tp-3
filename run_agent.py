from src.agent import build_app, init_app
from src.vectorService import search_similar

app = init_app()

# data = search_similar("Camila dev exp", top_k=2, debug=False, ui=False)


print("=== Chat CV RAG con desambiguación ===")
session_id = "user-123"

while True:
    try:
        q = input("\nUsuario: ").strip()
        if not q or q.lower() in {"exit", "quit"}:
            break

        # Primer invoke
        s = app.invoke({
            "session_id": session_id,
            "query": q
        })

        answer = s.get("answer", "")
        trace = s.get("trace", {})

        print("\nAsistente:", answer)

        # Caso ambigüedad → repregunta
        if trace.get("need_user_input"):
            candidates = s.get("candidates", [])   
            choice = input("\nElige persona (número, nombre o ID): ").strip()
            
            s2 = app.invoke({
                "session_id": session_id,
                "query": choice,
                "disambiguation_choice": choice,
                "candidates": candidates, 
            })
            
            print("\nAsistente:", s2.get("answer", ""))

    except KeyboardInterrupt:
        print("\nSaliendo…")
        break
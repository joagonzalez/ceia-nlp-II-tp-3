"""
RAG sobre CVs con LangGraph + memoria corta por persona + Groq LLM.
"""
import json
from typing import TypedDict, List, Dict, Any, Literal

from langgraph.graph import StateGraph, END

from groq import Groq

from src.config.settings import GROQ_API_KEY
from src.config.settings import GROQ_LLM_MODEL
from src.config.settings import PINECONE_NAMESPACE
from src.config.settings import PINECONE_INDEX
from src.config.settings import PINECONE_PERSONA_INDEX

from src.vectorService import search_similar


# Umbrales
AMBIG_DELTA = 0.04
MIN_SCORE = 0.05
TOPK_RETRIEVE = 50
TOPK_CONTEXT = 4


# ========= CLIENTES =========
groq_client = Groq(api_key=GROQ_API_KEY)

# ========= MEMORIA CORTA =========
class ShortMemory:
    """Memoria corta por (session_id, persona_id). Resetea si cambia la persona."""
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self.buffers: Dict[tuple, List[Dict[str, str]]] = {}
        self.last_persona_by_session: Dict[str, str] = {}

    def get(self, session_id: str, persona_id: str) -> List[Dict[str, str]]:
        return self.buffers.get((session_id, persona_id), [])

    def append(self, session_id: str, persona_id: str, user_msg: str, assistant_msg: str):
        buf = self.buffers.setdefault((session_id, persona_id), [])
        
        buf.append({"role": "user", "content": user_msg})
        buf.append({"role": "assistant", "content": assistant_msg})

        # recortar a los últimos max_turns*2 (porque cada turno son 2 mensajes)
        if len(buf) > self.max_turns * 2:
            buf[:] = buf[-self.max_turns * 2:]

        self.last_persona_by_session[session_id] = persona_id

    def reset_if_person_changed(self, session_id: str, new_persona_id: str):
        last = self.last_persona_by_session.get(session_id)
        if last is not None and last != new_persona_id:
            # limpiar todas las memorias de esa sesión
            for key in list(self.buffers.keys()):
                if key[0] == session_id:
                    del self.buffers[key]


MEM = ShortMemory(max_turns=4)

# ========= STATE =========
class AgentState(TypedDict, total=False):
    session_id: str
    query: str
    candidates: List[Dict[str, Any]]        # {persona_id, name, score, source_name}
    persona_ids: List[str]
    chunks: List[Dict[str, Any]]
    history: List[Dict[str, str]]
    answer: str
    trace: Dict[str, Any]
    disambiguation_choice: str              # NUEVO: "1" / "2" / persona_id / nombre
    reuse_last_persona: bool
    mode: Literal["multi","single"] 

# ========= LLAMADAS A PINECONE =========
def _ensure_hits(obj):
    """Normaliza el retorno de search_similar: dict -> [dict], list -> list, None -> []."""
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        return [obj]
    return []

def pinecone_query_people(queries: List[str]) -> List[Dict[str, Any]]:
    """
    Busca personas por texto usando tu search_similar() en el namespace de personas.
    Devuelve candidatos deduplicados por persona_id, ordenados por score desc.
    Estructura de salida (por item):
    {
        "persona_id": str,
        "name": str,
        "score": float,
        "source_name": str
    }
    """
    out: List[Dict[str, Any]] = []

    for query_text in queries:
        hits = _ensure_hits(
            search_similar(
                text=query_text,
                top_k=5,
                namespace=PINECONE_NAMESPACE,
                debug=False,
                index=PINECONE_PERSONA_INDEX,
                ui=False,  # importante: así devuelve dicts con _id, _score, fields
            )
        )
        for m in hits:
            fields = m.get("fields", {}) or {}
            person_id = fields.get("person_id") or m.get("_id")
            # arma nombre completo si viene separado
            name = fields.get("canonical_name") or fields.get("name") or ""
            lastname = fields.get("lastname") or ""
            full_name = name if not lastname else f"{name} {lastname}"

            out.append({
                "persona_id": str(person_id) if person_id is not None else None,
                "name": full_name.strip(),
                "score": float(m.get("_score", 0.0)),
                "source_name": query_text,
            })

    # deduplicación por persona_id quedándote con el mejor score
    best: Dict[str, Dict[str, Any]] = {}
    for c in out:
        k = c["persona_id"]
        if not k:
            continue
        if k not in best or c["score"] > best[k]["score"]:
            best[k] = c
            
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)

def pinecone_query_cv(query_text: str, persona_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Busca chunks de CV usando search_similar() en el índice de CVs,
    filtrando server-side por person_id.
    """
    pid_list = [str(x) for x in (persona_ids or [])]
    if not pid_list:
        return []

    # Filtro server-side por uno o varios IDs
    where = {"person_id": {"$eq": pid_list[0]}} if len(pid_list) == 1 else {"person_id": {"$in": pid_list}}

    hits = _ensure_hits(
        search_similar(
            text=query_text,
            top_k=TOPK_RETRIEVE,
            namespace=PINECONE_NAMESPACE,
            debug=False,
            ui=False,             # dicts con _id, _score, fields
            index=PINECONE_INDEX, # índice de CVs
            metadata_filter=where,
        )
    )

    out: List[Dict[str, Any]] = []
    for m in hits:
        fields = m.get("fields", {}) or {}
        out.append({
            "chunk_id": fields.get("chunk_id") or m.get("_id"),
            "text": fields.get("chunk_text", ""),
            "meta": fields,
            "score": float(m.get("_score", 0.0)),
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:TOPK_RETRIEVE]

# ========= SYSTEM PROMPTS PARA DISTINTAS TAREAS =========
SYSTEM = (
    "Eres un asistente que responde SOLO con información provista en el contexto.\n"
    "- Cita fragmentos con referencias [#] y al final lista (id=... | sección/empresa si aplica).\n"
    "- Si falta información, dilo explícitamente.\n"
    "- Resume en bullets claros (Experiencia, Educación, Skills) cuando corresponda."
)

COREF_SYS = (
    "Eres un clasificador binario. Respondes SOLO 'yes' o 'no'. "
    "Decide si la pregunta del usuario parece referirse a la MISMA persona del turno anterior "
    "(si existiera una persona ya en contexto) o si pretende introducir una NUEVA persona.\n"
    "- Responde 'yes' si la consulta es anafórica/continuación (p.ej. '¿y sus últimas experiencias?', "
    "'¿dónde estudió?', '¿y su email?').\n"
    "- Responde 'no' si menciona un nombre propio explícito o sugiere cambio de persona.\n"
    "- Si no hay persona previa, responde 'no'."
)

EXTRACT_NAMES_SYS = """
Extrae todos los nombres de personas mencionados en el texto del usuario.
Responde SOLO un JSON array de strings, sin texto adicional.
Ejemplo: ["Camila", "Valentina Rodríguez"]
"""

# ========= GROQ LLM =========
def llm_chat(system: str, user: str) -> str:
    resp = groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return resp.choices[0].message.content.strip()

def decide_coref_with_llm_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")
    last_persona = MEM.last_persona_by_session.get(session_id)
    q = state["query"]

    # Si venimos de segunda vuelta de desambiguación, dejamos que siga el flujo normal:
    if state.get("disambiguation_choice") and state.get("candidates"):
        return {**state, "reuse_last_persona": False}

    # Sin persona previa → no hay follow-up
    if not last_persona:
        return {**state, "reuse_last_persona": False}

    user_msg = (
        f"Pregunta del usuario: {q}\n"
        f"Hay una persona previa ya seleccionada en contexto.\n"
        f"¿La pregunta parece referirse a esa MISMA persona? (yes/no)"
    )
    reuse = llm_yesno(COREF_SYS, user_msg)
    tr = {**state.get("trace", {}), "coref_reuse": reuse}
    return {**state, "reuse_last_persona": bool(reuse), "trace": tr}

def render_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "(sin historia)\n"
    lines = []
    for h in history[-8:]:
        role = "Usuario" if h["role"] == "user" else "Asistente"
        lines.append(f"{role}: {h['content']}")
    return "\n".join(lines) + "\n"

def build_context(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        meta = c["meta"]
        src = "/".join([x for x in [meta.get("section"), meta.get("company")] if x])
        lines.append(f"[{i}] (id={c['chunk_id']} | {src}) {c['text']}")
    return "\n\n".join(lines)

def llm_yesno(system: str, user: str) -> bool:
    """Devuelve True/False a partir de una pregunta binaria controlada."""
    resp = groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=5,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    return "yes" in text or "sí" in text or "si" in text

def extract_names_with_llm(q: str) -> list[str]:
    raw = llm_chat(EXTRACT_NAMES_SYS, q)
    try:
        names = json.loads(raw)
        print(names)
        return [n.strip() for n in names if isinstance(n, str) and n.strip()]
    except Exception:
        return []
    
# ========= NODOS =========
def classify_mode_node(state: AgentState) -> AgentState:
    q = (state["query"] or "").strip()

    # Extraer nombres con LLM (o regex si preferís)
    names = extract_names_with_llm(q)

    if len(names) >= 2:
        mode = "multi"
    else:
        mode = "single"   # fallback por defecto

    trace = {**state.get("trace", {}), "parsed_names": names}
    print("[classify_mode]", {"query": q, "mode": mode, "names": names})
    return {**state, "mode": mode, "trace": trace}

def route_by_mode(state: AgentState) -> str:
    mode = (state.get("mode") or "single").lower()
    next_node = "resolve_people_multi" if mode == "multi" else "decide_coref_with_llm"
    print("[route_by_mode]", {"mode": mode, "names": state.get("trace", {}).get("parsed_names"), "next": next_node})
    return next_node

def resolve_people_multi_node(state: AgentState) -> AgentState:
    names = state.get("trace", {}).get("parsed_names") or extract_names_with_llm(state["query"])
    persona_ids = []
    for name in names:
        hits = pinecone_query_people([name])[:1]
        if hits:
            persona_ids.append(hits[0]["persona_id"])
    trace = {**state.get("trace", {}), "multi_names_used": names, "multi_pids": persona_ids}
    return {**state, "persona_ids": persona_ids, "trace": trace}

def retrieve_cv_chunks_multi_node(state: AgentState) -> AgentState:
    pids = state.get("persona_ids", [])
    chunks = pinecone_query_cv(state["query"], pids) if pids else []
    return {**state, "chunks": chunks}

def generate_answer_multi_node(state: AgentState) -> AgentState:
    # reparto de contexto equitativo por persona
    pids = state.get("persona_ids", [])
    by_pid = {}
    for c in state.get("chunks", []):
        pid = str(c["meta"].get("person_id"))
        by_pid.setdefault(pid, []).append(c)
    k_each = max(1, TOPK_CONTEXT // max(1, len(pids) or 1))
    chosen = []
    for pid in pids:
        chosen += by_pid.get(pid, [])[:k_each]
    if len(chosen) < TOPK_CONTEXT:
        remainder = [c for pid in pids for c in by_pid.get(pid, [])[k_each:]]
        chosen += remainder[:(TOPK_CONTEXT - len(chosen))]
    context = build_context(chosen)
    prompt = (
        f"Contexto (múltiples personas):\n{context}\n\n"
        f"Pregunta: {state['query']}\n"
        f"Responde en secciones por persona (## Nombre/ID), con bullets y citas [#]."
    )
    answer = llm_chat(SYSTEM, prompt)
    return {**state, "answer": answer}

# Single
def resolve_people_node(state: AgentState) -> AgentState:
    """
    - Siempre busca candidatos en la vector DB a partir de state['query'].
    - EXCEPTO cuando venimos de una desambiguación y ya traemos 'candidates':
      en ese caso NO re-consulta y respeta el orden mostrado al usuario.
    """
    choice = (state.get("disambiguation_choice") or "").strip()

    # Si venimos del 2º paso de desambiguación y ya hay candidatos, no re-buscar.
    if choice and state.get("candidates"):
        return state

    # Caso normal: consultar índice de personas con el query actual
    q = state["query"]
    session_id = state.get("session_id", "default")
    last_persona = MEM.last_persona_by_session.get(session_id)
    
    # Reutilizar persona activa si el LLM lo marcó
    if state.get("reuse_last_persona") and last_persona:
        cands = [{
            "persona_id": last_persona,
            "name": "",
            "score": 1.0,
            "source_name": "[coref-llm]"
        }]
        return {**state, "candidates": cands}

    # Caso normal: buscar con el query actual
    candidates = pinecone_query_people([q])
    
    return {**state, "candidates": candidates}

def decide_disambiguation_node(state: AgentState) -> AgentState:
    cands = state.get("candidates", [])
    trace = {**state.get("trace", {})}

    # 1) no match confiable
    if not cands or cands[0]["score"] < MIN_SCORE:
        trace["decision"] = "no_match"
        return {**state, "persona_ids": [], "trace": trace}

    # 2) si el usuario ya eligió (segunda vuelta)
    choice = (state.get("disambiguation_choice") or "").strip()
    if choice:
        # choice puede ser "1"/"2", un persona_id o un nombre
        by_idx = None
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(cands):
                by_idx = cands[idx]["persona_id"]

        # match por persona_id
        by_id = next((c["persona_id"] for c in cands if c["persona_id"] == choice), None)
        # match por nombre
        by_name = next((c["persona_id"] for c in cands if c["name"].lower() == choice.lower()), None)

        decided = by_idx or by_id or by_name
        if decided:
            trace["decision"] = "user_selected"
            return {**state, "persona_ids": [decided], "trace": trace}
        # Si el choice no matchea, seguimos a repregunta otra vez
        trace["bad_choice"] = choice

    # 3) detectar ambigüedad top2
    if len(cands) > 1 and (cands[0]["score"] - cands[1]["score"]) < AMBIG_DELTA:
        trace["decision"] = "ambiguous_top2"
        return {**state, "trace": trace}

    # 4) caso claro
    trace["decision"] = "clear_top1"
    return {**state, "persona_ids": [cands[0]["persona_id"]], "trace": trace}

def ask_user_short_disambiguation_node(state: AgentState) -> AgentState:
    cands = state.get("candidates", [])[:3]  # top 2–3
    if not cands:
        return {**state, "answer": "No pude identificar a la persona.", "trace": {**state.get("trace", {}), "need_user_input": False}}

    lines = ["Encontré personas con nombres similares. Indicá la opción (1/2/3) o responde con el nombre exacto:"]
    for i, c in enumerate(cands, 1):
        # Podés incluir más señales (rol, empresa) si están en metadata
        lines.append(f"{i}. {c['name']}  (id={c['persona_id']}, score={c['score']:.3f})")
    lines.append("Tu elección:")
    question = "\n".join(lines)

    trace = {**state.get("trace", {}), "need_user_input": True, "disambiguation_options": [c["persona_id"] for c in cands]}
    return {**state, "answer": question, "trace": trace}

def route_after_decision(state: AgentState) -> str:
    tr = state.get("trace", {})
    if tr.get("decision") == "no_match":
        return "ask_user_short_disambiguation"  # o manejar de otra forma (p.ej. pedir nombre)
    if tr.get("decision") == "ambiguous_top2":
        return "ask_user_short_disambiguation"
    # user_selected o clear_top1 → seguimos al retriever
    return "retrieve_cv_chunks"

def retrieve_cv_chunks_node(state: AgentState) -> AgentState:
    persona_ids = state.get("persona_ids", [])
    if not persona_ids:
        return {**state, "chunks": []}
    chunks = pinecone_query_cv(state["query"], persona_ids)
    return {**state, "chunks": chunks}

def load_memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")
    persona_ids = state.get("persona_ids", [])
    if not persona_ids:
        return {**state, "history": []}
    persona_id = persona_ids[0]
    MEM.reset_if_person_changed(session_id, persona_id)
    history = MEM.get(session_id, persona_id)
    return {**state, "history": history}

def generate_answer_node(state: AgentState) -> AgentState:
    chunks = (state.get("chunks") or [])[:TOPK_CONTEXT]
    context = build_context(chunks)
    history_txt = render_history(state.get("history", []))
    user_q = state["query"]
    prompt = (
        f"Historial reciente:\n{history_txt}\n"
        f"Contexto:\n{context}\n\n"
        f"Pregunta actual: {user_q}\n"
        f"Responde con citas [#] y lista final de (id=...)."
    )
    answer = llm_chat(SYSTEM, prompt)
    return {**state, "answer": answer}

def save_memory_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "default")
    persona_ids = state.get("persona_ids", [])
    if persona_ids and state.get("answer"):
        MEM.append(session_id, persona_ids[0], state["query"], state["answer"])
    return state

# ========= GRAFO =========
def build_app():
    g = StateGraph(AgentState)

    # --- Nodos nuevos (router + multi) ---
    g.add_node("classify_mode", classify_mode_node)
    g.add_node("resolve_people_multi", resolve_people_multi_node)
    g.add_node("retrieve_cv_chunks_multi", retrieve_cv_chunks_multi_node)
    g.add_node("generate_answer_multi", generate_answer_multi_node)

    # --- Nodos existentes (single/stateful) ---
    g.add_node("decide_coref_with_llm", decide_coref_with_llm_node)
    g.add_node("resolve_people", resolve_people_node)
    g.add_node("decide_disambiguation", decide_disambiguation_node)
    g.add_node("ask_user_short_disambiguation", ask_user_short_disambiguation_node)
    g.add_node("retrieve_cv_chunks", retrieve_cv_chunks_node)
    g.add_node("load_memory", load_memory_node)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("save_memory", save_memory_node)

    # Entry: router de modo
    g.set_entry_point("classify_mode")

    # Router condicional a multi o single
    g.add_conditional_edges(
        "classify_mode",
        route_by_mode,
        {
            "resolve_people_multi": "resolve_people_multi",   # multi (stateless)
            "decide_coref_with_llm": "decide_coref_with_llm", # single (stateful)
        },
    )

    # --- Camino MULTI (stateless) ---
    g.add_edge("resolve_people_multi", "retrieve_cv_chunks_multi")
    g.add_edge("retrieve_cv_chunks_multi", "generate_answer_multi")
    g.add_edge("generate_answer_multi", END)

    # --- Camino SINGLE (stateful) — tu flujo actual ---
    g.add_edge("decide_coref_with_llm", "resolve_people")
    g.add_edge("resolve_people", "decide_disambiguation")

    g.add_conditional_edges(
        "decide_disambiguation",
        route_after_decision,
        {
            "ask_user_short_disambiguation": "ask_user_short_disambiguation",
            "retrieve_cv_chunks": "retrieve_cv_chunks",
        },
    )

    g.add_edge("ask_user_short_disambiguation", END)
    g.add_edge("retrieve_cv_chunks", "load_memory")
    g.add_edge("load_memory", "generate_answer")
    g.add_edge("generate_answer", "save_memory")
    g.add_edge("save_memory", END)

    app = g.compile()
    return app

app = None

# se implementa singletone
def init_app():
    global app
    if app is None:
        app = build_app()
    return app


if __name__ == "__main__":
    pass
    

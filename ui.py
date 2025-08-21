import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context

from src.agent import init_app  # LangGraph app
AGENT = init_app()
SESSION_ID = "dash-ui"

CHAT_TITLE = "Asistente para análisis de Curriculums - CEIA NLP II - TP3"
EXAMPLE_MESSAGES = [
    "¿Cuales son los datos personales de Valentina?",
    "¿Y sus ultimas 2 experiencias laborales?",
    "¿En que tecnologias de especializa Valentina?",
]

# ================= Helpers =================
def graph_invoke(query: str, disamb_choice: str | None = None, candidates=None):
    payload = {"session_id": SESSION_ID, "query": query}
    if disamb_choice:
        payload["disambiguation_choice"] = disamb_choice
    if candidates:
        payload["candidates"] = candidates
    s = AGENT.invoke(payload)
    return (
        s.get("answer", ""),
        s.get("trace", {}) or {},
        s.get("candidates", []) or [],
        s.get("chunks", []) or [],
    )

# ================= Dash App =================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)

app.layout = dbc.Container([
    # Estado mínimo para desambiguación
    dcc.Store(id="graph-store", data={"awaiting_choice": False, "candidates": []}),

    dbc.Row([
        dbc.Col([
            html.H1(CHAT_TITLE, className="text-center mb-4"),
            html.Hr(),

            # Historial
            html.H5("Historial de conversación:", className="mb-3"),
            html.Div(
                id="chat-history",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.Strong("Asistente: ", className="text-success"),
                            dcc.Markdown(
                                "¡Hola! Soy tu asistente para consultas sobre el CV. "
                                "Podés preguntarme sobre experiencia, habilidades o educación."
                            )
                        ])
                    ], className="mb-2 border-success")
                ],
                style={
                    "height": "400px",
                    "overflow-y": "auto",
                    "border": "1px solid #dee2e6",
                    "border-radius": "0.375rem",
                    "padding": "1rem",
                    "background-color": "#f8f9fa",
                    "margin-bottom": "2rem",
                }
            ),

            # Loading indicator
            dcc.Loading(
                id="chat-loading",
                type="circle",
                color="#007bff",
                children=html.Div(id="chat-loading-output"),
                style={"margin": "1rem 0"},
            ),

            # Ejemplos
            html.H5("Ejemplos de preguntas:", className="mb-3"),
            html.Div([
                dbc.Button(
                    example,
                    id=f"example-btn-{i}",
                    color="outline-primary",
                    size="sm",
                    className="me-2 mb-2",
                    n_clicks=0
                ) for i, example in enumerate(EXAMPLE_MESSAGES)
            ], className="mb-4"),

            html.Hr(),

            # Input + Enviar
            dbc.InputGroup([
                dbc.Input(
                    id="chat-input",
                    placeholder="Escribe tu pregunta aquí...",
                    type="text",
                    value="",
                    disabled=False
                ),
                dbc.Button(
                    "Enviar",
                    id="send-button",
                    color="primary",
                    n_clicks=0,
                    disabled=False
                )
            ], className="mb-4"),

            # Loading general
            dcc.Loading(id="loading", type="default", children=html.Div(id="loading-output")),
        ], width=8, className="mx-auto")
    ])
], fluid=True, style={"maxWidth": "none"})

# Auto‑scroll
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var el = document.getElementById('chat-history');
            if (el) el.scrollTop = el.scrollHeight;
        }, 100);
        return '';
    }
    """,
    Output('loading-output', 'children'),
    Input('chat-history', 'children')
)

# Botones de ejemplo → rellenan input
@app.callback(
    Output("chat-input", "value"),
    [Input(f"example-btn-{i}", "n_clicks") for i in range(len(EXAMPLE_MESSAGES))],
    prevent_initial_call=True
)
def update_input_from_examples(*clicks):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    bid = ctx.triggered[0]["prop_id"].split(".")[0]
    if "example-btn-" in bid:
        idx = int(bid.split("-")[-1])
        return EXAMPLE_MESSAGES[idx]
    return ""

# Chat (único callback)
@app.callback(
    [Output("chat-history", "children"),
     Output("chat-input", "value", allow_duplicate=True),
     Output("send-button", "disabled"),
     Output("chat-input", "disabled"),
     Output("graph-store", "data")],
    [Input("send-button", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("chat-history", "children"),
     State("graph-store", "data")],
    prevent_initial_call=True
)
def update_chat(send_clicks, input_submit, user_message, current_history, store):
    if not user_message or user_message.strip() == "":
        return current_history, "", False, False, store

    user_message = user_message.strip()
    new_history = current_history.copy()

    # Card del usuario
    new_history.append(
        dbc.Card([
            dbc.CardBody([
                html.Strong("Tú: ", className="text-primary"),
                dcc.Markdown(user_message)
            ])
        ], className="mb-2 border-primary")
    )

    new_history.append(
        dbc.Card([
            dbc.CardBody([
                html.Strong("Asistente: ", className="text-success"),
                html.Div([
                    dbc.Spinner(size="sm", color="primary"),
                    html.Span(" Generando respuesta...", className="ms-2")
                ], className="d-flex align-items-center")
            ])
        ], className="mb-2 border-success", id="loading-message")
    )

    try:
        # === Segunda vuelta (desambiguación) ===
        if store.get("awaiting_choice"):
            choice = user_message  # número que escribió el user
            answer, trace, _cands, _chunks = graph_invoke(
                query=choice,
                disamb_choice=choice,
                candidates=store.get("candidates", [])
            )

            # Quitar loading
            new_history = [msg for msg in new_history if not (hasattr(msg, 'id') and msg.id == "loading-message")]

            # Card del asistente
            new_history.append(
                dbc.Card([
                    dbc.CardBody([
                        html.Strong("Asistente: ", className="text-success"),
                        dcc.Markdown(answer)
                    ])
                ], className="mb-2 border-success")
            )

            # limpiar estado y re‑habilitar input
            return new_history, "", False, False, {"awaiting_choice": False, "candidates": []}

        # === Primera vuelta normal ===
        answer, trace, candidates, _chunks = graph_invoke(user_message)

        # Quitar loading
        new_history = [msg for msg in new_history if not (hasattr(msg, 'id') and msg.id == "loading-message")]

        if trace.get("need_user_input"):
            # Mostrar repregunta + opciones (texto), y pedir número
            options = "\n".join(
                [f"{i+1}. {c['name']}  (id={c['persona_id']})" for i, c in enumerate(candidates)]
            )
            new_history.append(
                dbc.Card([
                    dbc.CardBody([
                        html.Strong("Asistente: ", className="text-success"),
                        dcc.Markdown(f"{answer}\n\n{options}\n\n*Escribe el número elegido y presiona Enviar.*")
                    ])
                ], className="mb-2 border-success")
            )
            # Guardar candidatos y esperar número
            return new_history, "", False, False, {"awaiting_choice": True, "candidates": candidates}

        # Respuesta directa
        new_history.append(
            dbc.Card([
                dbc.CardBody([
                    html.Strong("Asistente: ", className="text-success"),
                    dcc.Markdown(answer)
                ])
            ], className="mb-2 border-success")
        )
        return new_history, "", False, False, {"awaiting_choice": False, "candidates": []}

    except Exception as e:
        # Quitar loading y mostrar error
        new_history = [msg for msg in new_history if not (hasattr(msg, 'id') and msg.id == "loading-message")]
        new_history.append(
            dbc.Card([
                dbc.CardBody([
                    html.Strong("Error: ", className="text-danger"),
                    dcc.Markdown(f"Ocurrió un error: `{e}`")
                ])
            ], className="mb-2 border-danger")
        )
        return new_history, "", False, False, store

# Normaliza Enter
@app.callback(
    Output("chat-input", "n_submit"),
    Input("chat-input", "n_submit"),
    prevent_initial_call=True
)
def handle_enter(n_submit):
    return 0

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

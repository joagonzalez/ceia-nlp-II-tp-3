import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback_context, MATCH, ALL
from src.chatService import session


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

CHAT_TITLE = "Asistente para análisis de Curriculums - CEIA NLP II"
# Example messages for demonstration
EXAMPLE_MESSAGES = [
    "¿Cuáles son las habilidades técnicas de Martín?",
    "¿Dónde trabajó Martín anteriormente?",
    "¿Qué certificaciones tiene Martín?"
]

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(CHAT_TITLE, className="text-center mb-4"),
            html.Hr(),
            
            # Chat history section (moved to top)
            html.H5("Historial de conversación:", className="mb-3"),
            html.Div(
                id="chat-history",
                children=[
                    dbc.Card([
                        dbc.CardBody([
                            html.Strong("Asistente: ", className="text-success"),
                            dcc.Markdown("¡Hola! Soy tu asistente para consultas sobre el CV. Puedes preguntarme sobre la experiencia, habilidades o cualquier información del currículum.")
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
                    "margin-bottom": "2rem"
                }
            ),
            
            # Loading indicator for chat responses
            dcc.Loading(
                id="chat-loading",
                type="circle",
                color="#007bff",
                children=html.Div(id="chat-loading-output"),
                style={"margin": "1rem 0"}
            ),
            
            # Example messages section (moved below chat)
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
            
            # Chat input section (at bottom)
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
            
            # Loading indicator for general purposes
            dcc.Loading(
                id="loading",
                type="default",
                children=html.Div(id="loading-output")
            )
        ], width=8, className="mx-auto")  # 70% width centered using mx-auto class
    ])
], fluid=True, style={"maxWidth": "none"})

# JavaScript for auto-scroll
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            var chatContainer = document.getElementById('chat-history');
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        }, 100);
        return '';
    }
    """,
    Output('loading-output', 'children'),
    Input('chat-history', 'children')
)

# Callback for example buttons
@app.callback(
    Output("chat-input", "value"),
    [Input(f"example-btn-{i}", "n_clicks") for i in range(len(EXAMPLE_MESSAGES))],
    prevent_initial_call=True
)
def update_input_from_examples(*clicks):
    ctx = callback_context
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if "example-btn-" in button_id:
        idx = int(button_id.split("-")[-1])
        return EXAMPLE_MESSAGES[idx]
    
    return ""

# Callback for chat functionality
@app.callback(
    [Output("chat-history", "children"),
     Output("chat-input", "value", allow_duplicate=True),
     Output("send-button", "disabled"),
     Output("chat-input", "disabled")],
    [Input("send-button", "n_clicks"),
     Input("chat-input", "n_submit")],
    [State("chat-input", "value"),
     State("chat-history", "children")],
    prevent_initial_call=True
)
def update_chat(send_clicks, input_submit, user_message, current_history):
    if not user_message or user_message.strip() == "":
        return current_history, "", False, False
    
    # Add user message to history
    new_history = current_history.copy()
    new_history.append(
        dbc.Card([
            dbc.CardBody([
                html.Strong("Tú: ", className="text-primary"),
                dcc.Markdown(user_message)
            ])
        ], className="mb-2 border-primary")
    )
    
    # Add loading message
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
        # Get context from RAG system
        from src.vectorService import search_similar
        context = search_similar(user_message, top_k=3, debug=False)
        
        # Get response from chat service
        response = session.chat(user_message)
        
        # Remove loading message
        new_history = [msg for msg in new_history if not (hasattr(msg, 'id') and msg.id == "loading-message")]
        
        # Generate unique ID for this response
        unique_id = len(new_history)
        
        # Add assistant response to history with Markdown support
        new_history.append(
            html.Div([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Strong("Asistente: ", className="text-success"),
                            dbc.Button(
                                "👁️ Ver contexto",
                                id={"type": "context-btn", "index": unique_id},
                                color="info",
                                size="sm",
                                className="float-end",
                                n_clicks=0
                            )
                        ], className="d-flex justify-content-between align-items-center mb-2"),
                        dcc.Markdown(
                            response,
                            dangerously_allow_html=False,
                            style={"margin": "0"}
                        )
                    ])
                ], className="mb-2 border-success"),
                html.Div([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("📄 Contexto RAG utilizado:", className="text-info mb-2"),
                            html.Div([
                                html.Small(ctx_item, className="d-block mb-1 text-muted")
                                for ctx_item in context
                            ])
                        ])
                    ], className="border-info", style={"background-color": "#f0f8ff"})
                ], id={"type": "context-collapse", "index": unique_id}, style={"display": "none"})
            ])
        )
        
    except Exception as e:
        # Remove loading message if exists
        new_history = [msg for msg in new_history if not (hasattr(msg, 'id') and msg.id == "loading-message")]
        
        # Add error message to history
        new_history.append(
            dbc.Card([
                dbc.CardBody([
                    html.Strong("Error: ", className="text-danger"),
                    dcc.Markdown(f"Ocurrió un error: {str(e)}")
                ])
            ], className="mb-2 border-danger")
        )
    
    return new_history, "", False, False

# Callback for context toggle buttons
@app.callback(
    Output({"type": "context-collapse", "index": MATCH}, "style"),
    Input({"type": "context-btn", "index": MATCH}, "n_clicks"),
    State({"type": "context-collapse", "index": MATCH}, "style"),
    prevent_initial_call=True
)
def toggle_context(n_clicks, current_style):
    if n_clicks and n_clicks > 0:
        if current_style.get("display") == "none":
            return {"display": "block"}
        else:
            return {"display": "none"}
    return current_style

# Callback for Enter key press
@app.callback(
    Output("chat-input", "n_submit"),
    Input("chat-input", "n_submit"),
    prevent_initial_call=True
)
def handle_enter(n_submit):
    return 0

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
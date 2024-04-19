import dash
import utils.clients as clients
from dash.dependencies import Input, Output, State
import dash_mantine_components as dmc
from dash import Dash, dcc, html, Input, Output, State, callback

app = dash.Dash(
    __name__,
    external_stylesheets=['styles.css'],
    title="Simplificación de Texto",
    update_title="Cargando...",
    prevent_initial_callbacks=True,
)


# Components

def get_header():
    title = dmc.Title(
        "Herramienta de Simplificación de Texto",
        id="header",
        order=1,
        variant='gradient',
        mt='20px',
        mb='10vh',
        fz='3rem',
    )
    container = dmc.Container(
        fluid=True, 
        children=[title],
        
        )

    return container


def get_text_input():
    phrase_input = dmc.TextInput(
        id="phrase_input",
        label="Frase a Simplificar",
        placeholder="Escribe o pega el texto que deseas simplificar",
        required=True,
        size='lg',
        radius='lg',
    )

    complex_word_input = dmc.TextInput(
        id="complex_word_input",
        label="Palabras a Simplificar",
        placeholder="Escribe la palabra que deseas simplificar",
        size='lg',
        radius='lg',
    )

    simplify_button = dmc.Button(
        id="simplify_button",
        children="Simplificar",
        color="blue",
        size="lg",
        variant="outline",
    )

    container = dmc.Center(
        id="text_input_container",
        children=[
            phrase_input,
            complex_word_input,
            simplify_button
        ],
    )

    return container
    

def get_text_output(text):
    title = dmc.Title(
        'Tu Texto simplificado es:',
        variant='gradient',
        mb='5vh'
    )
    output_text = dmc.Paper(
        children=[
            dmc.Text(
                text,
                size='lg',
            ),
        ],
        radius='lg',
        p='xl',
        shadow='md',
        withBorder=True,
    )
    
    back_button = dmc.Button(
        id="back_button",
        children="Volver",
        color="blue",
        size="lg",
        variant="outline",
        w='20%'
    )
    
    container = dmc.Container(
        id="text_output_container",
        children=[title, output_text, back_button],
        styles={
            'display': 'flex',
            'flex-direction': 'column',
            'align-items': 'center',
        }
    )
    
    return container


# Async Functions for model calls

async def get_simplified_text(phrase, word):
    client_manager = clients.ClientManager(['ChatGPT', 'Gemini-1.0-pro'])
    

# Callbacks

@app.callback(
    Output("text_input_container", "children", allow_duplicate=True),
    Input("simplify_button", "n_clicks"),
    State("phrase_input", "value"),
    State("complex_word_input", "value"),
    running=[(Output("simplify_button", "disabled"), True, False)],
    allow_duplicate=True,
) 
def get_simplified_text(n_clicks, phrase, word):
    print(f"Phrase: {phrase}")
    print(f"Word: {word}")
    manager = clients.ClientManager(['ChatGPT', 'Gemini-1.0-pro'])
    candidates = manager.get_response(phrase, word)
    print(candidates)
    candidates = clients.parse_outputs(candidates)
    
    candidates = clients.candidate_lists_to_dict(candidates)
    
    # select candidate key with highest value
    complex_word = max(candidates, key=candidates.get)
    print(complex_word)
    simplified_text = phrase.replace(word, complex_word)
    
    output = get_text_output(simplified_text)
    
    return output
    

@app.callback(
    Output("text_input_container", "children", allow_duplicate=True),
    Output('back_button', 'n_clicks'),
    Input("back_button", "n_clicks"),
    allow_duplicate=True,
)
def go_back(n_clicks):
    if n_clicks > 0:
        return get_text_input(), 0
    
    
# Layout


app.layout = dmc.MantineProvider(
    children=[
        dmc.Container(fluid=True, children=[
            get_header(),
            get_text_input()
        ],
            styles={
                'display': 'flex',
                'flex-direction': 'column',
                'align-items': 'center',
            }
        )
    ]
)


if __name__ == "__main__":
    app.run_server(debug=False)

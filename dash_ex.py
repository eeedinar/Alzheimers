import dash
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import plotly.io as pio
import pandas as pd

df = pd.read_csv("precious_metals_prices_2018_2021.csv")  # , usecols=["DateTime" , "Gold"]

print (pio.templates)

fig = px.line(df,x="DateTime", 
                 y=["Gold"],
                 color_discrete_map={"Gold":"gold"})
fig.update_layout(xaxis_title="DateTime", 
                  yaxis_title="Price", 
                  title="CSV",
                  font=dict(family="Verdana, sans-serif", size=10, color="green"),
                  template = "ggplot2")


app = Dash(__name__)
app.title = "CSV File"

app.layout = html.Div(
        id = 'app-container', style = {"backgroundColor":"white"},

        children = [html.Div(
                    id = 'header-area', style = {"backgroundColor":"blue"},
                    children = [
                        html.H1(id = "header-title",      
                                children="CSV Headline",        
                                style={"color": "white", "fontFamily": "Verdana, sans-serif"}),
                        
                        html.P( id = "header-description",
                                children= ("The cost of precious metals", html.Br(), "between 2018 and 2021") ,
                                style={"color": "white", "fontFamily": "Verdana, sans-serif"}
                                ),
                        
                        "Input: ", dcc.Input(id='my-input', value='initial value',  type='text'),
                        
                        html.Div(
                            id="graph-container",
                            children=dcc.Graph(
                                            id="price-chart",
                                            figure=fig,
                                            config={"displayModeBar": False}),
                                ),
                    ],
                ),
                    html.Div(
                    id = "menu-area",
                    children=[html.Div(
                                children=[html.Div(
                                            className='menu-title',
                                            children="Metal"
                                            ),

                                          dcc.Dropdown(
                                            id="metal-filter",
                                            className="dropdown",
                                            options= [{"label":metal, "value":metal} for metal in df.columns[1:] ],
                                            value="Gold",
                                            clearable=False,
                                            ),

                                            ]
                                            )
                            ],
                    ),

                    html.Br(),
                    html.Div(id='my-output'),
                    ]
    )

@app.callback(
    Output(component_id='my-output', component_property='children'),
    Output(component_id="price-chart",  component_property='figure'),

    Input (component_id='my-input', component_property='value'    ),
    Input (component_id="metal-filter", component_property='value'    ),

    )

# def update_output_div(input_value):
#     return 'Output: {}'.format(input_value)

def update_output(input_value, metal):
    fig = px.line(df,x="DateTime", 
                 y=[metal],
                 
                 color_discrete_map={
                    "Platinum": "#E5E4E2",
                    "Gold": "gold",
                    "Silver": "silver",
                    "Palladium": "#CED0DD",
                    "Rhodium": "#E2E7E1",
                    "Iridium": "#3D3C3A",
                    "Ruthenium": "#C9CBC8"
                 }

                 )
    fig.update_layout(xaxis_title="DateTime", 
                      yaxis_title="Price", 
                      title="CSV",
                      font=dict(family="Verdana, sans-serif", size=10, color="green"),
                      template = "ggplot2") 
    return 'Output: {}'.format(input_value), fig

if __name__=='__main__':
    app.run_server(debug = True)
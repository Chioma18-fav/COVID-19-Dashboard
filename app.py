import dash
from dash import dcc, html
from charts import fig_recoverate, fig_tcasesbyrecov, fig_contable, fig_impact, fig_totalcasesdeaths, fig_scatterplot
# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("COVID Dashboard", style={'textAlign': 'center', 'color': 'yellow', 'backgroundColor': 'darkblue'}),

    html.Div([
        html.Div([dcc.Graph(figure=fig_recoverate)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig_tcasesbyrecov)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig_contable)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig_impact)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig_totalcasesdeaths)], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([dcc.Graph(figure=fig_scatterplot)], style={'width': '48%', 'display': 'inline-block'}),
    ])
], style={'backgroundColor': '#001f4d'})  # dark blue background

# Run app
if __name__ == "__main__":
    app.run(debug=True)

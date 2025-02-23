import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import portfolio_optimizer  # Import the data processing module

# Get the processed data
data = portfolio_optimizer.get_final_results()

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Tab 1', value='tab-1'),
        dcc.Tab(label='Tab 2', value='tab-2'),
        dcc.Tab(label='Tab 3', value='tab-3'),
        dcc.Tab(label='Tab 4', value='tab-4'),
    ]),
    html.Div(id='tabs-content-example')
])

# Define the callback to update the content based on the selected tab
@app.callback(Output('tabs-content-example', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        fig = px.line(data, x='Date', y='Value1', title='Graph 1')
        return dcc.Graph(figure=fig)
    elif tab == 'tab-2':
        fig = px.line(data, x='Date', y='Value2', title='Graph 2')
        return dcc.Graph(figure=fig)
    elif tab == 'tab-3':
        fig = px.line(data, x='Date', y='Value3', title='Graph 3')
        return dcc.Graph(figure=fig)
    elif tab == 'tab-4':
        fig = px.line(data, x='Date', y='Value4', title='Graph 4')
        return dcc.Graph(figure=fig)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
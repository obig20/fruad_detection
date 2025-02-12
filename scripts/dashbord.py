import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    dcc.Graph(id='fraud-trends'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

@app.callback(Output('fraud-trends', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    response = requests.get('http://localhost:5000/fraud_data')
    data = response.json()
    figure = {
        'data': [{'x': [1, 2, 3], 'y': [data['fraud_cases'], 10, 15], 'type': 'line', 'name': 'Fraud Cases'}],
        'layout': {'title': 'Fraud Trends Over Time'}
    }
    return figure

if __name__ == '__main__':
    app.run_server(debug=True)
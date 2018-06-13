import os

from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

#import dash
#import dash_core_components as dcc
#import dash_html_components as html
#import grasia_dash_components as gdc

# from humantree import HumanTree

#app = dash.Dash(__name__)
#server = app.server
#
#app.scripts.append_script(
#    {"external_url": "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.7.0"})
#app.css.append_css(
#    {"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})
#
#app.layout = gdc.Import(src='predict.js')
# app.layout = html.Div([
#     html.H2('Hello World'),
#     dcc.Dropdown(
#         id='dropdown',
#         options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
#         value='LA'
#     ),
#     html.Div(id='display-value'),
# ])


# @app.callback(dash.dependencies.Output('display-value', 'children'),
#               [dash.dependencies.Input('dropdown', 'value')])
# def display_value(value):
#     return 'You have selected "{}"'.format(value)


if __name__ == '__main__':
    # ht = HumanTree()
    app.run_server(debug=True)

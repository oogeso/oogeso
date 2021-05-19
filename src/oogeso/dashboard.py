import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import glob
import os
import flask
import base64

# image_directory = '/Users/hsven/code/hybrid_energy_system/'
image_directory = ""

list_of_images = [
    os.path.basename(x) for x in glob.glob("{}*.png".format(image_directory))
]
static_image_route = "/static/"

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

image_filename = "pydotCombined.png"
encoded_image = base64.b64encode(open(image_filename, "rb").read())
# required to remove the "b" from the string
# b'iVBORw0KGg...' -> 'iVBORw0KGg...'
encoded_image = encoded_image.decode()
print(encoded_image[:10])

app.layout = html.Div(
    children=[
        html.H1(children="Energy system dashboard"),
        html.Div(
            children="""
        Dash: A web application framework for Python.
    """
        ),
        html.Div(
            children=[
                html.Label("Dropdown"),
                dcc.Dropdown(
                    options=[
                        {"label": "Method 1", "value": "m1"},
                        {"label": "Method 2", "value": "m2"},
                        {"label": "Method 3", "value": "m3"},
                    ],
                    value="m1   ",
                ),
                html.Div(children="Text"),
            ],
            style={"columnCount": 2},
        ),
        dcc.Graph(
            id="example-graph",
            figure={
                "data": [
                    {"x": [1, 2, 3], "y": [4, 1, 2], "type": "bar", "name": "SF"},
                    {
                        "x": [1, 2, 3],
                        "y": [2, 4, 5],
                        "type": "bar",
                        "name": u"Montréal",
                    },
                ],
                "layout": {"title": "Dash Data Visualization"},
            },
        ),
        # TODO: Make image zoomable (or create image from )
        html.Img(
            src="data:image/png;base64,{}".format(encoded_image), style={"width": "80%"}
        ),
        #    dcc.Dropdown(
        #        id='image-dropdown',
        #        options=[{'label': i, 'value': i} for i in list_of_images],
        #        value=list_of_images[0]
        #    ),
        #    html.Img(id='image')
    ]
)

# @app.callback(
#    dash.dependencies.Output('image', 'src'),
#    [dash.dependencies.Input('image-dropdown', 'value')])
# def update_image_src(value):
#    return static_image_route + value

# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server
@app.server.route("{}<image_path>.png".format(static_image_route))
def serve_image(image_path):
    image_name = "{}.png".format(image_path)
    if image_name not in list_of_images:
        raise Exception(
            '"{}" is excluded from the allowed static files'.format(image_path)
        )
    return flask.send_from_directory(image_directory, image_name)


if __name__ == "__main__":
    app.run_server(debug=True)

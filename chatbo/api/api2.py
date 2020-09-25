import flask
from flask import request, jsonify
import os
import json 

app = flask.Flask(__name__)

# Create some test data for our catalog in the form of a list of dictionaries.
resultats = [
    {'name': 9106,
    'index': ['ret', 'stdev', 'sharpe', 'FB', 'AMZ', 'AAPL', 'MMM', 'ABT'],
    'data': [0.2643968013,
    0.2159467564,
    1.2243610676,
    0.0156893366,
    8.87571e-05,
    0.4907666983,
    0.0961937155,
    0.3972614925]
    }
]


@app.route('/', methods=['GET'])
def home():
    return '''<h1>Flask is running !! </h1>'''

# Oumayma katli houni rodha POST
@app.route('/api/resultat',  methods=['POST', 'GET'])
def api_all():
    return jsonify(resultats)


@app.route('/api/v1/resources/books', methods=['GET'])
def api_id():
    # Check if an ID was provided as part of the URL.
    # If ID is provided, assign it to a variable.
    # If no ID is provided, display an error in the browser.
    if 'id' in request.args:
        id = int(request.args['id'])
    else:
        return "Error: No id field provided. Please specify an id."

    # Create an empty list for our results
    results = []

    # Loop through the data and match results that fit the requested ID.
    # IDs are unique, but other fields might return many results
    for book in resultats:
        if book['id'] == id:
            results.append(book)

    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
        return jsonify(results)

#Get data(Parameters from nodejs)  
@app.route('/test', methods=['POST','GET'])
def test():
    request_data = request.get_json()
    print('AAAAAAA'+str(request_data))
    return jsonify(
        message = request_data
    )

if __name__ == '__main__':
    app.run(port=5001) 
 
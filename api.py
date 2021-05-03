# -*- coding: utf-8 -*-
"""Flask API

Endpoints:
    /suggestions (List[int], POST):  Estimates and returns the library_ids following the three library_id sequence.
        request example: curl -X POST -H "Content-Type: application/json" -d '{"library_ids": [2,3,4]}' localhost:8888/suggestions

"""

import os

from flask import Flask, Blueprint, request, abort, jsonify


app = Flask(__name__)


@app.route('/suggestions', methods=['POST'])
def suggestions():
    payload = request.json
    library_ids = payload.get('library_ids')
    response = [{ "id": library_ids[0], "score": 0.7}, { "id": library_ids[1], "score": 0.2}, { "id": library_ids[2], "score": 0.1}]

    return jsonify(response), 201


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
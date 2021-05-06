# -*- coding: utf-8 -*-
"""Flask API

Endpoints:
    /suggestions (List[int], POST):  Estimates and returns the library_ids following the three library_id sequence.
        request example: curl -X POST -H "Content-Type: application/json" -d '{"library_ids": [2,3,4]}' localhost:8888/suggestions

"""

import os

from flask import Flask, Blueprint, request, abort, jsonify
import torch
from omegaconf import OmegaConf

from models.networks.simple_lstm import SimpleLSTM

app = Flask(__name__)

cfg = OmegaConf.load('./configs/project/navict.yaml')
cfg.model.initial_ckpt = "./model.pth"
model = SimpleLSTM(cfg)  

@app.route('/suggestions', methods=['POST'])
def suggestions():
    payload = request.json
    library_ids = payload.get('library_ids')
    inputs = torch.tensor([library_ids])
    inputs = inputs.to(model.device)
    outputs = model.network(inputs)
    print(outputs[0].tolist())

    outputs = outputs[0].tolist()
    scores = [ {"id": idx, "score": outputs[idx]} for idx in range(len(outputs))]
    response = sorted(scores, key=lambda x: x['score'], reverse=True)

    return jsonify(response), 201


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, host='0.0.0.0', port=port)
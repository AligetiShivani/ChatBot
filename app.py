from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
import os
import sys
import numpy as np
import pandas as pd
import pickle

from utils import *

from flask_cors import CORS



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# constants
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
device = "cpu"
model_name = 'cb_model'
attn_model = 'dot'

datafile = "formatted_lines.txt"
save_dir = os.path.join(".")
voc, pairs = loadPrepareData("qa", "kcc", datafile, save_dir)

# Load model
loadFilename = "1500_checkpoint.tar"
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

encoder = encoder.to(device)
decoder = decoder.to(device)

encoder.eval()
decoder.eval()

searcher = GreedySearchDecoder(encoder, decoder)



@app.route('/predict', methods=['POST','GET'])
def upload(): 
    if request.method == "POST":
        query = request.json['fin']
        print(query)
        response = evaluateInput(encoder, decoder, searcher, voc)
        return {'response' : response}
    return None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import math
import json
import tensorflow as tf

#local
import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


def model_predict(img_path):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath('__file__')),'models')
    print(checkpoint_path)
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary("./data/word_counts.txt")

  filenames = []
  for file_pattern in img_path.split(","):
    filenames.extend(tf.gfile.Glob(file_pattern))
  tf.logging.info("Running caption generation on %d files matching %s",
                  len(filenames), img_path)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    preds = ''
    out_data = []
    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      captions = generator.beam_search(sess, image)
      print("Captions for image %s:" % os.path.basename(filename))
      for i, caption in enumerate(captions):
        # Ignore begin and end words.
        sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
        sentence = " ".join(sentence)
        preds = str(i+1) + ") " + sentence + "(p=" + str(round(math.exp(caption.logprob),6)) + ")"
        out_data.append(preds)
        print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
    out_json = json.dumps(out_data)
    #print(out_json)
    return out_json


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])               # Convert to string
        return preds
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    print("please begin operate...")
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

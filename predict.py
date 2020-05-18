import os
import pandas as pd
import sys
from absl import logging, app, flags
import numpy as np

import pickle


from source import mapping, rev_mapping

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", "data/sample_data.csv", "path to data file")
flags.DEFINE_string("model_path", "model/full_model.pkl", "path to trained model")
flags.DEFINE_float("threshold", None, "Model threshold")
flags.DEFINE_string("label_name", None, "Label column")
flags.DEFINE_string("output", "predictions.csv", "output file")

def load_data(data_path="data/sample_data.csv", map_names=True):
    dat = pd.read_csv(data_path, index_col=0)
    if map_names:
        dat = dat.rename(columns=mapping)
    return dat

def load_model(model_path="model/full_model.pkl"):
    with open(model_path, "rb") as p:
        model = pickle.load(p)
    return model

def predict(data, model, label_name=None, thr=None):
    threshold = thr if thr is not None else model["threshold"]
    
    logging.info(f"Using threshold: {threshold}")
    
    logging.info("Preprocessing data")
    Xdata = data[model["features"]].rename(columns=rev_mapping)
    Xdata['Lymp/Neut'] = Xdata["Absolute Lymphocyte Count"] / Xdata["Absolute Neut Count"]

    # Scale input
    X = pd.DataFrame(
        model["scaler"].transform(Xdata), 
        columns=Xdata.columns).fillna(0)    
    
    logging.info("Predicting...")
    # Get preds
    prob = model["clf"].predict(X)
    pred = [1 if p >= threshold else 0 for p in prob]
    
    results = pd.DataFrame({"Probability": prob, "Prediction": pred}, index=X.index)
    if label_name in data:
        logging.info("Appending labels")
        results = results.merge(data[[label_name]], 
                                left_index=True, right_index=True, how="left"
                               ).rename(columns={label_name: "Actual"})
    return results

def main(_):
    logging.info(f"Loading data from {FLAGS.data_path}")
    data = load_data(FLAGS.data_path)
    logging.info(f"Loading model from {FLAGS.model_path}")
    model = load_model(FLAGS.model_path)
    
    
    logging.info(f"Data and model loaded.")
    results = predict(data, model, label_name=FLAGS.label_name, thr=FLAGS.threshold)
    
    print(results.head(10))
    
    
    if FLAGS.output is not None:
        logging.info(f"Writing to {FLAGS.output}")
        results.to_csv(FLAGS.output)
    
    logging.info("Done")
    
if __name__ == "__main__":
    app.run(main)
    
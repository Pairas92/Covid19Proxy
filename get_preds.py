import os
import pandas as pd
import sys
from absl import logging, app, flags
import numpy as np
from source.UclaDataLoader import UclaDataLoader
from source.CedarsDataLoader import CedarsDataLoader
from source.CombineData import CombineData
from source.SplitData import SplitData
from source.TrainModel import TrainModel
from source.TestModel import TestModel
from source import CombinedClassifier
import source.viz_utils

import pickle

# need to fix the parameters for this
def load_data(ddr_path="../data_freeze", cedars_path="../cedars"):
# ucla files
    logging.info("Loading UCLA data")
    ucla_labs = os.path.join(ddr_path,'covid_labs.rpt')
    ucla_encounters = os.path.join(ddr_path,'covid_encounters.rpt')
    ucla_patients = os.path.join(ddr_path,'covid_patients.rpt')

    # cedars files
    cedars_patients = os.path.join(cedars_path, "UCLA Patient File.csv") # <path_to_cedars_patients_file>
    cedars_flat_file = os.path.join(cedars_path, "UCLA Flat File Modified.csv") # <path_to_cedars_flat_file>

    # Read data #
    dataSets = ['ucla', 'cedars']
    dataDict = {}

    randomSeed=10 #1, 5, 28
    exculdeSupps = False

    filterBy = []#, 'Ferritin', 'D-DIMER', 'Lactate Dehydrogenase']
    dropCols = []#'D-DIMER', 'Ferritin', 'Lactate Dehydrogenase', 'C-Reactive Protein']
    anyInfMarker = True
    
    
    dataDict['ucla'] = {}
    dataDict['ucla']['data'], dataDict['ucla']['trainIDs'], dataDict['ucla']['testIDs'] =\
        UclaDataLoader(ucla_labs, ucla_encounters, ucla_patients, filterBy=filterBy, anyInfMarker=anyInfMarker)
    dataDict['cedars'] = {}
    dataDict['cedars']['data'], dataDict['cedars']['trainIDs'], dataDict['cedars']['testIDs'] =\
        CedarsDataLoader(cedars_patients, cedars_flat_file, filterBy=filterBy, anyInfMarker=anyInfMarker)
    
    allData, trainIDs, testIDs = CombineData(dataDict)
    
    # Dropping features, if necessary #
    dropCols = ['D-DIMER']#, 'Ferritin', 'Lactate Dehydrogenase', 'C-Reactive Protein']
    allData = allData.drop(dropCols, axis=1)
    
    return allData, trainIDs, testIDs

def load_model(model_path="model/model.p"):
    with open(model_path, "rb") as p:
        model = pickle.load(p)
    return model

def predict(X, model, thr=None):
    
    threshold = thr if thr is not None else model[2]
    
    # Scale input
    X = pd.DataFrame(model[1].transform(X), columns=X.columns, index=X.index)
    
    # Get preds
    prob = model[0].predict(X)
    pred = [1 if p >= threshold else 0 for p in prob]
    return prob, pred

def main(_):
    
    allData, trainIDs, testIDs = load_data()

    randSt=np.random.RandomState(randomSeed)

    # Split data #
    trainData, valData, testData = SplitData(
        allData, trainIDs, testIDs, n_val=30, n_test=60, \
        random_state=randSt.randint(1,100), \
        balanced=False)
    
    
if __name__ == "__main__":
    app.run(main)
    
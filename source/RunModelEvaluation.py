import os
import pandas as pd
import sys
from UclaDataLoader import UclaDataLoader
from CedarsDataLoader import CedarsDataLoader
from CombineData import CombineData
from SplitData import SplitData
from TrainModel import TrainModel
from TestModel import TestModel
#from VisualizeResults import VisualizeResults

#for nice output - probably should be removed later
import warnings
warnings.filterwarnings("ignore")

# ucla files
ucla_labs = <path_ucla_labs_file>
ucla_encounters = <path_ucla_encounters_file>
ucla_patients = <path_ucla_patients_file>

# cedars files
cedars_patients = <path_cedars_patients_file>
cedars_flat_file = <path_cedars_flat_file>

# Read data #
dataSets = ['ucla', 'cedars']
dataDict = {}

print('Loading UCLA data...')
dataDict['ucla'] = {}
dataDict['ucla']['data'], dataDict['ucla']['trainIDs'], dataDict['ucla']['testIDs'] = UclaDataLoader(ucla_labs, ucla_encounters, ucla_patients)
print('Done')
print('Loading Cedars data...')
dataDict['cedars'] = {}
dataDict['cedars']['data'], dataDict['cedars']['trainIDs'], dataDict['cedars']['testIDs'] = CedarsDataLoader(cedars_patients, cedars_flat_file)
print('Done')

# Combine data #
allData, trainIDs, testIDs = CombineData(dataDict)
print('Datasets combined')

# Split data #
trainData, valData, testData = SplitData(allData, trainIDs, testIDs)
print('Train, Validation, Test sets separated.')

# Train and tune model #
print('Training model...')
model = TrainModel(trainData, valData)
print('Model trained.')

# Test model #
print('Evaluating on test set...')
results = TestModel(model, testData)
print('Done')

print('Precision:')
print(results['prcurve']['precision'])
print('Recall:')
print(results['prcurve']['recall'])
print('AUC:')
print(results['roc']['auc'])

# Get visualized results #
#resultGallery = visualizeResults(results)
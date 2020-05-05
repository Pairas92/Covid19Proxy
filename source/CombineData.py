import pandas as pd

def CombineData(dataDict):
    # Make all encounter ID-s unique, then combine datasets and labelsets
    alldata = None
    trainIDs = []
    testIDs = []
    dbCnt = 1
    for db in dataDict:
        dataDict[db]['data']['PatientEncounterCSNID'] = dataDict[db]['data']['PatientEncounterCSNID'].apply(lambda x: x*10+dbCnt)
        dataDict[db]['trainIDs'] = [10*x+dbCnt for x in dataDict[db]['trainIDs']]
        dataDict[db]['testIDs'] = [10*x+dbCnt for x in dataDict[db]['testIDs']]
        dataDict[db]['dbNum'] = dbCnt
        if alldata is None:
            alldata = dataDict[db]['data']
        else:
            alldata = pd.concat([alldata,dataDict[db]['data']])
        trainIDs = trainIDs + list(dataDict[db]['trainIDs'])
        testIDs = testIDs + list(dataDict[db]['testIDs'])
        dbCnt += 1
        print(str(db)+' samples: '+str(len(dataDict[db]['data'])))

    lowpriority_features = [
        # Demographics
        "PatientBMI",
        "PatientHeight",
        "PatientTemperature",
        # Vitals
        "PatientPulse", "BPSystolic", "BPDiastolic", #Respirations
        # CBC
        "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
        "Red Cell Distribution Width-CV", "Red Cell Distribution Width-SD",
        "Mean Platelet Volume", "Leukocyte",
        # Inflammatory markers
        "Interleukin-6,Highly Sensitive", "Hgb A1c - HPLC", "Procalcitonin",
        "unknown",
        #remove for testing
        #"D-DIMER",
        #"Lactate Dehydrogenase",
        #"C-Reactive Protein",
        #"Ferritin"
    ]
    alldata=alldata.drop(lowpriority_features,axis=1)
    
    return alldata, trainIDs, testIDs
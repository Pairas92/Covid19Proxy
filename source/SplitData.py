import pandas as pd
import numpy as np

def SplitData(alldata, trainIDs, testIDs, n_val=30, n_test=30, random_state=7, balanced=True):
    randSt = np.random.RandomState(random_state)
    
    n_test_pos = n_test
    if balanced==False:
        n_test_pos = int(np.round(n_test/2))
        
    test_positives_random = alldata['PatientEncounterCSNID'].loc[\
                            (alldata['PatientEncounterCSNID'].isin(testIDs)) & \
                            (alldata['positive']==1)].sample(n=n_test_pos,random_state=randSt.randint(1,100))
    val_positives_random = alldata['PatientEncounterCSNID'].loc[\
                            (alldata['PatientEncounterCSNID'].isin(testIDs)) & \
                            ~(alldata['PatientEncounterCSNID'].isin(test_positives_random)) & \
                            (alldata['positive']==1)].sample(n=n_val,random_state=randSt.randint(1,100))
    test_negatives_random = alldata['PatientEncounterCSNID'].loc[\
                            (alldata['PatientEncounterCSNID'].isin(testIDs)) & \
                            (alldata['positive']==0)].sample(n=n_test,random_state=randSt.randint(1,100))
    val_negatives_random = alldata['PatientEncounterCSNID'].loc[\
                            (alldata['PatientEncounterCSNID'].isin(testIDs)) & \
                            ~(alldata['PatientEncounterCSNID'].isin(test_negatives_random)) & \
                            (alldata['positive']==0)].sample(n=n_val,random_state=randSt.randint(1,100))


    alldata_train = alldata.loc[\
                            ~(alldata['PatientEncounterCSNID'].isin(test_negatives_random)) & \
                            ~(alldata['PatientEncounterCSNID'].isin(test_positives_random)) & \
                            ~(alldata['PatientEncounterCSNID'].isin(val_negatives_random)) & \
                            ~(alldata['PatientEncounterCSNID'].isin(val_positives_random)) \
                            ]

    alldata_test = alldata.loc[\
                            (alldata['PatientEncounterCSNID'].isin(test_negatives_random)) | \
                            (alldata['PatientEncounterCSNID'].isin(test_positives_random)) \
                            ]
    alldata_val = alldata.loc[\
                            (alldata['PatientEncounterCSNID'].isin(val_negatives_random)) | \
                            (alldata['PatientEncounterCSNID'].isin(val_positives_random)) \
                            ]

    return alldata_train, alldata_val, alldata_test
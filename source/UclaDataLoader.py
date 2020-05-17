import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import tqdm
from datetime import date
from datetime import timedelta 
from datetime import datetime as dt


def UclaDataLoader(labs_path=None, encounters_path=None, patients_path=None, filterBy=[], anyInfMarker=False):
    if labs_path is None:
        labs_path = <path_ucla_labs_file>
    if encounters_path is None:
        encounters_path = <path_ucla_encounters_file>
    if patients_path is None:
        patients_path = <path_ucla_patients_file>

    df = pd.read_csv(labs_path, delimiter="$", error_bad_lines=False)
    df['OrderDateTime'] = pd.to_datetime(df['OrderDateTime'])
    df['OrderDateTime'] = df['OrderDateTime'].dt.to_pydatetime()
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['OrderDate'] = df['OrderDate'].dt.to_pydatetime()


    dfe = pd.read_csv(encounters_path, delimiter="$", error_bad_lines=False)
    dfe['EncounterEffectiveDate'] = pd.to_datetime(dfe['EncounterEffectiveDate'])
    dfe['EncounterEffectiveDate'] = dfe['EncounterEffectiveDate'].dt.to_pydatetime()
    dfp = pd.read_csv(patients_path, delimiter="$", error_bad_lines=False)
    dfp['BirthDate'] = pd.to_datetime(dfp['BirthDate'])
    dfp['BirthDate'] = dfp['BirthDate'].dt.to_pydatetime()

    ### Get vitals

    import re
    enc_vitals = dfe[['PatientEncounterCSNID', 'PatientBMI', 'PatientTemperature', 'PatientPulse', 'BPSystolic',
                      'BPDiastolic', 'PatientHeight']].set_index('PatientEncounterCSNID')

    r = re.compile(r"([0-9]+)' ([0-9]*\.?[0-9]+)\"")
    def get_inches(el):
        m = r.match(el)
        if m == None:
            return float('NaN')
        else:
            return int(m.group(1))*12 + float(m.group(2))
        
    enc_vitals['PatientHeight'] = enc_vitals['PatientHeight'].apply((lambda x: get_inches(str(x))))

    ### Create dataframe of encounter result labels

    df_res = dfe[['PatientID', 'PatientEncounterCSNID', 'TestResult']]

    ### Create dataframe of features


    pd.options.display.float_format = '{:,.10f}'.format

    dfp['Age'] = dfp['BirthDate'].apply(lambda x: date.today().year-x.year)

    #### Get minimal EncounterEffectiveDate for patients

    #Select first EncounterEffectiveDate for patients

    pid_mine = {}

    for pid, group in tqdm.tqdm(dfe.groupby('PatientID')):
        pid_mine[pid] = group['EncounterEffectiveDate'].min()
        
    #### Keep lab results only from this first encounter

    dfe['MinEffectiveDate'] = dfe['PatientID'].apply((lambda x: pid_mine[x] if x in pid_mine else np.nan))
    first_eids = dfe['PatientEncounterCSNID'].loc[dfe['MinEffectiveDate']==dfe['EncounterEffectiveDate']].values

    df_first = df.loc[df['PatientEncounterCSNID'].isin(first_eids)]

    ###################################
    # Keep only rows with test result #
    ###################################

    df_first = df_first.loc[~df_first['TestResult'].isna()]
    #print(len(df_first))

    #######################################
    # Get second encounters for patients! #
    #######################################

    #eids of non-first encounters of patients:
    non_first_eids = dfe['PatientEncounterCSNID'].loc[dfe['MinEffectiveDate']<dfe['EncounterEffectiveDate']].values
    dfe2 = dfe.loc[dfe['MinEffectiveDate']<dfe['EncounterEffectiveDate']]
    #lab results for these encounters:
    df_second = df.loc[df['PatientEncounterCSNID'].isin(non_first_eids)]
    #len(df_second) 47895
    pid_2mine = {}

    for pid, group in tqdm.tqdm(dfe2.groupby('PatientID')):
        pid_2mine[pid] = group['EncounterEffectiveDate'].min()
        
    dfe2['MinEffectiveDate'] = dfe2['PatientID']\
        .apply((lambda x: pid_2mine[x] if x in pid_2mine else np.nan))
    second_eids = dfe2['PatientEncounterCSNID'].loc[dfe2['MinEffectiveDate']==dfe2['EncounterEffectiveDate']].values

    df_second = df.loc[df['PatientEncounterCSNID'].isin(second_eids)]

    df_second = df_second.loc[~df_second['TestResult'].isna()]
    #print(len(df_second))

    #Encounter IDs from eids_first of patients who will appear a second time as well
    multi_eids = df_first['PatientEncounterCSNID'].loc[df_first['PatientID'].isin(df_second['PatientID'])]
    #len(multi_eids.unique())

    #print(len(df_first['PatientID'].unique()))
    #print(len(dfe['PatientID'].unique()))
    #print(len(df_second['PatientID'].unique()))
    #print(len(dfe2['PatientID'].unique()))

    pid_mineDF = dfe[['PatientEncounterCSNID', 'MinEffectiveDate', 'PatientID']].set_index('PatientEncounterCSNID')
    pid_2mineDF = dfe2[['PatientEncounterCSNID', 'MinEffectiveDate', 'PatientID']].set_index('PatientEncounterCSNID')


    df = pd.concat([df_first, df_second])

    #Get encounter ID-s where certain tests were ordered

    eids_ferritin = df['PatientEncounterCSNID'].loc[df['ComponentName']=="Ferritin"].unique()
    eids_cbc = df['PatientEncounterCSNID'].loc[df['OrderDescription']=="CBC"].unique()
    eids_crp = df['PatientEncounterCSNID'].loc[df['ComponentName']=="C-Reactive Protein"].unique()
    eids_dimer = df['PatientEncounterCSNID'].loc[df['ComponentName']=="D-DIMER"].unique() # >10000 cases!!!

    #print(len(eids_ferritin)) #1st: 338
    #print(len(eids_cbc)) #1st: 957
    #print(len(eids_crp)) #1st: 350
    #print(len(eids_dimer)) #1st: 284
    #print(len(set(eids_ferritin).intersection(set(eids_cbc)).intersection(set(eids_crp)).intersection(set(eids_dimer)))) #1st: 99
    #print(len(set(eids_ferritin).intersection(set(eids_cbc)).intersection(set(eids_dimer)))) #1st: 126

    #### Define different feature sets

    highpriority_features = [
        # Demographics
        #"Age", "Male", (added later)
        # Vitals
        #"PatientTemperature", (added later)
        # CBC
        "Hemoglobin", "Leukocyte", 
        "Absolute Lymphocyte Count", "Absolute Neut Count", #"Lymp/Neut", (added later)
        "PLATELET COUNT, AUTO", 
        # Inflammatory
        "C-Reactive Protein", "Ferritin", "D-DIMER"
    ]

    mediumpriority_features = [
        # Vitals
        # O2 saturation
        # CBC
        "Absolute Baso Count", "Absolute Eos Count", "Absolute Mono Count",
        # Inflammatory
        "Procalcitonin",
        "Lactate Dehydrogenase",
        "Red Blood Cell Count"
    ]

    lowpriority_features = [
        # Demographics
        #"PatientBMI", (added later) 
        # Vitals
        #"PatientPulse", "BPSystolic", "BPDiastolic", #Respirations (added later)
        # CBC
        "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
        "Red Cell Distribution Width-CV", "Red Cell Distribution Width-SD",
        "Mean Platelet Volume", 
        # Inflammatory markers
        "Interleukin-6,Highly Sensitive", "Hgb A1c - HPLC"
    ]


    feature_list = highpriority_features + mediumpriority_features + lowpriority_features

    print('Number of features: {}'.format(len(feature_list)))

    #### USE ENRICHED FOR OUR METHOD
    component_names = feature_list

    #### Get date of first COVID test for every encounter

    test_dates = {}
    for eid, group in df[['PatientEncounterCSNID','OrderDateTime', 'OrderDate']]\
    .loc[(df['OrderDescription'].str.contains('COVID', na=False))].groupby('PatientEncounterCSNID'):
        test_dates[eid] = group['OrderDate'].min()

    def get_test_date(x):
        try:
            return test_dates[x]
        except:
            return pd.Series(dt.fromtimestamp(0)).dt.to_pydatetime()[0]

    #print(test_dates)
    df['TestDate'] = df['PatientEncounterCSNID'].apply(lambda x: get_test_date(x))

    #get Age and Sex for every encounter
    patientdata = {}
    dfp2 = dfp[['PatientEncounterCSNID', 'Age', 'Sex']]


    features = {}
    df_groups = df[['PatientEncounterCSNID','OrderDateTime', 'ComponentName', 'OrderResultValue']]\
        .loc[(df['ComponentID']>=3000000) & (df['ComponentID']<=5000000) \
        & ~(df['OrderDescription'].str.contains('COVID', na=False)) & \
             ((df['OrderDate']==df['TestDate']) | (df['OrderDate']==df['TestDate']+timedelta(days=1)))]\
        .groupby('PatientEncounterCSNID')
     
     
    for eid, group in tqdm.tqdm(df_groups):
        #print(eid)
        features[eid]={}
        idx = group.groupby('ComponentName')['OrderDateTime'].transform(min) == group['OrderDateTime']
        first_vals = group[['ComponentName','OrderResultValue']][idx]
        for cn in component_names:
            try:
                v = first_vals['OrderResultValue'].loc[first_vals['ComponentName']==cn]
                vl = len(v)
                if vl>0:
                    #print(v.values[0])
                    features[eid][cn]=v.values[0]
            except:
                print("Unexpected error:", sys.exc_info()[0])
            
    #print(len(features))

    #### Cleaning data

    #features[90062155861.0]
    num_encounters = len(features)
    #print([features[x]['C-Reactive Protein'] for x in features if 'C-Reactive Protein' in features[x]])

    for eid in features:
        if 'C-Reactive Protein' in features[eid] and features[eid]['C-Reactive Protein'] == '<0.3':
            features[eid]['C-Reactive Protein'] = 0.2
        if 'D-DIMER' in features[eid] and features[eid]['D-DIMER'] == '>10000':
            features[eid]['D-DIMER'] = 11000
        if 'Procalcitonin' in features[eid] and features[eid]['Procalcitonin'] == '<0.10':
            features[eid]['Procalcitonin'] = 0.05
        if 'Procalcitonin' in features[eid] and features[eid]['Procalcitonin'] == '>115.00':
            features[eid]['Procalcitonin'] = 120.00
            
    #print([features[x]['C-Reactive Protein'] for x in features if 'C-Reactive Protein' in features[x]])

    #### Constructing the main DataFrame

    component_names_to_use = component_names

    dataset = {'PatientEncounterCSNID' : []}
    for cn in component_names_to_use:
        dataset[cn]=[]
        
    for eid in features:
        dataset['PatientEncounterCSNID'].append(eid)
        for cn in component_names_to_use:
            if cn in features[eid]:
                dataset[cn].append(features[eid][cn])
            else:
                dataset[cn].append(np.nan)
                
    datasetDF = pd.DataFrame(data=dataset)

    ############################
    # Adding Lymp/Neut feature #
    ############################

    datasetDF['Lymp/Neut'] = pd.to_numeric(datasetDF['Absolute Lymphocyte Count'])/pd.to_numeric(datasetDF['Absolute Neut Count'])
    #datasetDF

    #### Filtering alldata to have a "high-information-content" dataset

    alldata = pd.merge(datasetDF,df_res, on='PatientEncounterCSNID')
    alldata = pd.merge(alldata,dfp2, on='PatientEncounterCSNID')
    alldata = pd.merge(alldata,enc_vitals, on='PatientEncounterCSNID')
    alldata_all = alldata.copy()
    #alldata.head(20)
    alldata = alldata_all.loc[(~alldata_all['Absolute Mono Count'].isna())]# & (~alldata_all['D-DIMER'].isna())]#\
                            #& (~alldata_all['Mean Platelet Volume'].isna()) & (~alldata_all['PatientBMI'].isna())]
    for filtf in filterBy:
        alldata = alldata.loc[(~alldata[filtf].isna())]

    infMarkers = ["C-Reactive Protein", "Ferritin", "D-DIMER","Lactate Dehydrogenase"]
    if anyInfMarker:
        alldata = alldata_all.loc[(~alldata_all['C-Reactive Protein'].isna()) | (~alldata_all['D-DIMER'].isna()) |\
                                (~alldata_all['Ferritin'].isna()) | (~alldata_all['Lactate Dehydrogenase'].isna())]
    

    #influenzaB = pd.get_dummies(alldata['Influenza B PCR'],drop_first=True, prefix='influenzaB')
    #influenzaA = pd.get_dummies(alldata['Influenza  A PCR'],drop_first=True, prefix='influenzaA')
    #RsvPcr = pd.get_dummies(alldata['RSV PCR'],drop_first=True, prefix='RSV_PCR')

    TestResult = pd.get_dummies(alldata['TestResult'],drop_first=False)
    Sex = pd.get_dummies(alldata['Sex'],drop_first=True)

    #alldata.drop(['Influenza B PCR', 'Influenza  A PCR', 'RSV PCR'], axis=1, inplace=True)
    alldata.drop(['TestResult', 'PatientID', 'Sex'], axis=1, inplace=True)
    #alldata = pd.concat([alldata, influenzaB, influenzaA, RsvPcr],axis=1)
    alldata = pd.concat([alldata,TestResult, Sex],axis=1)

    #pd.set_option('max_columns', None)
    alldata = alldata.apply(pd.to_numeric, errors='coerce')

    #### Get latest testing encounter ids

    """test_num=40
    test_datesDF = pd.DataFrame()
    test_datesDF['PatientEncounterCSNID'] = test_dates.keys()
    test_datesDF['date'] = test_dates.values()
    test_datesDF = test_datesDF.loc[test_datesDF['PatientEncounterCSNID'].isin(alldata['PatientEncounterCSNID'])]
    test_datesDF = test_datesDF.merge(alldata[['PatientEncounterCSNID','positive']], on='PatientEncounterCSNID')
    test_datesDF = test_datesDF.loc[~(test_datesDF['PatientEncounterCSNID'].isin(multi_eids)) & \
                                    ~(test_datesDF['PatientEncounterCSNID'].isin(second_eids))]
    test_negatives = test_datesDF.loc[test_datesDF['positive']==0].sort_values(by='date').iloc[-test_num:]['PatientEncounterCSNID'].unique()
    test_positives = test_datesDF.loc[test_datesDF['positive']==1].sort_values(by='date').iloc[-test_num:]['PatientEncounterCSNID'].unique()
    #print(len(test_negatives))
    #print(len(test_positives))"""

    return alldata, list(alldata['PatientEncounterCSNID'].values),\
    list(alldata['PatientEncounterCSNID'].loc[~alldata['PatientEncounterCSNID'].isin(multi_eids)].values)

import numpy as np
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tqdm
from datetime import date
from datetime import timedelta



def CedarsDataLoader(patient_file_path=None, flat_file_path=None, filterBy=[], anyInfMarker=False):
    if patient_file_path is None:
        patient_file_path = <path_cedars_patients_file>
    if flat_file_path is None:
        flat_file_path = <path_cedars_flat_file>
    pf = pd.read_csv(patient_file_path)
    #pf.head()


    """if flat_file_path.split('.')[-1]=='xls':
        cedars_labs = pd.read_excel(flat_file_path)
    else:
        cedars_labs = pd.read_csv(flat_file_path)"""
    cedars_labs = pd.read_csv(flat_file_path)
    cedars_labs['rslt_dt'] = pd.to_datetime(cedars_labs['rslt_dt'])
    #cedars_labs.head()
    

    component_names=['HGB', 'MCH', 'MCV', 'AALY', 'AABA', 'HAEO', 'ALymph', 'RBC', 'AAMO', 'AAGR', 'HA1C', 'CRPI', 'DDIM', 'FER',\
                    'PCALC', 'IL6J', 'LDH', 'PAC', 'BMI', 'Temp', 'BP', 'Pulse']
    #cedars_labs.loc[cedars_labs['lab_test_cd'].isin(component_names)]
    encids = cedars_labs['enc_no'].unique()
    test_dates_c={}
    for enc, group in cedars_labs.groupby(['enc_no']):
        test_dates_c[enc]=group['rslt_dt'].min()
        
    def get_test_date_c(x):
        try:
            return test_dates_c[x]
        except:
            return 0

    #print(test_dates)
    cedars_labs['test_dt'] = cedars_labs['enc_no'].apply(lambda x: get_test_date_c(x))
    cedars_labs['test_dt'] = pd.to_datetime(cedars_labs['test_dt'], errors='coerce')
    
    features = {}
    df_groups = cedars_labs[['enc_no','rslt_dt', 'lab_test_cd', 'test_rslt']]\
        .loc[(cedars_labs['rslt_dt']==cedars_labs['test_dt']) | (cedars_labs['rslt_dt']==cedars_labs['test_dt']+timedelta(days=1))]\
        .groupby('enc_no')
        
    for eid, group in df_groups:
        #print(eid)
        features[eid]={}
        idx = group.groupby('lab_test_cd')['rslt_dt'].transform(min) == group['rslt_dt']
        first_vals = group[['lab_test_cd','test_rslt']][idx]
        for cn in component_names:
            try:
                v = first_vals['test_rslt'].loc[first_vals['lab_test_cd']==cn]
                vl = len(v)
                if vl>0:
                    #print(v.values[0])
                    features[eid][cn]=v.values[0]
                    # debugging:
                    #if cn=='CRP':
                    #    print(v.values[0])
            except:
                print("Unexpected error:", sys.exc_info()[0])
            
    #print(len(features))
    
    component_names_to_use = component_names

    dataset = {'enc_no' : []}
    for cn in component_names_to_use:
        dataset[cn]=[]
        
    for eid in features:
        dataset['enc_no'].append(eid)
        for cn in component_names_to_use:
            if cn in features[eid]:
                dataset[cn].append(features[eid][cn])
            else:
                dataset[cn].append(np.nan)
                
    datasetDF = pd.DataFrame(data=dataset).set_index('enc_no')
       

    pdd = pd.merge(datasetDF, cedars_labs[['enc_no', 'id']], on='enc_no')
    datasetDF = pd.merge(pdd,pf, on='id').drop(['admit','icu','intubated','prone','death','id','BMI'], axis=1).drop_duplicates()
    datasetDF = datasetDF.rename(columns={
        'enc_no':'PatientEncounterCSNID',
        'HGB':'Hemoglobin',
        'MCH':'Mean Corpuscular Hemoglobin',
        'MCV':'Mean Corpuscular Volume',
        'AALY':'Leukocyte',
        'AABA':'Absolute Baso Count',
        'HAEO':'Absolute Eos Count',
        'ALymph':'Absolute Lymphocyte Count',
        'RBC':'Red Blood Cell Count',
        'AAMO':'Absolute Mono Count',
        'AAGR': 'Absolute Neut Count',
        'HA1C':'Hgb A1c - HPLC',
        'CRPI':'C-Reactive Protein',
        'DDIM':'D-DIMER',
        'FER':'Ferritin',
        'PCALC':'Procalcitonin',
        'IL6J':'Interleukin-6,Highly Sensitive',
        'LDH':'Lactate Dehydrogenase',
        'PAC':'PLATELET COUNT, AUTO',
        'bmi':'PatientBMI',
        'Temp':'PatientTemperature',
        'BP':'BPSystolic',
        'Pulse':'PatientPulse',
        'age':'Age',
        'sex':'Sex'
                             })
    
    datasetDF['Lymp/Neut'] = pd.to_numeric(datasetDF['Absolute Lymphocyte Count'])/pd.to_numeric(datasetDF['Absolute Neut Count'])
                             
    alldata_cedars = datasetDF.copy()
    alldata_cedars['positive']=1
    alldata_cedars['negative']=0   
    alldata_cedars = alldata_cedars.loc[(~alldata_cedars['Absolute Mono Count'].isna())]# & (~alldata_cedars['D-DIMER'].isna())]
                            #& (~alldata_cedars['PatientBMI'].isna())]
       
    for filtf in filterBy:
        alldata_cedars = alldata_cedars.loc[(~alldata_cedars[filtf].isna())]
        
        
    infMarkers = ["C-Reactive Protein", "Ferritin", "D-DIMER","Lactate Dehydrogenase"]
    if anyInfMarker:
        alldata_cedars = alldata_cedars.loc[(~alldata_cedars['C-Reactive Protein'].isna()) | (~alldata_cedars['D-DIMER'].isna()) |\
                                (~alldata_cedars['Ferritin'].isna()) | (~alldata_cedars['Lactate Dehydrogenase'].isna())]

    Sex = pd.get_dummies(alldata_cedars['Sex'],drop_first=True)

    alldata_cedars.drop(['Sex'], axis=1, inplace=True)
    alldata_cedars = pd.concat([alldata_cedars, Sex],axis=1)


    #pd.set_option('max_columns', None)
    alldata_cedars = alldata_cedars.apply(pd.to_numeric, errors='coerce')
    alldata_cedars = alldata_cedars.rename(columns={1.0:'Male'})


    #print(set(alldata_cedars.columns).difference(set(alldata.columns)))
    #print(set(alldata.columns).difference(set(alldata_cedars.columns)))
    ucla_columns = ['PatientEncounterCSNID', 'Hemoglobin', 'Leukocyte',\
       'Absolute Lymphocyte Count', 'Absolute Neut Count',\
       'PLATELET COUNT, AUTO', 'C-Reactive Protein', 'Ferritin', 'D-DIMER',\
       'Absolute Baso Count', 'Absolute Eos Count', 'Absolute Mono Count',\
       'Procalcitonin', 'Lactate Dehydrogenase', 'Red Blood Cell Count',\
       'Mean Corpuscular Volume', 'Mean Corpuscular Hemoglobin',\
       'Red Cell Distribution Width-CV', 'Red Cell Distribution Width-SD',\
       'Mean Platelet Volume', 'Interleukin-6,Highly Sensitive',\
       'Hgb A1c - HPLC', 'Lymp/Neut', 'Age', 'PatientBMI',\
       'PatientTemperature', 'PatientPulse', 'BPSystolic', 'BPDiastolic',\
       'PatientHeight', 'negative', 'positive', 'unknown', 'Male']
    diffcols = list(set(ucla_columns).difference(set(alldata_cedars.columns)))
    for col in diffcols:
        alldata_cedars[col]=np.nan    

        
    return alldata_cedars, list(alldata_cedars['PatientEncounterCSNID'].values), []
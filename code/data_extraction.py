#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:42:37 2023

@author: chouche7
"""

import pandas as pd
import numpy as np

#read_data
#patients = pd.read_csv('../../mimic3/data/PATIENTS.csv')
admissions = pd.read_csv('../../mimic3/data/ADMISSIONS.csv')

lab_events = pd.read_csv('../../mimic3/data/LABEVENTS.csv')
lab_items = pd.read_csv("../../mimic3/data/D_LABITEMS.csv")

vital_signs = pd.read_csv("../../mimic3/data/CHARTEVENTS.csv")
vital_items = pd.read_csv("../../mimic3/data/D_ITEMS.csv")

#extract item_ids
vitals = ["heart rate", "non invasive blood pressure systolic", "non invasive blood pressure diastolic", "temperature celsius", "respiratory rate", "par-oxygen saturation"]
lab = ["lactate", "pc02", "ph", "albumin", "bicarbonate", "calcium",\
        "creatinine", "glucose", "magnesium", "potassium", "sodium", \
            "blood urea nitrogen", "platelet count"]


#select only the labevents that pertain to the 16 tests
lab_items["LABEL"] = lab_items["LABEL"].map(str.lower)
lab_item_id = lab_items.loc[lab_items["LABEL"].isin(lab), "ITEMID"] 
lab_events = lab_events.loc[lab_events["ITEMID"].isin(lab_item_id)]
lab_events = lab_events[["SUBJECT_ID", "ITEMID", "VALUENUM", "CHARTTIME"]]
lab_events = lab_events[lab_events["VALUENUM"].notnull()]
lab_events.to_csv("../../mimic3/data/LABEVENTS_SHORT.csv")

#select only the signs that pertain to the 6 vitals
vital_items["LABEL"] = vital_items["LABEL"].astype(str).map(str.lower)
vital_item_id = vital_items.loc[vital_items["LABEL"].isin(vitals), "ITEMID"] 
vital_item_id = vital_item_id[(vital_item_id != 211) & (vital_item_id != 618)]
vital_signs = vital_signs.loc[vital_signs["ITEMID"].isin(vital_item_id)]
vital_signs = vital_signs[["SUBJECT_ID", "ITEMID", "VALUENUM", "CHARTTIME"]]
vital_signs = vital_signs[vital_signs["VALUENUM"].notnull()]
vital_signs.to_csv("../../mimic3/data/CHARTEVENTS_SHORT.csv")

#create our labels using deathtime
admissions["mortality"] = ~admissions["DEATHTIME"].isnull()
admissions["mortality"] = admissions["mortality"].astype(int)
mortality = admissions[["SUBJECT_ID", "mortality"]].drop_duplicates()
both = mortality.groupby('SUBJECT_ID').filter(lambda x: len(x) > 1)
mortality = mortality[~mortality['SUBJECT_ID'].isin(both['SUBJECT_ID'].unique()) | (mortality['mortality'] == 1)]
mortality.to_csv("../../mimic3/data/MORTALITY.csv")
#admissions.to_csv("../../mimic3/data/ADMISSIONS_SHORT.csv")







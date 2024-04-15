import matplotlib
import pandas
import pickle
import ShoeUtils as UT
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog
import tkinter as tk
from os.path import exists
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
picklefile =  'PWI_ALL_Subj'
RELOAD = True
#Update pickle file with appropriate sub
folder_path = 'G:\My Drive\ROAR Research\Students\Montaha\Research - PWE'

if RELOAD:

    fileNames = ['2TM'] #
    subjectNames = ['Sub006']#,,

    allTrials = []
    for s in subjectNames:
        print(s)
        for f in fileNames:
            fileBaseline = os.path.join(folder_path, s, f"{f}.xlsx")
            if exists(fileBaseline):
                aux = UT.MatObject(fName=fileBaseline, freq=120, newMat=True)
                aux.raw['Subject'] = s
                aux.raw['Trial'] = f
                aux.raw['Side'] = 'Right'
                aux.raw['Side'].loc[aux.raw['isLeft']] = 'Left'

                allTrials.append(aux.raw)

    allTrials = pandas.concat(allTrials)
    all_data = {'Mat Data': allTrials}
    with open(picklefile, 'wb') as f:
            pickle.dump(all_data, f)

else:
    with open(picklefile, 'rb') as f:
        all_data = pickle.load(f)
        allTrials = all_data['Mat Data']
#when ready, you can add the interesting variables here
# interesting_variables = []
spatial = ['Step Length (cm.)', 'Stride Length (cm.)', 'Stride Width (cm.)']
temporal = ['Stance Time (sec.)', 'Swing Time (sec.)', 'Step Time (sec.)', 'Stride Time (sec.)',
            'Stride Velocity (cm./sec.)']
balance = ['Toe In/Out Angle (degrees)', 'Stance COP Dist. (cm.)']
colsPars = ['Laps', 'isLeft', 'StepIndex', 'num', 'Index', 'Sub', 'FogLap']
cols = ['Integ. Pressure (p x sec.)', 'Foot Length (cm.)', 'Foot Length %', 'Foot Width (cm.)',
        'Foot Area (cm. x cm.)', 'Foot Angle (degrees)', 'Toe In/Out Angle (degrees)',
        'Foot Toe X Location (cm.)', 'Foot Toe Y Location (cm.)', 'Foot Heel X Location (cm.)',
        'Foot Heel Y Location (cm.)', 'Foot Center X Location (cm.)', 'Foot Center Y Location (cm.)',
        'Step Length (cm.)', 'Absolute Step Length (cm.)', 'Stride Length (cm.)', 'Stride Width (cm.)',
        'Step Time (sec.)', 'Stride Time (sec.)', 'Stride Velocity (cm./sec.)', 'DOP (degrees)',
        'Gait Cycle Time (sec.)', 'Stance Time (sec.)', 'Stance %', 'Swing Time (sec.)', 'Swing %',
        'Single Support (sec.)', 'Single Support %', 'Initial D. Support (sec.)', 'Initial D. Support %',
        'Terminal D. Support (sec.)', 'Terminal D. Support %', 'Total D. Support (sec.)',
        'Total D. Support %', 'CISP Time (sec.)', 'CISP AP (%)', 'CISP ML (%)', 'Stance COP Dist. (cm.)',
        'SS COP Dist. (cm.)', 'DS COP Dist. (cm.)', 'Stance COP Dist. %', 'SS COP Dist. %',
        'Stance COP Path Eff. %', 'SS COP Path Eff. %', 'DS COP Path Eff.%']
pressures = ['Left Foot Pressure', 'Right Foot Pressure', 'Total Pressure', 'Other Pressure']
#
#





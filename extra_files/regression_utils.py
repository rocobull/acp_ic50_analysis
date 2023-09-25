#!/usr/bin/env python3

import random
import numpy as np
import pandas as pd
#import scipy.stats
#import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
#from sklearn.externals import joblib


# Viruses in the Seaman panel
seaman2010 = [
    'MW965_26', 'SF162', 'MN_3', 'DJ263_8', '00836_2_5', 
    'BX08_16', 'BAL_26', 'HXB2', '242_14', '25710_2_43', 
    'H077_31', '6535_3', 'SS1196_1', 'ZM197_7', 'H029_12', 
    '1012_11_TC21_3257', 'H086_8', 'Q23_17', 'BZ167_12', '25711_2_4', 
    'ZM109_4', '25925_2_22', '1056_10_TA11_1826', '16845_2_22', 'DU156_12', 
    'TRO_11', '26191_2_48', 'SC422_8', 'WITO4160_33', '0013095_2_11', 
    'REJO4541_67', 'ZM135_10A', '700010040_C9_4520', 'DU123_06', 'ZM249_1', 
    '16936_2_21', 'Q842_D12', 'SC05_8C11_2344', 'PRB926_04_A9_4237', 'Q259_17', 
    'T255_34', 'H022_7', 'Q769_D22', '1006_11_C3_1601', 'T280_5', 
    'CAP210_E8', 'WEAU_D15_410', '263_8', '0330_V4_C3', 'AC10_29', 
    '3415_V1_C1', '6240_08_TA5_4622', 'T252_7', 'CH110_2', '6244_13_B5_4567', 
    'Q168_A2', '235_47', '62357_14_D3_4589', 'CAAN5342', 'H080_23', 
    'DU172_17', 'DU151_02', 'ZM233_6', 'CAP45_G3', 'ZM214_15', 
    '0439_V5_C1', '269_12', '9021_14_B2_4571', 'H030_7', '3718_3_11', 
    '0260_V5_C1', '247_23', 'QH0692_42', 'THRO4156', '3365_V2_C20', 
    'ZM53_12', '1054_07_TC4_1499', '16055_2_3', 'RHPA4259_7', 'CH181_12', 
    'T250_4', '001428_2_42', 'DU422_1', 'CH064_20', 'H031_7', 
    'CH111_8', 'CH119_10', 'Q461_E2', '211_9', 'CH117_4', 
    'H035_18', '928_28', 'CH070_1', 'CH038_12', 'CH114_8', 
    'H061_14', 'T253_11', 'T257_31', 'PVO_4', 'TRJO4551_58', 
    'H078_14', 'T278_50', 'H079_2', 'CH115_12', 'CH120_6', 'T251_18']


# Antibodies under study
antibodies = [
    '12A21', '1B2530', '3BNC117', '8ANC131', '8ANC134', 
    'b12', 'b13', 'CAP257-RH1', 'CH103', 'CH235', 
    'CH235.12', 'HJ16', 'IOMA', 'N6', 'NIH45-46', 
    'VRC01', 'VRC03', 'VRC06', 'VRC06b', 'VRC07', 
    'VRC23', 'VRC-CH31', 'VRC-PG04', 'VRC-PG20' ]


# Experimental breadth of the previous antibodies
breadth_exp = [
    0.46, 0.11, 0.78, 0.32, 0.31,
    0.18, 0.03, 0.01, 0.39, 0.00,
    0.64, 0.16, 0.19, 0.97, 0.77,
    0.70, 0.45, 0.16, 0.31, 0.84,
    0.16, 0.67, 0.64, 0.58 ]


open_files = list()
def get_open_file(fname):
    '''
        Open file and store it in memory.
    '''
    for opened in open_files:
        if opened[0]==fname:
            return opened[1]
    fp = open(fname, 'r')
    lines = [l.strip().split() for l in fp.readlines()]
    open_files.append((fname, lines))
    return lines
    
def get_keys(nkeys, verbose=False):
    """
        Get the "most important" nkeys.
        Default model is obtained with nkeys=19
    """
    keys = []
    keys += ['E_elec', 'E_vdw', 'E_gbsa']
    keys += ['foldx']
    keys += ['prodigy']
    keys += ['MD_h1']   # Highest correlation between MD_h1 and MD_h2 -> 0.964         19
    keys += ['ZRANK']   # Highest correlation between ZRANK and ZRANKr -> 0.845        18
    keys += ['ENM_R6']  # Highest correlation between ENM_EXP and ENM_R6 -> 0.833      17
    keys += ['DFIRE']   # Highest correlation between ZRANKr and DFIRE -> 0.731        16
    keys += ['IC_CP']   # Highest correlation between IC_CP and RFCB -> 0.675          15
    keys += ['ZRANKr']  # Highest correlation between RFHA and ZRANKr -> 0.642         14
    keys += ['IC_CC']   # Highest correlation between MD_h2 and IC_CC -> 0.632         13
    keys += ['MD_sasa'] # Highest correlation between MD_sasa and ENM_EXP -> -0.590    12
    keys += ['RFCB']    # Highest correlation between RFCB and RFHA -> 0.573           11
    keys += ['IC_AC']   # Highest correlation between IC_AP and IC_AC -> -0.549        10
    keys += ['IC_AA']   # Highest correlation between IC_AA and RFHA -> -0.488          9
    keys += ['IC_PP']   # Highest correlation between IC_PP and ENM_EXP -> 0.426        8
    keys += ['NIS_A']   # Highest correlation between RFHA and NIS_A -> 0.407           7
    keys += ['RFHA']    # Highest correlation between ENM_EXP and RFHA -> 0.401         6
    keys += ['ENM_EXP'] # Highest correlation between ENM_EXP and EXP -> -0.292         5
    keys += ['IC_AP']   # Highest correlation between IC_AP and EXP -> 0.158            4
    keys += ['MD_h2']   # Highest correlation between MD_h2 and MD_h3 -> -0.154         3
    keys += ['MD_h3']   # Highest correlation between MD_h3 and EXP -> -0.084           2
    keys += ['NIS_C']   # Highest correlation between NIS_C and EXP -> 0.056            1
    keys = keys[::-1]
    keys = keys[:nkeys]
    if verbose:
        print(keys)
        print("Number of descriptors : %d/%d" % (len(keys), 24))
    return keys


def extract_data(fname, keys, exclude=None, keep=0):
    """
        Extract the wanted descriptors from a data file.
        fname   : filename containing all descriptors (any of ic50_*.dat)
        keys    : list of descriptors to extract
    """
    kept = 0
    indexes = [] ###
    if exclude:
        klist = None
        data_init = False
        with open(fname, 'r') as fp:
            for line in fp:
                if line[0]=='#':
                    klist = line.strip().split()[1:]
                    continue
                split = line.split()
                indexes.append(split[0]) ###
                ab = split[0].split('__')[1]
                if ab in exclude:
                    if kept>=keep:
                        continue
                    kept += 1
                desc = np.array(split[1:], dtype=float)
                if data_init:
                    data = np.append(data, [desc], axis=0)
                else:
                    data = [desc]
                    data_init = True
    else:
        klist = None
        with open(fname, 'r') as fp:
            for line in fp:
                if line[0]=='#':
                    klist = line.split()
                else:
                    indexes.append(line.split()[0]) ###
        data = np.genfromtxt(fname, delimiter='\t')
    if not klist:
        raise ValueError('Impossible to find header line in %s' % (fname))
    ndx = [klist.index(k) for k in keys]
    return pd.DataFrame(data[:,ndx], columns=keys, index=indexes) ###


def fit_regressor(keys, fname='ic50_regression.dat', random_state=np.random, to_exclude=None, verbose=False):
    """
        Fit an MLP regressor.
        keys         : list of descriptors to use
        fname        : file name containing all descriptors
        random_state : an integer to use as seed for the random number generator
        to_exclude   : list of antibodies to exclude from the training
    """

    # Extract experimental values and wanted descriptors
    Y = extract_data(fname, ['IC50'], to_exclude).ravel()
    X = extract_data(fname, keys, to_exclude)

    # Apply log ~ regressor over pIC50
    Y = -np.log(Y)

    # Scaling all descriptors
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    # Split training and test set
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle=True, random_state=random_state)

    # Fit MLP regressor
    # model = MLPRegressor(hidden_layer_sizes=(10), activation='logistic', solver='lbfgs', random_state=random_state)
    # model.fit(x_train, y_train)
    # y_test_predict = model.predict(x_test)
    # y_train_predict = model.predict(x_train)

    return [x_train, y_train, x_test, y_test] #,y_train_predict, y_test_predict] #[keys, scaler, model],


def fit_classifier(keys, fname='ic50_classify.dat', random_state=np.random, to_exclude=None, keep=0, verbose=False, method='MLP1'):
    """
        Fit an MLP classifier
        keys         : list of descriptors to use
        fname        : file name containing all descriptors
        random_state : an integer to use as seed for the random number generator
        to_exclude   : list of antibodies to exclude from the training
        keep         : amount o experimental data to keep from the to_exclude list
        method       : name of the machine learning method to use (MLP1, MLP2, RF31, KNN, SVM)
    """

    # Extract experimental values and wanted descriptors
    Y = extract_data(fname, ['IC50'], to_exclude, keep).ravel()
    X = extract_data(fname, keys, to_exclude, keep)

    # Scale all descriptors
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Split the available data in train and test sets (50/50) randomly.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, shuffle=True, random_state=random_state)
    if verbose:
        print('Number of data : %d' % (Y.size))
        print('Train size     : %d' % (y_train.size))
        print('Test size      : %d' % (y_test.size))

    # Fit the MLPClassifier
    if method=='MLP1':
        model = MLPClassifier(hidden_layer_sizes=(10), activation='logistic', solver='lbfgs', random_state=random_state)
    elif method=='MLP2':
        model = MLPClassifier(hidden_layer_sizes=(10,5), activation='logistic', solver='lbfgs', random_state=random_state)
    elif method=='RF31':
        model = RandomForestClassifier(n_estimators=31)
    elif method=='SVM':
        model = SVC(kernel='rbf')
    elif method=='KNN':
        model = KNeighborsClassifier(15, weights='distance')
    else:
        raise ValueError('Unknown method %s' % (method))
    model.fit(x_train, y_train)

    # Make prediction
    y_test_predict = model.predict(x_test)
    y_train_predict = model.predict(x_train)

    # Print confusion matrix
    CF_test  = confusion_matrix(y_test, [ round(float(i)) for i in y_test_predict ])*100.0/y_test_predict.shape[0]
    CF_train = confusion_matrix(y_train, [ round(float(i)) for i in y_train_predict ])*100.0/y_train_predict.shape[0]
    #print('TN', CF_train[0][0])
    #print('FP', CF_train[0][1])
    #print('FN', CF_train[1][0])
    #print('TP', CF_train[1][1])
    #print('P ', CF_train[1][1]+CF_train[1][0])
    #print('N ', CF_train[0][0]+CF_train[0][1])
    ACC_train = CF_train[0][0]+CF_train[1][1]
    ACC_test = CF_test[0][0]+CF_test[1][1]
    BACC_train = 100*0.5*(CF_train[1][1]/(CF_train[1][1]+CF_train[1][0]) + CF_train[0][0]/(CF_train[0][0]+CF_train[0][1]))
    BACC_test = 100*0.5*(CF_test[1][1]/(CF_test[1][1]+CF_test[1][0]) + CF_test[0][0]/(CF_test[0][0]+CF_test[0][1]))
    if verbose:
        print("Confusion Matrix:")
        print("                  TN      FN      FP      TP     ACC    BACC")
        print("Check     : ", " ".join([ "%6.2f%%" % (d) for d in CF_train.ravel(order='F')]), " %6.2f%% %6.2f%%" % (ACC_train, BACC_train))
        print("Predict   : ", " ".join([ "%6.2f%%" % (d) for d in CF_test.ravel(order='F')]), " %6.2f%% %6.2f%%" % (ACC_test, BACC_test))

    return [keys, scaler, model], [x_train, y_train, y_train_predict, x_test, y_test, y_test_predict], ACC_test


def eval_model(abname, agname, fname, model):
    """
        Apply the model to a given antibody/antigen complex, taking descriptors from fname.
        Good for IC50 regression and classify.
        abname  : name of the antibody to study
        agname  : name of the antigen to study
        fname   : file containing all descriptors
        model   : a list containing: list of descriptors, the standard scaler, and the MLP
    """
    if agname in ['271_11', 'T266_60', 'T33_7']:
        return None
    cpxname = '%s__%s' % (agname, abname)
    fields = list()
    found = False
    fp = get_open_file(fname)
    #with open(fname, 'r') as fp:
    for line in fp:
        if line[0][0]=='#':
            #fields = line.strip().split()[2:]
            fields = line[2:]
        #if line.split()[0]==cpxname:
        if line[0]==cpxname:
            #alldesc = np.array(line.split()[2:], dtype=float)
            alldesc = np.array(line[2:], dtype=float)
            found = True
            break
    if not fields:
        raise ValueError('Could not find header in descriptors file')
    if not found:
        raise ValueError('Could not find descriptors for complex %s' % (cpxname))

    #############################################################
    klist, scaler, mdl = model
    alldesc = alldesc.reshape(1, -1)
    ndx = [ fields.index(k) for k in klist ]
    desc = alldesc[:,ndx]
    desc = scaler.transform(np.array(desc).reshape(1, -1))
    calc = mdl.predict(desc)[0]
    #############################################################
    return calc


def compute_breadth(abname, model, fname='ic50_breadth.dat'):
    """
        Compute the breadth of an antibody given the model to use.
        abname  : name of the antibody
        model   : a list containing: list of descriptors, the standard scaler, and the MLP
        fname   : file containing all descriptors
    """
    br = 0
    tot = 0
    for vname in seaman2010:
        ic50 = eval_model(abname, vname, fname, model)
        if ic50 is not None:
            br += ic50>=1
            tot += 1
    return tot, br, br/tot


def compute_breadth_average(abname, keys, fname, classifier=True, ntrials=30, random_state=np.random, to_exclude=None, keep=0, verbose=False):
    """
        Compute the breadth for the given antibody.
        It makes an average over ntrials generated models.
    """
    allbreadth = list()
    for i in range(ntrials):
        if classifier:
            model, _, _ = fit_classifier(keys, fname=fname, random_state=random_state*i, to_exclude=to_exclude, keep=keep, verbose=verbose)
        else:
            model, _ = fit_regressor(keys, fname=fname, random_state=random_state*i, to_exclude=to_exclude)
        _, _, breadth = compute_breadth(abname, model)
        allbreadth.append(breadth)
    avg = np.mean(allbreadth)
    std = np.std(allbreadth)
    return avg, std


def compute_breadth_all_average(ablist, keys, fname, predictor, ntrials=30, random_state=np.random, method='MLP1'):
    """
        Compute the breadth for all antibodies in ablist.
        It makes an average over ntrials generated models.
    """
    allbreadth = dict()
    #acc_list = list()
    for abname in ablist:
        allbreadth[abname] = list()
    for i in range(ntrials):
        x_train, y_train, x_test, y_test = fit_regressor(keys, fname=fname, random_state=random_state*i)
        predictor.fit(x_train, y_train)
        for abname in ablist:
            _, _, breadth = compute_breadth(abname, predictor)
            allbreadth[abname].append(breadth)
    toret = dict()
    for abname in ablist:
        avg = np.mean(allbreadth[abname])
        std = np.std(allbreadth[abname])
        toret[abname] = (avg, std)
    return toret #, acc_list



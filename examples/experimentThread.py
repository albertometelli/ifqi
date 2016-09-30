# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 22:08:52 2016

@author: samuele
"""

from __future__ import print_function
import os
import sys
import cPickle
#import numpy as np


sys.path.insert(0, os.path.abspath('../'))

#from ifqi.fqi.FQI import FQI
#import ifqi.evaluation 
import argparse



"""

Single thread of experimentThreadManager
"""


from ifqi.experiment import Experiment
#from ifqi.fqi.FQI import FQI
from ifqi.utils.datasetGenerator import DatasetGenerator
#from ifqi.examples.variableLoadSave import ExperimentVariables

#------------------------------------------------------------------------------
# Retrive params
#------------------------------------------------------------------------------


parser = argparse.ArgumentParser(description='Execution of one experiment thread provided a configuration file and\n\t A regressor (index)\n\t Size of dataset (index)\n\t Dataset (index)')

parser.add_argument("configFile", type=str, help="Provide the filename of the configuration file")
parser.add_argument("regressor", type=int, help="Provide the index of the regressor listed in the configuration file")
parser.add_argument("size", type=int, help="Provide the index of the size listed in the configuration file")
parser.add_argument("dataset", type=int, help="Provide the index of the dataset")
args = parser.parse_args()

config_file = args.configFile
#Every experiment just run a regressor that is selected by the ExperimentThreadManager, and here we get the index
regressorN = args.regressor
#Every experiment just run a specific dataset size. ExperimentThreadManager select one index of size
sizeN = args.size
#Every experiment just run a specific dataset. ExperimentThreadManager select one specific dataset
datasetN = args.dataset


print("Started experiment with dataset " + str(datasetN) + ", size " + str(sizeN))

mainFolder = "results/"

exp = Experiment(config_file)



#------------------------------------------------------------------------------
# Variables
#------------------------------------------------------------------------------

dir_ = mainFolder + exp.config["experimentSetting"]["savePath"]
iterations = exp.config['rlAlgorithm']['nIteration']
repetitions = exp.config['experimentSetting']["nRepetition"]

nEvaluation = exp.config['experimentSetting']["evaluation"]['nEvaluation'] 
evaluationEveryNIteration = exp.config['experimentSetting']["evaluation"]['everyNIteration'] 

saveFQI = False
if "saveEveryNIteration" in exp.config['experimentSetting']:
    saveFQI = True
    saveEveryNIteration = exp.config['experimentSetting']["saveEveryNIteration"]

experienceReplay = False
if "experienceReplay" in exp.config['experimentSetting']:
    experienceReplay = True
    nExperience = exp.config['experimentSetting']["evaluation"]['nExperience'] 
    experienceEveryNIteration = exp.config['experimentSetting']["evaluation"]['everyNIteration'] 

#Here I take the right size
size = exp.config['experimentSetting']['datasetSize'][sizeN]

environment = exp.getMDP()
regressorName = exp.getModelName(regressorN)
fit_params = exp.getFitParams(regressorN)

ds_filename = mainFolder +  ".regressor_" + str(regressorName) + "size_" + str(size) + "dataset_" + str(datasetN) + ".npy"
pkl_filename = mainFolder +  ".regressor_" + str(regressorName) + "size_" + str(size) + "dataset_" + str(datasetN) + ".pkl"
#TODO:
action_dim = 1

#------------------------------------------------------------------------------
# Dataset Generation
#------------------------------------------------------------------------------

dataset = DatasetGenerator(environment)
if not os.path.isfile(ds_filename):
    dataset.load(ds_filename)
dataset.generate(nEpisodes=size)

#TODO: i have to implement
sast , r = dataset.sastr

#------------------------------------------------------------------------------
# FQI Loading
#------------------------------------------------------------------------------

actualRepetition = 0
actualIteration = 0

if not os.path.isfile(pkl_filename):    #if no fqi present in the directory
    fqi = exp.getFQI(regressorN)
    fqi.partial_fit(sast[:], r[:], **fit_params)
    min_t = 1
else:
    fqi_obj = cPickle.load(open(pkl_filename, "rb"))
    dataset.reset()
    dataset.load()
    fqi = fqi_obj["fqi"]
    actualIteration = fqi_obj["actualIteration"] + 1
    actualRepetition = fqi_obj["actualRepetition"]
    
#------------------------------------------------------------------------------
# FQI Iterations
#------------------------------------------------------------------------------

dataset = None #
replay_experience = False
for repetition in xrange(actualRepetition, repetitions):
    for iteration in xrange(actualIteration, iterations):
        
        #----------------------------------------------------------------------
        # Fit
        #----------------------------------------------------------------------
        
        if replay_experience:
            fqi.partial_fit(sast[:], r[:], **fit_params)
            replay_experience=False
        else:
            fqi.partial_fit(None, None, **fit_params)
                
        
        #----------------------------------------------------------------------
        # SAVE FQI STATUS
        #----------------------------------------------------------------------
        
        if iteration % saveEveryNIteration == 0:
            directory = os.path.dirname(pkl_filename)
            if not os.path.isdir(directory): 
                os.makedirs(directory)
            cPickle.dump({'fqi':fqi,'actualIteration':iteration, "actualRepetition":repetition},open(pkl_filename, "wb"))
            dataset.save(ds_filename)
            
        
    actualRepetition = 0
import numpy as np
import json
import warnings
from gym import spaces
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression

import envs
from models.mlp import MLP
from models import ensemble
from models.actionregressor import ActionRegressor
from ifqi.fqi.FQI import FQI


def _get_MDP(self):
    """
    This function loads the mdp required in the configuration file.
    Returns:
        The required mdp.

    """
    if self.config['mdp']['mdp_name'] == 'CarOnHill':
        return CarOnHill()
    elif self.config['mdp']['mdp_name'] == 'SwingUpPendulum':
        return InvPendulum()
    elif self.config['mdp']['mdp_name'] == 'Acrobot':
        return Acrobot()
    elif self.config["mdp"]["mdp_name"] == "BicycleBalancing":
        return Bicycle(navigate=False)
    elif self.config["mdp"]["mdp_name"] == "BicycleNavigate":
        return Bicycle(navigate=True)
    elif self.config["mdp"]["mdp_name"] == "SwingPendulum":
        return SwingPendulum()
    elif self.config["mdp"]["mdp_name"] == "CartPole":
        return CartPole()
    elif self.config["mdp"]["mdp_name"] == "LQG1D":
        return LQG1D()
    else:
        raise ValueError('Unknown mdp type.')

def _get_model(self, index):
    """
    This function loads the model required in the configuration file.
    Returns:
        the required model.

    """

    stateDim, actionDim = spaceInfo.getSpaceInfo(self.mdp)
    modelConfig = self.config['regressors'][index]

    fitActions = False
    if 'fitActions' in modelConfig:
        fitActions = modelConfig['fitActions']

    if modelConfig['modelName'] == 'ExtraTree':
        model = ExtraTreesRegressor
        params = {'n_estimators': modelConfig['nEstimators'],
                  'criterion': self.config["regressors"][index]['supervisedAlgorithm']
                                          ['criterion'],
                  'min_samples_split': modelConfig['minSamplesSplit'],
                  'min_samples_leaf': modelConfig['minSamplesLeaf']}
    elif modelConfig['modelName'] == 'ExtraTreeEnsemble':
        model = ExtraTreeEnsemble
        params = {'nEstimators': modelConfig['nEstimators'],
                  'criterion': self.config["regressors"][index]['supervisedAlgorithm']
                                          ['criterion'],
                  'minSamplesSplit': modelConfig['minSamplesSplit'],
                  'minSamplesLeaf': modelConfig['minSamplesLeaf']}
    elif modelConfig['modelName'] == 'MLP':
        model = MLP
        params = {'nInput': stateDim,
                  'nOutput': 1,
                  'hiddenNeurons': modelConfig['nHiddenNeurons'],
                  'nLayers': modelConfig['nLayers'],
                  'optimizer': modelConfig['optimizer'],
                  'activation': modelConfig['activation']}
        if fitActions:
            params["nInput"] = stateDim + actionDim
    elif modelConfig['modelName'] == 'MLPEnsemble':
        model = MLPEnsemble
        params = {'nInput': stateDim,
                  'nOutput': 1,
                  'hiddenNeurons': modelConfig['nHiddenNeurons'],
                  'nLayers': modelConfig['nLayers'],
                  'optimizer': modelConfig['optimizer'],
                  'activation': modelConfig['activation']}
        if fitActions:
            params["nInput"] = stateDim + actionDim
    elif modelConfig['modelName'] == 'Linear':
        model = LinearRegression
        params = {}
    elif modelConfig['modelName'] == 'LinearEnsemble':
        model = LinearEnsemble
        params = {}
    else:
        raise ValueError('Unknown estimator type.')

    if fitActions:
        return model(**params)
    else:
        if isinstance(self.mdp.action_space, spaces.Box):
            warnings.warn("Action Regressor cannot be used for continuous "
                          "action environment. Single regressor will be "
                          "used.")
            return model(**params)
        return ActionRegressor(model,
                               self.mdp.action_space.values, decimals=5,
                               **params)
import base_wavelet_model
from importlib import reload
reload(base_wavelet_model)
import novel_activations
reload(novel_activations)
from hypothesis_tests import HypothesisTest
import os
#{'exp_name': 'Wavelet SeparateQMF L4 B1 K8', 'exp_description': 'Baseline implementaiton of a residual network for time series, BtachNorm components removed removed', 'batch_size': 1, 'width': 4096, 'K': 7, 'epochs': 50, 'levels': 4, 'initKernels': [8, 8, 8, 8], 'stride': (2, 1), 'dilations': None, 'padding': 'SAME', 'kernelInitMode': 'SeparateQMF', 'trainKernels': True, 'performReconstruction': True, 'activationLayer': <class 'novel_activations.BiasRelu'>, 'activationLayerParams': [0.5, True], 'data_folder': './mit-bih-arrhythmia-database-1.0.0'}
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

batch_size = 2
for levels in [2,5,7]:
    for minGrad in [0.01,0.05,0.1]:
        config_dict = {
            'exp_name': None,
            'exp_description': None,
            'batch_size': batch_size,
            'width': 4096,
            'K': 10,
            'epochs': 50,
            'levels': levels,
            'initKernels': [16 for _ in range(levels*2)],
            'stride': (2, 1),
            'dilations': None,
            'padding': 'SAME',
            'kernelInitMode': 'Independent',
            'trainKernels': True,
            'performReconstruction': True,
            'activationLayer': novel_activations.TripleLinearAct,
            'activationLayerParams': [None, True, minGrad],
            'useSigmoidOnOutput' : True,
            'inputBatchNorm' : True,
            "data_folder" : "./mit-bih-arrhythmia-database-1.0.0"
        }

        exp_name =  f"TLA Min Grad {minGrad} {levels}-level Independent Trainable Wavelet"
        if exp_name in os.listdir("Tensorboard") and "result_plots.png" in os.listdir("Tensorboard/"+exp_name):
            pass
        else:
            print(exp_name)
            config_dict["exp_name"] = exp_name
            config_dict["exp_description"] = exp_name
            testE = base_wavelet_model.Experiment()
            testE.execute_experiment(config_dict)
os.system("touch /home/adn1/Documents/fyp-main/finished.done")
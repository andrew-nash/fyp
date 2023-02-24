import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pickle
import scipy



class HypothesisTest:
    def __init__(self, exp_name):
        # load an experiment to Test
        try:
            with open(f"./Tensorboard/{exp_name}/config.json.pickle", 'rb') as f:
                self.config_dict = pickle.load(f)
            with open(f"./Tensorboard/{exp_name}/hist.dict.pickle", 'rb') as f:
                self.hist = pickle.load(f)
        except:
            raise ValueError("ERROR: couldn't load expermiment "+f"./Tensorboard/{exp_name}/, make sure you have specified the experimnet name correctly and that the pickle files exist")
        self.final_val_losses   = [self.hist[i]['hist'].history['val_loss'][-1] for i in self.hist.keys()]
        self.final_train_losses = [self.hist[i]['hist'].history['loss'][-1] for i in self.hist.keys()]
    def get_val_loss(self):
        return self.final_val_losses 
    def get_train_loss(self):
        return self.final_train_losses 
    
    def get_overfitted_pair_loss(self):
        return [t-v for t, v in zip(self.final_train_losses,self.final_val_losses)]
    

    def test_val_means_equal(self, secondExp):
        '''
        H0: the mean validation loss for this experiment is the same
        as that of secondExp
        HA: the means are different
        '''
        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        
        return scipy.stats.mannwhitneyu(self.get_val_loss(), secondExp.get_val_loss())

    def test_val_mean_less(self, secondExp):
        '''
        H0: the mean validation loss for this experiment is the same
        as that of secondExp
        HA: the mean loss for this experiment is lower
        '''

        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        
        return scipy.stats.mannwhitneyu(self.get_val_loss(), secondExp.get_val_loss(), alternative="less")

    def test_less_overfitting(self, secondExp):
        '''
        H0: The mean difference between train and validation loss is the same
        HA: The mean difference between train and validation loss is lower for this experiment
        '''

        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        
        return scipy.stats.mannwhitneyu(self.get_overfitted_pair_loss(), secondExp.get_overfitted_pair_loss(), alternative="less")
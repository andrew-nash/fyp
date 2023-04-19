import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import pickle
import scipy
import  base_wavelet_model
from tensorflow.keras.losses import BinaryCrossentropy
Lf = BinaryCrossentropy()


class HypothesisTest:
    def __init__(self, exp_name):
        self.names = [exp_name]
        self.r_losses = []
        self.preds = []

        # load an experiment to Test
        try:
            with open(f"./Tensorboard/{exp_name}/config.json.pickle", 'rb') as f:
                self.config_dict = pickle.load(f)
            with open(f"./Tensorboard/{exp_name}/hist.dict.pickle", 'rb') as f:
                self.hist = pickle.load(f)
        except Exception as e:
            print(e)
            raise ValueError("ERROR: couldn't load expermiment "+f"./Tensorboard/{exp_name}/, make sure you have specified the experimnet name correctly and that the pickle files exist")
        self.final_val_losses   = [self.hist[i]['hist']['val_loss'][-1] for i in range(self.hist['K'])]
        self.final_train_losses = [self.hist[i]['hist']['loss'][-1] for i in range(self.hist['K'])]
    
    def combine(self, secondExp):
        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        self.names+=secondExp.names
        self.final_val_losses+=secondExp.get_val_loss()
        self.final_train_losses+=secondExp.get_train_loss()

    def get_val_loss(self):
        return self.final_val_losses 
    def get_train_loss(self):
        return self.final_train_losses 
    
    def get_overfitted_pair_loss(self):
        return [t-v for t, v in zip(self.final_train_losses,self.final_val_losses)]
            

    def test_val_mean_equal(self, secondExp):
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

    def test_val_mean_greater(self, secondExp):
        '''
        H0: the mean validation loss for this experiment is the same
        as that of secondExp
        HA: the mean loss for this experiment is higher
        '''

        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        
        return scipy.stats.mannwhitneyu(self.get_val_loss(), secondExp.get_val_loss(), alternative="greater")

    def test_less_overfitting(self, secondExp):
        '''
        H0: The mean difference between train and validation loss is the same
        HA: The mean difference between train and validation loss is lower for this experiment
        '''

        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        
        return scipy.stats.mannwhitneyu(self.get_overfitted_pair_loss(), secondExp.get_overfitted_pair_loss(), alternative="less")
    
    def test_more_overfitting(self, secondExp):
        '''
        H0: The mean difference between train and validation loss is the same
        HA: The mean difference between train and validation loss is lower for this experiment
        '''

        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        
        return scipy.stats.mannwhitneyu(self.get_overfitted_pair_loss(), secondExp.get_overfitted_pair_loss(), alternative="greater")
    
    def load_models(self, folds):
        mods = []
        for name in self.names:
            for fold in range(folds):
                foldpadded = '0'*(3-len(str(fold)))+str(fold)      
                mods.append(keras.models.load_model( f"./Tensorboard/{name}/models/{foldpadded}-model"))

        self.models =  mods


    def load_r_peak_preds(self, folds, number_samples):
        exp = base_wavelet_model.Experiment()
        signals, annots = exp.load_data("../mit-bih-arrhythmia-database-1.0.0")
        splits = self.hist['fold_split']

        self.preds = []
        self.trues = []
        self.trueMods = []
        self.predMods = []
        self.load_models(folds)
        i=0
        for name in self.names:
            print(f"{i} of {len(self.names)*folds}", end=', ')
            for k in range(folds):
                model = self.models[i]
                i+=1
                valid_signals = signals[np.where(splits==k)]
                valid_annots = annots[np.where(splits==k)]                
                for sig in range(len(valid_signals)):
                    valid_signal = valid_signals[sig]
                    valid_annot = valid_annots[sig]
                    full_annot = np.zeros(len(valid_signal))
                    full_annot[valid_annot] = 1.0
                    for n in range(number_samples):
                        
                        true_annot = full_annot[4096*n:4096*(n+1)]
                        peaks = np.where(true_annot > 0.3)
                        testS = valid_signal[4096*n:4096*(n+1)]
                        pred = np.array(model(testS.reshape(1,-1))[-1]).reshape(-1)
                        self.preds.append(pred[peaks])
                        self.predMods.append(name)
                valid_signals = signals[np.where(splits!=k)]
                valid_annots = annots[np.where(splits!=k)]                
                for sig in range(len(valid_signals)):
                    valid_signal = valid_signals[sig]
                    valid_annot = valid_annots[sig]
                    full_annot = np.zeros(len(valid_signal))
                    full_annot[valid_annot] = 1.0
                    for n in range(number_samples):
                        true_annot = full_annot[4096*n:4096*(n+1)]
                        nonpeaks = np.where(true_annot < 0.3)
                        testS = valid_signal[4096*n:4096*(n+1)]
                        pred = np.array(model(testS.reshape(1,-1))[-1]).reshape(-1)
                        self.trues.append(pred[nonpeaks])
                        self.trueMods.append(name)
    def get_r_peak_loss(self): 
        if self.preds == []:
            self.load_r_peak_preds(self.hist['K'], 4)
        self.r_losses = []
        for k in range(len(self.preds)):
            true = np.array([1.0 for _ in range(len(self.preds[k]))])
            self.r_losses.append(Lf(true, np.array(self.preds[k])))
        self.r_losses = [x.numpy() for x in self.r_losses]
        return self.r_losses

    def get_non_r_peak_loss(self): 
        if self.trues == []:
            self.load_r_peak_preds(self.hist['K'], 4)
        self.offr_losses = []
        for k in range(len(self.trues)):
            true = np.array([0.0 for _ in range(len(self.trues[k]))])
            self.offr_losses.append(Lf(true, np.array(self.trues[k])))
        self.offr_losses = [x.numpy() for x in self.offr_losses]
        return self.offr_losses
    
    def test_val_mean_less_at_r_peaks(self, secondExp):
        if isinstance(secondExp, str):
            secondExp = HypothesisTest(secondExp)
        if self.r_losses == []:
            self.get_r_peak_loss()
        if secondExp.r_losses == []:
            secondExp.get_r_peak_loss()
        return scipy.stats.mannwhitneyu(self.r_losses, secondExp.r_losses, alternative="less")
    
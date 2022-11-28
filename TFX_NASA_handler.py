from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score,roc_auc_score
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np # numbers and algebra in python
import pandas as pd # tables and processing tabular data

class TFX_NASA_Handler:
    """ class to handle data and prepare for ML """
    def __init__(self, name, dirPath, settings = 'default'):
        self.name = name
        self.dirPath = dirPath
        self.settings = settings
        self.X_train = None
        self.y_train = None
        self.X_test = None
        
    def __str__(self):
        return f"{self.name}({self.settings})"
    
    def load_data(self):
        index_names = ['unit_number', 'time_cycles']
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        sensor_names = ['s_{}'.format(i+1) for i in range(0,21)]
        col_names = index_names + setting_names + sensor_names
        
        train = pd.read_csv(self.dirPath + 'train_FD001.txt',sep='\s+',names=col_names)
        test = pd.read_csv(self.dirPath + 'test_FD001.txt',sep='\s+',names=col_names)
        y_test = pd.read_csv(self.dirPath + 'RUL_FD001.txt',sep='\s+',header=None)
        
        return train, test, y_test
    
    def data_prep(self, train, test, drop_sensors = ['s_1','s_5','s_6','s_10','s_16','s_18','s_19'], rollingSmooth = True):
        train = self.add_remaining_useful_life(train)
        train = self.gate_the_RUL(train,cap=150)
        setting_names = ['setting_1', 'setting_2', 'setting_3']
        drop_labels = setting_names + drop_sensors
        self.X_train = train.drop(drop_labels, axis=1)
        self.y_train = self.X_train.pop('RUL')
        self.X_test = test.groupby('unit_number').last().reset_index().drop(drop_labels, axis=1)
        if rollingSmooth:
            f = lambda x: x.rolling(5,min_periods=0).mean() 
            self.X_train = self.X_train.groupby('unit_number').apply(f)
            print("Data Prep Complete, columns dropped and smoothing applied ")
        else:
            print("Data Prep Complete, columns dropped ")  
        
    @staticmethod
    def gate_the_RUL(train,cap=150):
        # Rectifier
        train['RUL'] = np.minimum(train['RUL'],cap)
        return train
    
    @staticmethod
    def add_remaining_useful_life(df):
        # Get the total number of cycles for each unit
        grouped_by_unit = df.groupby(by="unit_number")
        max_cycle = grouped_by_unit["time_cycles"].max()

        # Merge the max cycle back into the original frame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_number', right_index=True)

        # Calculate remaining useful life for each row
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life

        # drop max_cycle as it's no longer needed
        result_frame = result_frame.drop("max_cycle", axis=1)
        return result_frame
    
    @staticmethod
    def binary_classification_performance(y_test, y_pred):
        tp, fp, fn, tn = confusion_matrix(y_test, y_pred).ravel()
        accuracy = round(accuracy_score(y_pred = y_pred, y_true = y_test),2)
        precision = round(precision_score(y_pred = y_pred, y_true = y_test),2)
        recall = round(recall_score(y_pred = y_pred, y_true = y_test),2)
        f1_score = round(2*precision*recall/(precision + recall),2)
        specificity = round(tn/(tn+fp),2)
        npv = round(tn/(tn+fn),2)
        auc_roc = round(roc_auc_score(y_score = y_pred, y_true = y_test),2)


        result = pd.DataFrame({'Accuracy' : [accuracy],
                             'Precision (or PPV)' : [precision],
                             'Recall (senitivity or TPR)' : [recall],
                             'f1 score' : [f1_score],
                             'AUC_ROC' : [auc_roc],
                             'Specificty (or TNR)': [specificity],
                             'NPV' : [npv],
                             'True Positive' : [tp],
                             'True Negative' : [tn],
                             'False Positive':[fp],
                             'False Negative':[fn]})
        return result
    
    @staticmethod
    def ROC_curve(X_test, y_test):
        """ compare Precision and Recall """
        display = PrecisionRecallDisplay.from_estimator(
            classModel, X_test, y_test, name="rand_forest"
        )
        _ = display.ax_.set_title("2-class Precision-Recall curve")
        
    @staticmethod
    def zscore(X_train,X_test):
        # zero-centered normalise (standardise or zscore)
        mu = np.mean(X_train,axis=0)
        sig = np.std(X_train,axis=0)
        X_train = (X_train - mu) / sig
        X_test = (X_test - mu) / sig
        return X_train, X_test
    
    @staticmethod
    def rephrase_BinaryClassification(y_train, y_test, limit = 30):
        """ apply a threshold: above and below becoming classes """
        f = lambda y: y<=limit
        y_train = y_train.apply(f)
        y_test = y_test.apply(f)
        y_test = np.squeeze(y_test)
        return y_train, y_test

    


import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox, filedialog
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC


# select attributes for process
class AttributeSelector(BaseEstimator, TransformerMixin):
    # select attributes
    def __init__(self, Attributes):
        self.attr = Attributes

    def fit(self, df):
        return self
    
    # return dataframe values
    def transform(self, df):
        return df[self.attr].values
    
# data mining process
class DataMining(BaseEstimator, TransformerMixin):
    pass
import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox, filedialog, font
from pathlib import Path
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC


# Data Mining And Machin learning process
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
    __model = None
    __df_prepared = None

    def __preprocessing(self, base, new):
        num_cat = ['Upset', 'control_important_affairs', 'Nervous', 'disability', 'ability', 'Academic_situation', 'control_upset_affairs','Top_performance', 
                   'angered_performance', 'Not_overcome']
        
        obj_cat1 = ["Age", "Academic_Year", "CGPA", "Scholarship"]                      # ordinal encode feature
        obj_cat2 = ["Gender"]                                                           # onehot encode feature

        # number process
        number_pipeline = Pipeline([
            ("attribute_selector", AttributeSelector(Attributes=num_cat)),
            ("standard_scaler", StandardScaler())
        ])

        # ordinal encode process
        obj1_pipeline = Pipeline([
            ("attribute_selector", AttributeSelector(Attributes=obj_cat1)),
            ("ordinal_encode", OrdinalEncoder(handle_unknown='error')),
            ("standard_scaler", StandardScaler())
        ])

        # onehot encode process
        obj2_pipeline = Pipeline([
            ("attribute_selector", AttributeSelector(Attributes=obj_cat2)),
            ("ordinal_encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("standard_scaler", StandardScaler())
        ])

        # final process
        final_pipeline = FeatureUnion(transformer_list=[
            ("number_process", number_pipeline),
            ("object_process1", obj1_pipeline),
            ("object_process2", obj2_pipeline)
        ])

        base_prepared = pd.DataFrame(final_pipeline.fit_transform(base))
        self.__df_prepared = pd.DataFrame(final_pipeline.transform(new))        # new data frame prepared

        return base_prepared

    def fit(self, df):
        # model
        svc = SVC(C=10, kernel='linear')
        # base file reading for machin learn
        base = pd.read_csv('Stress.csv')
        base.columns = ["Age", "Gender", "University", "Department", "Academic_Year", "CGPA", "Scholarship", "Upset", "control_important_affairs", "Nervous", 
                  "disability", "ability", "Academic_situation", "control_upset_affairs", "Top_performance", "angered_performance","Not_overcome", 
                  "Stress_Value", "Stress_Label"]
        
        trans_list = []
        for i in base["Stress_Label"]:
            if i == "Low Stress":
                trans_list.append(0)

            elif i == "Moderate Stress":
                trans_list.append(1)

            elif i == "High Perceived Stress":
                trans_list.append(2)

        base_y = trans_list

        base_prepared = self.__preprocessing(base=base, new=df)

        self.__model = svc.fit(base_prepared, base_y)
        
        return self
    
    def transform(self, df):
        df_prepared = self.__df_prepared.copy()

        df_predict = self.__model.predict(df_prepared)

        y_transform = []
        for i in df_predict:
            if i == 0:
                y_transform.append("Low Stress")
            elif i == 1:
                y_transform.append("Moderate Stress")
            else:
                y_transform.append("High Perceived Stress")

        y_transform = pd.DataFrame(y_transform, columns=["Stress_Label"])

        df.reset_index(inplace=True, drop=True)
        final_df = pd.concat([df, y_transform], axis=1)

        return final_df
    

#======================== create windows ==========================
def single_answer():
    window.destroy()

    root = tk.Tk()
    root.title("Insert Details")
    root.geometry("900x650")

    root.mainloop()

    return main_window()

def group_answer():
    pass

# main window create
def main_window():
    def tips():
        # create tip window
        tip_window = tk.Toplevel(window)
        tip_window.title("Tips")
        tip_window.geometry("250x350")

        img = tk.PhotoImage(file="images/Tips_bg.png")
        Label(tip_window, image=img).place(x=0, y=0)

        # tip for enter details button
        tk.Label(tip_window, text="Enter Details", font=font.Font(size=12, weight='bold'), bg="#F5F5F5", fg="#4A7758").place(x=72, y=10)
        tk.Label(tip_window, text="This button is for personal results", font=("", "11"), bg="#F5F5F5", fg="#6A8372").place(x=12, y=50)

        # tips for upload details button
        tk.Label(tip_window, text="Upload Details", font=font.Font(size=12, weight='bold'), bg="#F5F5F5", fg="#4A7758").place(x=65, y=120)
        tk.Label(tip_window, text="This button is used for uploading files from universities or institudes", 
                 font=("", "11"), wraplength=230, bg="#F5F5F5", fg="#6A8372").place(x=15, y=160)

        tip_window.mainloop()


    global window
    window = tk.Tk()
    window.title("Stress Prediction")
    window.geometry("600x580")
    window.resizable(0, 0)

    main_img = tk.PhotoImage(file="images/Main_pic.png")
    Label(window, image=main_img).place(x=0, y=0)

    tk.Label(window, text="Select the type of stress level result", font=("", "14"), bg="#ffffff", fg="#4A7758").pack(pady=20)

    single_btn = tk.Button(window, text="Enter Details", bg="#6A8372", fg="#ffffff", font=("", "12"), command=single_answer)
    single_btn.place(x=68, y=100, width=120, height=50)

    group_btn = tk.Button(window, text="Upload Details", bg="#6A8372", fg="#ffffff", font=("", "12"), command=group_answer)
    group_btn.place(x=410, y=100, width=120, height=50)

    tip_img = tk.PhotoImage(file="images/Tips.png")
    tk.Button(window, image=tip_img, command=tips).place(x=570, y=0)

    window.mainloop()


# run window 
main_window()
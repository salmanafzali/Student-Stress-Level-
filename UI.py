import tkinter as tk
from tkinter.ttk import *
from tkinter import messagebox, filedialog, font
from pathlib import Path
import webbrowser
import pandas as pd 
import numpy as np
from sklearn.impute import SimpleImputer
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
            ("simple_imputer", SimpleImputer(missing_values=np.nan, strategy='median')),
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
    def save():
        pass

    # To guide the user in answering
    def tips():
        # create window
        tip_window = tk.Toplevel(root)
        tip_window.title("Help Page")
        tip_window.geometry("770x430")

        tip_window.resizable(0, 0)

        tip_img = tk.PhotoImage(file="images/Person_Tip_main.png")
        Label(tip_window, image=tip_img).place(x=0, y=0)

        #========================= questions text =========================
        # CGPA calculator 
        tk.Label(tip_window, text="CGPA:", font=font.Font(weight='bold',  size=10), bg="#5A8377", fg="white").place(x=40, y=5)
        lbl_link = tk.Label(tip_window, text="https://www.aun.edu.ng/docs/cgpa/",cursor="hand2", fg="blue", font=("", "10"), bg="#5A8377")
        lbl_link.place(x=90, y=4)
        lbl_link.bind("<Button-1>", lambda s: webbrowser.open_new("https://www.aun.edu.ng/docs/cgpa/"))

        # questions
        tk.Label(tip_window, text="Question 1:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=40)
        tk.Label(tip_window, text="Did you receive a waiver or scholarship at your university?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=40)

        tk.Label(tip_window, text="Question 2:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=75)
        tk.Label(tip_window, text="In a semester, how often have you felt upset due to something that happened in your academic affairs?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=75)

        tk.Label(tip_window, text="Question 3:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=110)
        tk.Label(tip_window, text="In a semester, how often you felt as if you were unable to control important things in your academic affairs?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=110)

        tk.Label(tip_window, text="Question 4:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=145)
        tk.Label(tip_window, text="In a semester, how often you felt nervous and stressed because of academic pressure?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=145)

        tk.Label(tip_window, text="Question 5:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=180)
        tk.Label(tip_window, text="In a semester, how often you felt as if you could not cope with all the mandatory academic activities?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=180)

        tk.Label(tip_window, text="Question 6:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=215)
        tk.Label(tip_window, text="In a semester, how often you felt confident about your ability to handle your academic / university problems?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=215)

        tk.Label(tip_window, text="Question 7:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=250)
        tk.Label(tip_window, text="In a semester, how often you felt as if things in your academic life is going on your way?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=250)

        tk.Label(tip_window, text="Question 8:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=285)
        tk.Label(tip_window, text="In a semester, how often are you able to control irritations in your academic / university affairs?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=285)

        tk.Label(tip_window, text="Question 9:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=320)
        tk.Label(tip_window, text="In a semester, how often you felt as if your academic performance was on top?", font=("","10"), bg="#5A8377", fg="white").place(x=90, y=320)

        tk.Label(tip_window, text="Question 10:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=355)
        tk.Label(tip_window, text="In a semester, how often you got angered due to bad performance or low grades that is beyond your control?", font=("","10"), bg="#5A8377", fg="white").place(x=95, y=355)

        tk.Label(tip_window, text="Question 11:", font=font.Font(weight='bold', size=10), bg="#5A8377", fg="white").place(x=10, y=390)
        tk.Label(tip_window, text="In a semester, how often you felt as if academic difficulties are piling up so high that you could not overcome them?", font=("","10"), bg="#5A8377", fg="white").place(x=95, y=390)

        tip_window.mainloop()


    # close prewiew window
    window.destroy()

    # create window
    root = tk.Tk()
    root.title("Insert Details")
    root.geometry("950x630")
    root.resizable(0, 0)           # close resize window

    # main picture
    main_img = tk.PhotoImage(file="images/Personal_Main.png")
    Label(root, image=main_img).place(x=0, y=0)

    tk.Label(root, text="Click on the light to receive and view the questions.", font=font.Font(size=12, weight='bold'), bg="#f5f5f5").place(x=125, y=5)

    # light guidance
    tip_img = tk.PhotoImage(file="images/Person_Tip.png")
    tk.Button(root, image=tip_img, command=tips, bg="#5A8377").place(x=540, y=0)

    #============================features input for operations============================
    lbl_name = tk.Label(root, text="Your Name", font=("", 12), bg="#f5f5f5")
    lbl_name.place(x=167, y=70)

    ent_name = tk.Entry(root, font=("", "12"), fg="#5A8377", justify='center')
    ent_name.place(x=145, y=100, width=130)

    #=========next feature=========
    lbl_age = tk.Label(root, text="Age", font=("", "12"), bg="#f5f5f5")
    lbl_age.place(x=380, y=70)

    ent_age = tk.Entry(root, font=("", "12"), fg="#5A8377", justify='center')
    ent_age.place(x=378, y=100, width=40)

    #=========next feature=========
    lbl_cgpa = tk.Label(root, text="CGPA", font=("", "12"), bg="#f5f5f5")
    lbl_cgpa.place(x=550, y=70)

    txt_cgpa = tk.Entry(root, font=("", "12"), fg="#5A8377", justify='center')
    txt_cgpa.place(x=535, y=100, width=80)

    #=========next feature=========
    lbl_gender = tk.Label(root, text="Gender", font=("", "12"), bg="#f5f5f5")
    lbl_gender.place(x=150, y=180)

    gender_menu = ["equivocal", "Female", "Male", "Prefer not to say"]
    gender_select = tk.StringVar()
    OptionMenu(root, gender_select, *gender_menu).place(x=128, y=210, height=23, width=105)

    #=========next feature=========
    lbl_year = tk.Label(root, text="Academic Year", font=("", "12"), bg="#f5f5f5")
    lbl_year.place(x=350, y=180)

    year_menu = ["equivocal", "First Year", "Second Year", "Third Year", "Fourth Year", "Other"]
    year_select = tk.StringVar()
    OptionMenu(root, year_select, *year_menu).place(x=320, y=210, height=23, width=150)

    #=========next feature=========
    lbl_quest1 = tk.Label(root, text="Question 1", font=("", "12"), bg="#f5f5f5")
    lbl_quest1.place(x=570, y=180)

    quest1_menu = ["equivocal", "No", "Yes"]
    quest1_select = tk.StringVar()
    OptionMenu(root, quest1_select, *quest1_menu).place(x=570, y=210, height=23, width=85)

    #  list for all question menu
    questions_menu = ["equivocal", "0", "1", "2", "3", "4"]

    #=========next feature=========
    lbl_quest2 = tk.Label(root, text="Question 2", font=("", "12"), bg="#f5f5f5")
    lbl_quest2.place(x=770, y=180)

    quest2_select = tk.StringVar()
    OptionMenu(root, quest2_select, *questions_menu).place(x=770, y=210, height=23, width=85)

    #=========next feature=========
    lbl_quest3 = tk.Label(root, text="Question 3", font=("", "12"), bg="#f5f5f5")
    lbl_quest3.place(x=140, y=300)

    quest3_select = tk.StringVar()
    OptionMenu(root, quest3_select, *questions_menu).place(x=140, y=330, height=23, width=85)

    #=========next feature=========
    lbl_quest4 = tk.Label(root, text="Question 4", font=("", "12"), bg="#f5f5f5")
    lbl_quest4.place(x=350, y=300)

    quest4_select = tk.StringVar()
    OptionMenu(root, quest4_select, *questions_menu).place(x=350, y=330, height=23, width=85)

    #=========next feature=========
    lbl_quest5 = tk.Label(root, text="Question 5", font=("", "12"), bg="#f5f5f5")
    lbl_quest5.place(x=560, y=300)

    quest5_select = tk.StringVar()
    OptionMenu(root, quest5_select, *questions_menu).place(x=560, y=330, height=23, width=85)

    #=========next feature=========
    lbl_quest6 = tk.Label(root, text="Question 6", font=("", "12"), bg="#f5f5f5")
    lbl_quest6.place(x=770, y=300)

    quest6_select = tk.StringVar()
    OptionMenu(root, quest6_select, *questions_menu).place(x=770, y=330, height=23, width=85)

    #=========next feature=========
    lbl_quest7 = tk.Label(root, text="Question 7", font=("", "12"), bg="#f5f5f5")
    lbl_quest7.place(x=120, y=420)

    quest7_select = tk.StringVar()
    OptionMenu(root, quest7_select, *questions_menu).place(x=120, y=450, height=23, width=85)

    #=========next feature=========
    lbl_quest8 = tk.Label(root, text="Question 8", font=("", "12"), bg="#f5f5f5")
    lbl_quest8.place(x=270, y=420)

    quest8_select = tk.StringVar()
    OptionMenu(root, quest8_select, *questions_menu).place(x=270, y=450, height=23, width=85)

    #=========next feature=========
    lbl_quest9 = tk.Label(root, text="Question 9", font=("", "12"), bg="#f5f5f5")
    lbl_quest9.place(x=430, y=420)

    quest9_select = tk.StringVar()
    OptionMenu(root, quest9_select, *questions_menu).place(x=430, y=450, height=23, width=85)

    #=========next feature=========
    lbl_quest10 = tk.Label(root, text="Question 10", font=("", "12"), bg="#f5f5f5")
    lbl_quest10.place(x=600, y=420)

    quest10_select = tk.StringVar()
    OptionMenu(root, quest10_select, *questions_menu).place(x=600, y=450, height=23, width=85)

    #=========next feature=========
    lbl_quest11 = tk.Label(root, text="Question 11", font=("", "12"), bg="#f5f5f5")
    lbl_quest11.place(x=770, y=420)

    quest11_select = tk.StringVar()
    OptionMenu(root, quest11_select, *questions_menu).place(x=770, y=450, height=23, width=85)

    # Receive Result Button
    tk.Button(root, text="Stress Level Predicton!", command=save, bg="#5A8377", fg="white").place(x=570, y=530, width=130, height=50)

    # back window Button
    tk.Button(root, text="previous window", command=root.destroy, bg="#960018", fg="white").place(x=240, y=530, width=130, height=50)


    root.mainloop()
    
    # back to main window after close root
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

        # background image for tips window
        img = tk.PhotoImage(file="images/Main_Tips_bg.png")
        Label(tip_window, image=img).place(x=0, y=0)

        # tip for enter details button
        tk.Label(tip_window, text="Enter Details", font=font.Font(size=12, weight='bold'), bg="#F5F5F5", fg="#4A7758").place(x=72, y=10)
        tk.Label(tip_window, text="This button is for personal results", font=("", "11"), bg="#F5F5F5", fg="#6A8372").place(x=12, y=50)

        # tips for upload details button
        tk.Label(tip_window, text="Upload Details", font=font.Font(size=12, weight='bold'), bg="#F5F5F5", fg="#4A7758").place(x=65, y=120)
        tk.Label(tip_window, text="This button is used for uploading files from universities or institudes", 
                 font=("", "11"), wraplength=230, bg="#F5F5F5", fg="#6A8372").place(x=15, y=160)

        tip_window.mainloop()

    # create main window on global for better close and open
    global window
    window = tk.Tk()
    window.title("Stress Prediction")
    window.geometry("600x580")
    window.resizable(0, 0)

    # background image
    main_img = tk.PhotoImage(file="images/Main_pic.png")
    Label(window, image=main_img).place(x=0, y=0)

    tk.Label(window, text="Select the type of stress level result", font=("", "14"), bg="#ffffff", fg="#4A7758").pack(pady=20)

    # personal button result
    single_btn = tk.Button(window, text="Enter Details", bg="#6A8372", fg="#ffffff", font=("", "12"), command=single_answer)
    single_btn.place(x=68, y=100, width=120, height=50)

    # file button result 
    group_btn = tk.Button(window, text="Upload Details", bg="#6A8372", fg="#ffffff", font=("", "12"), command=group_answer)
    group_btn.place(x=410, y=100, width=120, height=50)
    
    # light guidance
    tip_img = tk.PhotoImage(file="images/Main_Tips.png")
    tk.Button(window, image=tip_img, command=tips).place(x=570, y=0)

    window.mainloop()


# run window 
main_window()
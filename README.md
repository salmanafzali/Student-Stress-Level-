# Student Stress Level Prediction
An app to let students know their stress levels

## Features
- Has a visual appearance for entering values.
- Move between pages without getting crowded.
- Using machine models to accurately predict stress levels.
- Checking values ​​and selections for correctness.
- Converting the entered values ​​to the required values ​​in the database.

## Prerequisites
Minimum requirement of Python 3 and above.
Requirement to install tkinter, scikit-learn, pandas, numpy, matplotlib libraries.

- installing libararys
pip install tkinter
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn

## how to run 
python UI.py

## Structure Project
- pycache\
- images\                                       # Folder for project images
- ├── car_main.png                                # Main page image
- ├── delete_main.png                             # Delete page image
- ├── desert-white-5120x2880-21880.jpg            # Downloaded main image
- ├── desert-white-5120x2880-21880.png            # Product import page image
- ├── mclaren-speedtail-3840x2160-23016.jpg       # Downloaded main image
- ├── predict_main.png                            # Prediction page image
- └── up_main.png                                 # Upload page image
- Backend_ml.py                                 # Project working codes and their functions
- Car F and P.csv                               # Main dataset for data mining implementation
- data_mining.ipynb                             # Data analysis and data step implementation Mining
- front.py                                      # Implement the frontend
- README.text                                   # Help file
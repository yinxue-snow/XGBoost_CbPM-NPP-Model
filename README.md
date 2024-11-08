# XGBoost_CbPM-NPP-Model
Based on the quasi-measured ocean primary productivity profiles constructed using the carbon-based NPP model and BGC-Argo measurements, an XGBoost model was trained to develop a global ocean primary productivity profile remote sensing inversion model.
1. main_train_XBGoost_CbPM.py
Train the XGBoost_CbPM NPP inversion model based on the dataset_profile.xlsx data.
2. main_apply_XGBoost_CbPM.py
Input the globally remotely sensed parameters and apply the XGBoost_CbPM model to obtain the global ocean primary productivity profiles.However, due to the input data exceeding 3 GB, it cannot be uploaded. If necessary, please contact the author to obtain it.
3.main_plot_npp.m
plot

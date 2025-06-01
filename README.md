Repository link: https://github.com/anna-k-00/Price-prediction-with-textual-data
Repository structure:
1.	embeddings generation – modules for embeddings creation. Used in test_pipeline.py (main experiment module) and predict.py (class for model deployment and prediction on new data – here used for loading).
a.	gate.py – unused module with gating nn over rubert embeddings
b.	rubert_transformer.py – module for ruBerty-tiny-2 transformer learning fully
c.	tfidf_transformer.py  - module for tfidf implementation
d.	w2v_transformer.py- module for tfidf implementation
2.	preprocessors - modules for preprocessing parsed data. Used in test_pipeline.py and predict.py
a.	preprocessor_params_hex.py – preprocessor for non-textual data. Used in both test_pipeline and predict
b.	preprocessor_text.py - preprocessor for textual data. Not integrated to test_pipeline and predict, as used for in-advanced preprocessing (less time consuming)
3.	main methods – methods for models learning, saving and deploying. Also used for practical application of models
a.	ann.py – our naïve neural network for price prediction tasks
b.	fraud_detection.py – naïve method for fraud detection with the use of predicted vs actual prices
c.	resource_monitor.py – supply method for resource monitoring during test loops to log them in MLflow
d.	test_pipeline.py – main module for testing models’ hyperparameters, fitting, assessing and saving them (in MLflow and model regitry). Also contains methods for loading models’ components in a propper way
e.	predict.py – main module for models’ deployment with following usage for prediction tasks. Interracted with test_pipeline methods for models search in MLflow and their loading
4.	experiments – folder with supplementary materials needed for experiments conduction
a.	manual_text_features_full.json – manual features dictionary used for searching Boolean parameters presented in text 
b.	manual_text_features_negative. json – same as 4.a but only has potentially strongly affecting negative keys
c.	uniques_dict.json – dictionary with unique value in categorical columns in initial dataset. Used for checking if any new values appeared in test_pipeline.py method for initial dataset validation
d.	sample_300.csv – short sample dataset which can be used for testing test_pipeline.py 
5.	samples – folder with notebooks showing how to interact with modules. Has information on how to import and use them properly. Allow repeating all the code-connected operations in this paper (including experiments evaluation) 
a.	EDA + feature importance.ipynb – notebook presenting code for chapter 3 with EDA stage. Not interacted to GoogleDrive datasets and folders directly.
b.	 parsing.ipynb - notebook presenting code for parsing new dataset. Also includes application of preprocessor_text.py before saving datasets for further usage. Attention: parser is not working from google Collaboratory. Better to use locally.
c.	experiments_5_1_results_processing.ipynb – notebook illustrating all the code for experiments in chapter 5.1 (after all the experiments have been conducted). Offers integration with our results of experiments available in GoogleDrive with all the aggregation tables creation and statistical tests conducted. The instructions on how to use GoogleDrive files are also presented there.
d.	experiments_5_2.ipynb– notebook storing all the code for experiments in chapter 5.2, including the deployment of modeule and its usage on new datasets for may and april, plus all the historical data from tests in chapter 5.1. Offers integration with our results of experiments available in GoogleDrive with all the aggregation tables creation and statistical tests conducted. The instructions on how to use GoogleDrive files are also presented there.
e.	experiments_initiation.ipynb - notebook presenting all the code for experiment initiation. Has 2 parts: one is for quick testing (small datatset, less iterations); and another one is for full reproductibility of our experiments (including parameters and datset). The instructions on how to use GoogleDrive file with full march data for tests is also available.
f.	model_deploy.ipynb notebook showing how to deploy the model with our modules and apply it to predict new prices. Fully integrated with our MLFlow GoogleDrive folder (can be accessed through Google collab) and new datasets available for tests. Includes all the instructions to access the files. Also illustrates how to apply fraud_detection.py to new prediction and shows sample results
6.	Non-folder files
a.	parser_avito.py – module for parsing 
b.	requirements.txt – file with all the python dependencis specified to fully reproduce all the code sections
c.	standard README.MD

Notably, GitHub does not support large files storage. So we also use GoogleDrive folders to organize acces for the new users to datasets for tests and to keep experiment results. All the folders can be accessed through comment mode. The intructions are available in corresponding notebooks.
Folders structure
1.	price_prediction_data https://drive.google.com/drive/folders/10uxDBjledOSIg6biJpLv6WgCMVQqzesT?usp=sharing
Folder with datasets and experiment results in non-MLFlow forms. Since the MLflow folder with a large number of runs becomes difficult to iterate through for searching runs, and Colab prohibits launching local servers to work with the MLflow UI, we duplicated the saving of key experiment files in the central pipeline for experimentation.
a.	raw_data_parsed_desc_preprocessed_full.csv – file with march dataset (with preprocessed description column to be used directly in experiment class). Data parsed in March
b.	raw_data_parsed_desc_preprocessed_full_may.csv - same as 1.a for data parsed in May 
c.	raw_data_parsed_desc_preprocessed_full_april.csv - same as 1.a for data parsed in April
d.	Results_pca – folder with main course of experiments results (chapter 5.1) in non-MLFlow format. Next is the pseudo structure for simplicity
i.	XGBR – folder with model results 
1.	metrics (store values for last runs of experiments for each combination of each feature_set x embedding techniques)
a.	cv metrics 
b.	optuna trials values
c.	predictions on validation set
d.	test metrics
e.	summary metrics (all metrics aggregated)
2.	csv files with aggregate metrics
3.	log file for experiments
ii.	ANN same as 1.d.i for another model
iii.	LinearSVR same as 1.d.i for another model
iv.	RFR same as 1.d.i for another model
e.	manual_text_features_negative same as 1.d, but for additional experiments with negative manual features with XGBoost
f.	manual_text_features same as 1.d, but for additional experiments with all manual features with XGBoost
2.	mlflow_data https://drive.google.com/drive/folders/195Ie1O3SPhwsoMSiLTUydNI-9Tf_QnU7?usp=share_link
Full mlflow folder with our experiment. Experiment coducts all the 30 experiment setups with different runs for each of then
a.	209504058848569233 – main experiment folder. Contains runs. Each run has all the artifacts needed for full reproducibility
i.	Artifacts
Artifacts 1-6 only stored for models that were fully trained (best models in our case – XGBoost and RFR)
1.	full_model  - final model pipeline 
2.	params – base model param dictionary
3.	preprocessing – pipeline for  preprocessing initial df (stores scalers, hexagon data etc)
4.	transformer – transformer pipeline for textual features process (with needed embedding generation + apply PCA stored if needed)
5.	transformer raw – without PCA or Gate application (important for weighted rubert interpretation)
6.	tokenizer – if applicable
7.	cv metrics for run
8.	predictions on validation set for run
9.	test metrics 
10.	summary
ii.	Metrics – Mlflow logged R^2, RMSE, SMAPE, MEDAPE for Optuna, CV, and validation set
iii.	Params - Mlflow logged parameters including
1.	Models parameters (also with PCA or Gate if applicable)
2.	Preprocessing parameters
3.	Optuna parameters
4.	 Environmental parameters
iv.	Taggs - Mlflow logged tags with additional information (final R^2, paths, time, names etc.)
b.	models – folder with all the models stored via ModelRegistry. Has folders  for each model (unique parameter combination in name) and contains the versions connected with MLFlow runs for export
 

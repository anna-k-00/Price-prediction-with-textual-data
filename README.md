# Price-prediction-with-textual-data

Code implementation procedure

# 1. Parsing

  from parser_avito import AvitoParser
  
  example of usage
  parser = AvitoParser()
  parser.initializer(target_types = ['dom','dacha',cottage],df_path='data_new_house.csv') 

# 2. Preprocessing
  
  Standard params of the objects:
  
    from preprocessors.preprocessor_params import DataProcessingPipeline
      
      example of usage
      process = DataProcessingPipeline(df,
                                   log_needed = True,
                                   norm_needed = True,
                                   one_hot_only = False)
      df_params = df.process.process_for_ml()

  Textual preprocessing
    from preprocessors.preprocessor_text import  TextPreprocessor
      
      example of usage
      process = TextPreprocessor(df,
                                  text_columns = ['description'], 
                                 lemmatize_text = False, 
                                 remove_stopwords = False,
                                 remove_punctuation = False,
                                 ) #For BERT
      df_text = df.process.process()
  
  #set params of preprocessing depending on model and embedding generation technique
  
# 3. Model testing with fixed feature set (text-only, parameter-only and mixed), model and embedding tecnique across embedding and model hyperparameters

 initialize testing with script from alike experiment_initiation_rf or similar ones

 by importing these methods and calling final initiation loop (or single experiment) depending on model chosen and parameters to test for this model
  
  from embeddings_generation.tfidf_transformer import TfidfTransformer
  from embeddings_generation.w2v_transformer import Word2VecTransformer
  from embeddings_generation.rubert_transformer import RuBertTiny2Embedder
  from exp_setting import ANNRegressor
  from exp_setting import PricePredictionExperiment

  
 models available for testing for now:
     - NN (custom one)
     - XGBoost
     - LSVR 
     - RFR
  new "out of the box" models can be inserted into the class with simple adding them into class initiation function and passing right parameters while calling test method
  some custom methods (like fine-tuning) need to be carefully designed before adding them to prevent leakage of train data wgile performing CV in main test loop

  
  embedding generation techniques available for testing for now:
    - fine-tuned RuBERT tiny
    - Word2Vec
    - TF-IDF
  their hyperparameters are fixed inside the experiment method

  ### automatic logging with log and MLFlow is initiated inside the testing method ##
  so you may want to start MLFlow local server before calling the method :) 

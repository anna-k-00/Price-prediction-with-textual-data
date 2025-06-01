# Repository Structure

## 📁 Main Directories

### 1. `embeddings_generation/` - Modules for embeddings creation
- `gate.py` - Unused module with gating NN over RuBERT embeddings  
- `rubert_transformer.py` - RuBERT-tiny-2 transformer implementation  
- `tfidf_transformer.py` - TF-IDF vectorizer  
- `w2v_transformer.py` - Word2Vec implementation  

### 2. `preprocessors/` - Data preprocessing modules
- `preprocessor_params_hex.py` - Non-textual data preprocessor (used in main pipeline)  
- `preprocessor_text.py` - Textual data preprocessor (for advanced preprocessing)  

### 3. `main_methods/` - Core functionality
- `ann.py` - Custom neural network for price prediction  
- `fraud_detection.py` - Fraud detection using predicted vs actual prices  
- `resource_monitor.py` - Resource monitoring during experiments  
- `test_pipeline.py` 🚀 - Main module for model training/evaluation  
- `predict.py` 🚀 - Model deployment and prediction  

### 4. `experiments/` - Supplementary materials
- `manual_text_features_*.json` - Dictionaries for text feature extraction  
- `uniques_dict.json` - Unique values for categorical columns validation  
- `sample_300.csv` - Sample dataset for testing  

### 5. `samples/` - Example notebooks
| Notebook | Purpose |
|----------|---------|
| `EDA + feature importance.ipynb` | Chapter 3 EDA |
| `parsing.ipynb` | Dataset parsing (local execution recommended) |
| `experiments_5_1_results_processing.ipynb` | Chapter 5.1 experiments analysis |
| `experiments_5_2.ipynb` | Chapter 5.2 deployment and testing |
| `experiments_initiation.ipynb` | Experiment setup (quick/full modes) |
| `model_deploy.ipynb` | Model deployment guide |

## 📄 Root Files
- `parser_avito.py` - Data parsing module  
- `requirements.txt` - Python dependencies  
- `README.md` - This documentation  

## 🔗 External Resources
- [Google Drive: Datasets & Results](https://drive.google.com/drive/folders/10uxDBjledOSIg6biJpLv6WgCMVQqzesT)  
- [Google Drive: MLflow Experiments](https://drive.google.com/drive/folders/195Ie1O3SPhwsoMSiLTUydNI-9Tf_QnU7)  

## 🛠️ Usage
```bash
git clone https://github.com/your-repo.git
pip install -r requirements.txt
   

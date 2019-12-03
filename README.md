# Paper

Our paper's information can be found here:

* Title: LogTransfer: Robust Log-based Anomaly Detection for Emerging Services
* Authors: 
* Paper link: [LogTransfer](https://)

## Requirements
    python 3.6.8
    kreas 2.2.4 
    tensorflow 1.9.0
    numpy 1.16.4
    scikit-learn 0.21.2 

## Project Main Structure

    ├── checkpoint              # Model_checkpoint
        ├── model               # Init train model's checkpoint 
        ├── transfer            # Transfer model's checkpoint 
    ├── codes                   # Code
    ├── data_processing         # Data processing code
    ├── labels                  # Label files dir
    ├── Logs                    # Training data files dir
    ├── record                  # Record dir(for backtracking errors)
    ├── result                  # Predict results dir   
    ├── README.md               # README file

## Test easily (with default parameters)
```shell 
git clone https://github.com/logtransfergit/LogTransfer.git
cd LogTransfer
pip install -r requirements.txt
cd codes
bash test.sh
```

## How to use single file:


### ./codes/data_keyword_remove.py
  * Removing logs which containing keywords and the corresponding labels.
  * Script example: `python data_keyword_remove.py --rootdir ../Logs --label_dir ../labels --removal log_in`
  * | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rootdir` | str | `'../Logs'` | log data dir|
| `label_dir` | str | `'../labels'` | label data dir |
| `removal` | str | `r'logined the switch|logouted from the switch|logout from|logged in from|login from |logged out from|Service=login-UserName'` | keywords to be detect |

### ./codes/models.py
  * Initially training a model with abundant labeled data.
  * Script example: `python models.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/S5/ --vector_path ../Logs/template_vec.dat --selected_switchid_list [30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50]`
  * | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `window_size` | int | `10` | the length of input |
| `single_step`|  int | `1` | the step span when choosing train data |
| `rootdir` | str | `'../Logs'` | train data dir |
| `label_dir` | str | `'../labels/'` | label data dir |
| `vector_path` | str | `'../Logs/template_vec.dat'` | pretrained template vector |
| `selected_switchid_list` | list | `[]` | train data file lists |

### ./codes/transferlearning.py
  * Revising the trained model with a small amount of specific data.
  * Script example: `python transferlearning.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/B6220/ --vector_path ../Logs/template_vec.dat  --selected_switchid_list [1,2,3,4,5]`
  * | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `window_size` | int | `10` | the length of input |
| `single_step`| int |`1`| the step span when choosing train data |
| `rootdir` | str |`'../Logs'`| train data dir|
| `label_dir`| str | `'../labels/B6220/'`  | label data dir |
| `vector_path`| str | `'../Logs/template_vec.dat'` | pretrained template vector |
| `selected_switchid_list` | list | `[]` | train data file lists|

### ./codes/predict.py
  * Predicting whether anomalies with trained models and evaluating predictions.
  * Script example: `python predict.py --single_step 3 --window_size 20 --rootdir ../Logs --label_dir ../labels/B6220/ --vector_path ../Logs/template_vec.dat  --selected_switchid_list [6,7,0] --save_model_dir ../checkpoint/transfer/ --save_result_dir ../result2/ --mode predict`
  * | Argument | Type | Default | Description |
|--------------------|------|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `window_size` | int | `10` | the length of input |
| `single_step` | int |`1` | the step span when choosing train data |
| `rootdir` | str | `'../Logs'` | train data dir|
| `label_dir` | str | `'../labels/B6220/'`  | label data dir |
| `vector_path` | str | `'../Logs/template_vec.dat'` | pretrained template vector |
| `selected_switchid_list` | list | `[]` | train data file lists |
| `save_model_dir` | str |`'../checkpoint/transfer/'`|  model checkpoint dir |
| `save_result_dir` | str |`'../result/'` | save result dir |
| `mode` | str | `'predict'` | choose wether to predict or calculate performance |

### ./codes/utils.py
  * Some tool functions.
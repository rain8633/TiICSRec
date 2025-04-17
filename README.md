# TiICSRec

This is our Pytorch implementation for the paper: "**Time Interval-Enhanced Intent Contrastive Learning for Sequential Recommendation**".

## Environment  Requirement

* Pytorch>=1.7.0
* Python>=3.7  

## Usage

Please run the following command to install all the requirements:  

```python
pip install -r requirements.txt
```

## Evaluate Model

We provide the trained models on Beauty, Sports datasets in `./src/output/<Data_name>`folder. You can directly evaluate the trained models on test set by running:

```
python main.py --data_name <Data_name> --model_idx 0 --do_eval --encoder SAS
```

On Beauty:

```python
python main.py --data_name Beauty --model_idx 0 --do_eval --encoder SAS
```

```
{'Epoch': 0, 'HIT@5': '0.0650', 'NDCG@5': '0.0455', 'HIT@10': '0.0946', 'NDCG@10': '0.0550', 'HIT@20': '0.1329', 'NDCG@20': '0.0657'}
```

On Sports_and_Outdoors:

```python
python main.py --data_name Sports_and_Outdoors --model_idx 0 --do_eval --encoder SAS
```

```
{'Epoch': 0, 'HIT@5': '0.0401', 'NDCG@5': '0.0275', 'HIT@10': '0.0598', 'NDCG@10': '0.0339', 'HIT@20': '0.0863', 'NDCG@20': '0.0406'}
```



## Train Model

Please train the model using the Python script `main.py`.

You can run the following command to train the model on Beauty datasets:

```
python main.py --data_name Beauty --rec_weight 1. --lambda_0 0.3 --beta_0 0.1 --f_neg --intent_num 256 
```
or

You can use the training scripts in the `./src/scrips` folder to train the model 
```angular2html
bash beauty.sh
bash sports.sh
```
## Acknowledgment

- Transformer and training pipeline are implemented based on [TiCoSeRec](https://github.com/KingGugu/TiCoSeRec). Thanks them for providing efficient implementation.

```
- Please kindly cite our paper if this paper and the code are helpful. 


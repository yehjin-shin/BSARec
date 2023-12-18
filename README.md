# Code for AAAI_BSARec
## 1. Install conda environments 

```
conda env create -f bsarec_env.yaml
```

## 2. Train BSARec
Note that pretrained model (.pt) and train log file (.log) will saved in `BSARec/output`
### (1) How to train
- `train_name`: name for log file and checkpoint file
```
python main.py  --data_name [DATASET] \
                --lr [LEARNING_RATE] \
                --alpha [ALPHA] \ 
                --c [C] \
                --num_attention_heads [N_HEADS] \
                --train_name [LOG_NAME]
```
### (2) Example: Beauty
```
python main.py  --data_name Beauty \
                --lr 0.0005 \
                --alpha 0.7 \
                --c 5 \
                --num_attention_heads 1 \
                --train_name BSARec_Beauty
```
### (3) Example: LastFM
```
python main.py  --data_name LastFM \
                --lr 0.001 \
                --alpha 0.9 \
                --c 3 \
                --num_attention_heads 1 \
                --train_name BSARec_LastFM
```

## 3. Test pretrained BSARec
Note that pretrained model (.pt file) must be in `BSARec/output`
### (1) How to test pretrained model
- `load_model`: pretrained model name without .pt
```
python main.py  --data_name [DATASET] \
                --alpha [ALPHA] \ 
                --c [C] \
                --num_attention_heads [N_HEADS] \
                --load_model [PRETRAINED_MODEL_NAME] \
                --do_eval
```
### (2) Beauty
```
python main.py  --data_name Beauty \
                --alpha 0.7 \
                --c 5 \
                --num_attention_heads 1 \
                --load_model BSARec_Beauty_best \
                --do_eval
```
### (3) LastFM
```
python main.py  --data_name LastFM \
                --alpha 0.9 \
                --c 3 \
                --num_attention_heads 1 \
                --load_model BSARec_LastFM_best \
                --do_eval
```

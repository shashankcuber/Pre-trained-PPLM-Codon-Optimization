#!/bin/bash
python3 prot_bert_train.py 
# / --cai_threshold=0.8 --stability_threshold=-0.4 
/ --cai_type='mse' --mask='True' 
/ --stability_type='mfe' --alpha=0.5 
/ --tool_pkg='vienna' --temperature=37

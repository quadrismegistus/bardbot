import gpt_2_simple as gpt2
import tensorflow as tf
import os
import requests
import sys
sys.path.insert(0,'/Users/ryan/github/prosodic')
import prosodic as p
from collections import Counter

p.config['print_to_screen']=0

PATH_HERE=os.path.abspath(os.path.dirname(__file__))
PATH_DATA=os.path.join(PATH_HERE,'data')
PATH_MODELS=os.path.join(PATH_DATA,'models')
PATH_CHKP=os.path.join(PATH_DATA,'checkpoints')
PATH_TEXTS=os.path.join(PATH_DATA,'texts')
PATH_SHAKS=os.path.join(PATH_TEXTS,'sonnets.txt')
PATH_SAMPLES=os.path.join(PATH_DATA,'samples')

MODEL_NAME='124M'
CHECKPOINT='latest'
RUN_NAME='run1'



MODEL=None
TOKENIZER=None

def get_tokenizer():
    global TOKENIZER
    if TOKENIZER is None:
        from transformers import GPT2Tokenizer
        TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
    return TOKENIZER

def get_model():
    global MODEL
    if MODEL is None:
        print('Loading GPT2 model')
        from transformers import GPT2LMHeadModel
        tokenizer = get_tokenizer()
        MODEL = GPT2LMHeadModel.from_pretrained('gpt2',pad_token_id=tokenizer.eos_token_id)
    return MODEL


def generate(
        s,
        max_new_tokens=10,
        top_k=50, 
        top_p=0.95,
        **kwargs
    ):
    tokenizer = get_tokenizer()
    inputs = tokenizer(s, return_tensors="pt")
    
    model=get_model()
    generation_output = model.generate(
        **inputs,
        do_sample=True, 
        # max_length=max_length, 
        top_k=top_k, 
        top_p=top_p, 
        num_return_sequences=1,
        max_new_tokens=max_new_tokens
    )
    o=tokenizer.decode(generation_output[0])
    #o=[tokenizer.decode(x) for x in generation_output[0]]
    return o
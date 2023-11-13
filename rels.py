import os
import nltk
import spacy
import torch
from nltk.tokenize import sent_tokenize
from fastcoref import FCoref
coref_model = FCoref()
checkpoint = "flax-community/t5-base-wikisplit"
# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
tokenizer = AutoTokenizer.from_pretrained("ibm/knowgl-large")
model = AutoModelForSeq2SeqLM.from_pretrained("ibm/knowgl-large")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

nlp = spacy.load('en_core_web_sm')

def replace_text(text, name):
  text = re.sub(r"We are", name +" is", text) 
  text = re.sub(r"We have", name + " has", text)
  text = re.sub(r" we ", " " + name + " ", text) 
  text = re.sub(r" our ", " " + name + "'s ", text)
  text = re.sub(r" us\.", " " + name + ".", text)
  text = re.sub(r"Our ", name + " ", text)
  text = re.sub(r"the company", name, text)
  text = re.sub(r"The Company", name, text) 
  text = re.sub(r"the Company", name, text)

  return text

def rels(text, i):
    outputs=[]
    
    doc=nlp(text)

    sentences = [sent.text for sent in doc.sents]
    for sentence in sentences:
        print('sentence: ' + sentence)
        i=i+1
        inputs = tokenizer([sentence], truncation=True,  return_tensors="pt")
        summary = model.generate(**inputs)
        output=tokenizer.decode(summary[0])
        print(output)
        outputs.append(output)
        # Open file for writing 

   
        with open('extracted_relations_5.txt', 'a') as f:
            f.writelines(outputs) 

folder = 'filing_text'
i=0

for file_name in sorted(os.listdir(folder), reverse=True):
    file_path = os.path.join(folder, file_name)
    ##print(file_path)
    name = file_name.split(' (')[0]
    if 'Item1_' in file_name:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            i=i+1
            text = replace_text(text, name)
            
            output = rels(text, i)
            print('processing: ' + file_name)

#find out a way to get company name with filing
#product entity

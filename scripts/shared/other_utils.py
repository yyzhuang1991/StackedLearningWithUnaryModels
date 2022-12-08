import sys, os, re
from os.path import exists 

def read_inlist(file):
    with open(file, "r") as f:
        lines = f.read().strip().split("\n")
    files = [] 
    for line in lines:
        line = line.strip()
        if line == '': continue
        ann_file, text_file, corenlp_file = line.split(",")
        if not ann_file.endswith(".ann"):
            raise NameError(f'Invalid ann file: {ann_file}')
        if not text_file.endswith(".txt"):
            raise NameError(f"Invalid text file: {text_file}")
        if not corenlp_file.endswith(".txt.json"):
            raise NameError(f"Invalid corenlp file: {corenlp_file}")
        files.append((ann_file, text_file, corenlp_file))

    ann_files, text_files, corenlp_files = zip(*files)

    for k in ann_files + text_files + corenlp_files:
        if not exists(k):
            raise NameError(f"FILE does not exists: {k}")

    return ann_files, text_files, corenlp_files

def add_marker_tokens(tokenizer, ner_labels):

    new_tokens = []
    for label in ner_labels:
        print(f"Adding token to tokenizer: <ner_start={label.lower()}>")
        print(f"Adding token to tokenizer: <ner_end={label.lower()}>")
        new_tokens.append('<ner_start=%s>'%label.lower())
        new_tokens.append('<ner_end=%s>'%label.lower())
        new_tokens.append('<%s>'%label.lower())

    tokenizer.add_tokens(new_tokens)
    print("Samplec Output from tokenizer:")
    for label in ner_labels:
        ids = tokenizer.convert_tokens_to_ids(new_tokens)
        print(ids)
        print(tokenizer.convert_ids_to_tokens(ids))
    print("------")


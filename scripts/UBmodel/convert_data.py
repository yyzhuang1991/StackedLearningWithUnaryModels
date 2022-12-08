import re, pickle, sys, argparse, random, json, os
from sys import stdout
from copy import deepcopy 
from os.path import abspath, dirname, join, exists
from os import makedirs, listdir

sharedpath = "../shared"
sys.path.insert(0, sharedpath)

from structures import Relation, Span, MteDocument
from extraction_utils import get_docid
from other_utils import read_inlist, add_marker_tokens

sys.path.append("../")
from config import mte_validRelationSet
from transformers import *


def make_unary_instances(annFiles, textFiles, corenlpFiles, validRelationSet, outdir, relation, ner, maxLen = 512):

    """
    This function extract unary instances from files for training and evaluation

    Args:
        ann_files: 
            a list of ann files ( used to assign gold labels (binary label for whether the instance is a Container instance) to target instances)
        text_files:
            a list of text files where target instances would be extracted 
        corenlp_files:
            a list of files that store the parsing results of text files from CoreNLP.
        outdir:
            output directory to save the extracted instances 
        max_len:
            an integer to indicate the maximum number of tokens to keep for a sentence after bert tokenization (bert allows a list of input tokens with <= 512 tokens )
        is_training:
            boolean to indicate whether the extracted instances are used for training or not. 
        use_sys_ners: 
            boolean to indicate whether system ners instead of gold ners should be used to extract the target instances. 

    """
    if len(validRelationSet) != 1:
        raise NameError(f"Only support 1 relation in validRelationSet now! Got {validRelationSet}")

    otherValidNers = set() # store other argument types for 'ner' and 'relation'
    for ner1, ner2, tempRelation in validRelationSet:
        if tempRelation == relation: 
            if ner1 == ner:  
                otherValidNers.add(ner2)
            if ner2 == ner:
                otherValidNers.add(ner1)

    docid2files = {}
    for textFile, annFile, corenlpFile in zip(textFiles, annFiles, corenlpFiles):
        year, docname,docid = get_docid(annFile)
        if docid not in docid2files:
            docid2files[docid] = [] 
        docid2files[docid].append((textFile, annFile, corenlpFile))
    inferInstances = [] # for inference
    evalInstances = [] # for eval  
    for i, docid in enumerate(docid2files):
        stdout.write(f"\rMaking {i}/{len(docid2files)} documents")
        stdout.flush()
        textFiles, annFiles, corenlpFiles = list(zip(*docid2files[docid]))
        if len(annFiles) > 1:
            texts = [open(t).read() for t in textFiles]
            if len(set(texts)) > 1:
                # if their texts are not aligned, I have no good idea to align them. So I just pick one. Here I give non-mer-a priority 
                annFile = None 
                for a, t, c in zip(annFiles, textFiles, corenlpFiles):
                    if 'mer-a' not in a:
                        annFile = a
                        textFile = t
                        corenlpFile = c 
                if annFile is None:
                    annFile = annFiles[0]
                    textFile = textFiles[0]
                    corenlpFile = corenlpFiles[0]
                annFiles, textFiles, corenlpFiles = [annFile], [textFile], [corenlpFile] 

        doc = MteDocument(annFiles, textFiles, corenlpFiles, validRelationSet)
        seenInferIds = set()
        seenEvalIds = set()
        sentId2inRelationStdtext = {} # for training
        tempInferInstances = [] 
        tempEvalData = [] 
        for ner1, ner2, tempRelation in validRelationSet:
            if tempRelation != relation:
                continue
            if ner != ner1 and ner != ner2:
                continue  

            # construct unary instances to infer
            for binaryRelation in doc.relation2inferBinaryRelations.get(tempRelation, []):
                u1, u2 = binaryRelation.span1, binaryRelation.span2 # u1 and u2 have already had its gold labels for being in unary relations or not 
                if binaryRelation.relationLabel != 'O':
                    if u1.sentId not in sentId2inRelationStdtext:
                        sentId2inRelationStdtext[u1.sentId] = set()
                    sentId2inRelationStdtext[u1.sentId].add(u1.stdText)
                    sentId2inRelationStdtext[u1.sentId].add(u2.stdText)

                if u1.ner == ner and u1.id not in seenInferIds:      
                    # u1.insert_type_markers(tokenizer)
                    tempInferInstances.append(u1)
                    seenInferIds.add(u1.id)
                    
                if u2.ner == ner and u2.id not in seenInferIds:      
                    # u2.insert_type_markers(tokenizer)
                    tempInferInstances.append(u2)
                    seenInferIds.add(u2.id)
                    
            # construct eval data 
            for u in doc.relation2evalUnaryInstances.get(tempRelation, []):
                if u.ner != ner or u.id in seenEvalIds:
                    continue 
                seenEvalIds.add(u.id)  
                evalInstances.append(u)

        for u in tempInferInstances:
            if u.relationLabel == 'O' and u.stdText in sentId2inRelationStdtext.get(u.sentId, []): # negative training instance
                u.haveNoisyLabel = True
            else:
                u.haveNoisyLabel = False
        inferInstances.extend(tempInferInstances)

    print(f"Collected {len(inferInstances)} instances for inference, {len(evalInstances)} for evaluation")
    numPos = len([1 for u in inferInstances if u.relationLabel != 'O'])
    numNeg = len(inferInstances) - numPos
    print(f"Collected {len(inferInstances)} unary instances for training and evaluation.")

    if not exists(outdir):
        os.makedirs(outdir)

    outfile = join(outdir, f"inferInstances.pkl")
    print(f"Saving to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(inferInstances, f)

    outfile = join(outdir, f"evalInstances.pkl")
    print(f"Saving to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(evalInstances, f)



def make_binary_instances(annFiles, textFiles, corenlpFiles, validRelationSet, outdir, relation, ner1, ner2, maxLen = 512):
    if len(validRelationSet) != 1:
        raise NameError(f"Only support 1 relation in validRelationSet now! Got {validRelationSet}")
    assert (ner1, ner2, relation) in validRelationSet

    docid2files = {}
    for textFile, annFile, corenlpFile in zip(textFiles, annFiles, corenlpFiles):
        year, docname,docid = get_docid(annFile)
        if docid not in docid2files:
            docid2files[docid] = [] 
        docid2files[docid].append((textFile, annFile, corenlpFile))
    inferInstances = [] # for inference
    evalInstances = [] # for eval  
    for i, docid in enumerate(docid2files):
        stdout.write(f"\rMaking {i}/{len(docid2files)} documents")
        stdout.flush()
        textFiles, annFiles, corenlpFiles = list(zip(*docid2files[docid]))
        if len(annFiles) > 1:
            texts = [open(t).read() for t in textFiles]
            if len(set(texts)) > 1:
                annFile = None 
                for a, t, c in zip(annFiles, textFiles, corenlpFiles):
                    if 'mer-a' not in a:
                        annFile = a
                        textFile = t
                        corenlpFile = c 
                if annFile is None:
                    annFile = annFiles[0]
                    textFile = textFiles[0]
                    corenlpFile = corenlpFiles[0]
                annFiles, textFiles, corenlpFiles = [annFile], [textFile], [corenlpFile] 
        doc = MteDocument(annFiles, textFiles, corenlpFiles, validRelationSet)
        seenInferIds = set()
        seenEvalIds = set()
        sentId2inRelationStdPairs = {} # for training
        for tempNer1, tempNer2, tempRelation in validRelationSet:
            if tempRelation != relation:
                continue
            if tempNer1 != ner1 and tempNer2 != ner2:
                continue  

            # construct eval data 
            for r in doc.relation2evalBinaryRelations.get(tempRelation, []):
                if r.id in seenEvalIds:
                    continue 
                seenEvalIds.add(r.id)  
                evalInstances.append(r)
                sentId = r.span1.sentId 
                if sentId not in sentId2inRelationStdPairs:
                    sentId2inRelationStdPairs[sentId] = set()
                sentId2inRelationStdPairs[sentId].add((r.span1.stdText, r.span2.stdText)) 

            # construct binary instances to infer
            for r in doc.relation2inferBinaryRelations.get(tempRelation, []): 
                if r.id in seenInferIds:
                    continue 
                seenInferIds.add(r.id)

                if r.relationLabel == 'O' and (r.span1.stdText, r.span2.stdText) in sentId2inRelationStdPairs.get(r.span1.sentId, []):
                    r.haveNoisyLabel = True
                else:
                    r.haveNoisyLabel = False
                inferInstances.append(r)

                    

    print(f"Collected {len(inferInstances)} instances for inference, {len(evalInstances)} for evaluation")
    numPos = len([1 for u in inferInstances if u.relationLabel != 'O'])
    numNeg = len(inferInstances) - numPos
    print(f"Collected {len(inferInstances)} unary instances for training and evaluation.")

    if not exists(outdir):
        os.makedirs(outdir)

    outfile = join(outdir, f"inferInstances.pkl")
    print(f"Saving to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(inferInstances, f)

    outfile = join(outdir, f"evalInstances.pkl")
    print(f"Saving to {outfile}")
    with open(outfile, "wb") as f:
        pickle.dump(evalInstances, f)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required = True, choices = ['binary', 'unary', 'both'])
    parser.add_argument('--relation', required = True)
    parser.add_argument('--trainList', required = True, help = "input list of files for TRAIN data. Each line is in the format of '<ann_file>,<text_file>,<corenlp_file>' where <corenlp_file> is the json file that stores the parsing results of the text file using CoreNLP. ")
    parser.add_argument('--devList', required = True, help = "input list of files for DEV data. It is in the same format as trainList ")
    parser.add_argument('--testList', required = True, help = "input list of files for TEST data.  It is in the same format as trainList ")
    parser.add_argument('--outdir', required = True)
    parser.add_argument('--maxLen', type = int, default = 512, help = "maximum number of tokens in a sentence encoded by BERT")
    args = parser.parse_args()
    return args 

def main():
    args = get_parser()
    if not exists(args.outdir):
        makedirs(args.outdir)
    print(f"MODE: {args.mode}")
    validRelationSet = mte_validRelationSet[args.relation]

    makeUnary = args.mode == 'both' or args.mode == 'unary'
    makeBinary = args.mode == 'both' or args.mode == 'binary'
    ner1, ner2, relation = list(validRelationSet)[0]
    for inList, name in zip([args.trainList, args.devList, args.testList], ['train','dev','test']):
        annFiles, textFiles, corenlpFiles = read_inlist(inList)
        if makeUnary:
            for ner in [ner1, ner2]:
                unaryName = f"{ner}-{relation}"
                print(f"UNARY {unaryName}: Making {name} data from {len(annFiles)} files")
                make_unary_instances(annFiles, textFiles, corenlpFiles, validRelationSet, join(args.outdir, unaryName, name), relation, ner, maxLen = args.maxLen)
        if makeBinary:
            seenRelations = set()
            relationName = f"{ner1}-{ner2}-{relation}"
            if relation in seenRelations: continue 
            seenRelations.add(relation)
            make_binary_instances(annFiles, textFiles, corenlpFiles, validRelationSet, join(args.outdir, relationName, name ), relation, ner1, ner2, maxLen = args.maxLen)
            
if __name__ == "__main__":
    main()

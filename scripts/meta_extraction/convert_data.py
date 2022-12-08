import re, pickle, sys, argparse, random, json, os
from sys import stdout
from copy import deepcopy 
from os.path import abspath, dirname, join, exists
from os import makedirs, listdir
sys.path.insert(0, "../shared")
from structures import Relation, Span, MteDocument
from extraction_utils import get_docid
from other_utils import read_inlist, add_marker_tokens
sys.path.append("../")
from config import mte_validRelationSet


def get_feature_info(tempInferInstances, doc, unaryId2preds, binaryId2preds):
    # get sentid2entites
    sentId2spans = {}

    for e in doc.entities:
        sentId = e['sentid']
        if sentId not in sentId2spans:
            sentId2spans[sentId] = [] 
        sentId2spans[sentId].append(Span(doc.docId, e['doc_start_char'], e['doc_end_char'], e['text'], e['std_text'], e['label'], sentStartIdx = e['sent_start_idx'],sentEndIdx = e['sent_end_idx']))

    for sentId in sentId2spans:
        for u in sentId2spans[sentId]:
            if u.id in unaryId2preds: # some span may not be in unaryId2preds, since they do not have cooccurrence with other arguments
                u.predScore, u.predRelationLabel, _ = unaryId2preds[u.id]

    for r in tempInferInstances:
        r.entities = sentId2spans[r.span1.sentId]

        for u in [r.span1, r.span2]:
            u.predScore, u.predRelationLabel, _ = unaryId2preds[u.id]
        r.predScore, r.predRelationLabel, tempR = binaryId2preds[r.id]
        r.decodedInput = tempR.decodedInput


def make_instances(annFiles, textFiles, corenlpFiles, validRelationSet, outdir, relation, ner1, ner2, unaryId2preds, binaryId2preds):
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
            tempInferInstances = [] 
            for r in doc.relation2inferBinaryRelations.get(tempRelation, []): 
                if r.id in seenInferIds:
                    continue 
                seenInferIds.add(r.id)
                if r.relationLabel == 'O' and (r.span1.stdText, r.span2.stdText) in sentId2inRelationStdPairs.get(r.span1.sentId, []):
                    r.haveNoisyLabel = True
                else:
                    r.haveNoisyLabel = False
                tempInferInstances.append(r)

            get_feature_info(tempInferInstances, doc, unaryId2preds, binaryId2preds)
            for ins in tempInferInstances:
                for span in ins.entities:
                    assert hasattr(span, 'sentStartIdx')
            inferInstances.extend(tempInferInstances)

    print(f"Collected {len(inferInstances)} instances for inference, {len(evalInstances)} for evaluation")
    numPos = len([1 for u in inferInstances if u.relationLabel != 'O'])
    numNeg = len(inferInstances) - numPos
    numNoisyNeg = len([1 for u in inferInstances if u.haveNoisyLabel])
    print(f"Collected {len(inferInstances)} unary instances for training and evaluation. Among them, {numNoisyNeg} have noisy 'O' labels. Excluding the noisy labels, there are {numPos}({numPos/(numPos + numNeg - numNoisyNeg)*100:.2f}) POS and {numNeg}({numNeg/(numPos + numNeg - numNoisyNeg)*100:.2f}) NEG instances")

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
    parser.add_argument('--relation', required = True, help = 'relation type')

    parser.add_argument('--trainList', required = True, help = "input list of files for TRAIN data. Each line is in the format of '<ann_file>,<text_file>,<corenlp_file>' where <corenlp_file> is the json file that stores the parsing results of the text file using CoreNLP. ")
    parser.add_argument('--trainU1Predictions', required = True)
    parser.add_argument('--trainU2Predictions', required = True)
    parser.add_argument('--trainBinaryPredictions', required = True)
    parser.add_argument('--devList', required = True, help = "input list of files for DEV data. It is in the same format as trainList ")
    parser.add_argument('--devU1Predictions', required = True)
    parser.add_argument('--devU2Predictions', required = True)

    parser.add_argument('--devBinaryPredictions', required = True)
    parser.add_argument('--testList', required = True, help = "input list of files for TEST data.  It is in the same format as trainList ")
    parser.add_argument('--testU1Predictions', required = True)
    parser.add_argument('--testU2Predictions', required = True)
    parser.add_argument('--testBinaryPredictions', required = True)
    parser.add_argument('--outdir', required = True)
    args = parser.parse_args()
    return args 

def make_prediction_map(predFiles, binary = False):
    id2pred = {} 
    count = 0 
    for file in predFiles:
        with open(file, 'rb') as f:
            preds = pickle.load(f)
            for p in preds:
                # this is one special case where the entitiy is tagged with Target and Property. It should be removed if it exists in the dataset
                if p.id == '2005_2269-5083-5087' and isinstance(p, Span) and p.ner == 'Property':
                    continue
                assert p.id not in id2pred
                id2pred[p.id] = [p.predScore, p.predRelationLabel, p]
    return id2pred

def main():
    args = get_parser()
    if not exists(args.outdir):
        makedirs(args.outdir)
    validRelationSet=mte_validRelationSet[args.relation]
    ner1, ner2, relation = list(validRelationSet)[0]
    unaryId2preds = {}
    unaryId2preds['train'] = make_prediction_map([args.trainU1Predictions, args.trainU2Predictions])
    unaryId2preds['dev'] = make_prediction_map([args.devU1Predictions, args.devU2Predictions])
    unaryId2preds['test'] = make_prediction_map([args.testU1Predictions, args.testU2Predictions])
    binaryId2preds = {}
    binaryId2preds['train'] = make_prediction_map([args.trainBinaryPredictions])
    binaryId2preds['dev'] = make_prediction_map([args.devBinaryPredictions])
    binaryId2preds['test'] = make_prediction_map([args.testBinaryPredictions])

    for inList, name in zip([args.trainList, args.devList, args.testList], ['train','dev','test']):
        annFiles, textFiles, corenlpFiles = read_inlist(inList)
        seenRelations = set()
        relationName = f"{ner1}-{ner2}-{relation}"
        if relation in seenRelations: continue 
        seenRelations.add(relation)
        make_instances(annFiles, textFiles, corenlpFiles, validRelationSet, join(args.outdir, relationName, name ), relation, ner1, ner2, unaryId2preds[name], binaryId2preds[name])
if __name__ == "__main__":
    main()

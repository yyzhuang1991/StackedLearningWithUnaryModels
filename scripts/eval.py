import pmetrics, argparse, json, numpy as np 

def eval(predSigns, goldSigns):
    # eval binary relations
    predSignSet = set(predSigns)
    goldSignSet = set(goldSigns)
    numCorrect = len(predSignSet.intersection(goldSignSet))
    recall = numCorrect / len(goldSignSet)
    precision = numCorrect / len(predSignSet) if len(predSignSet) else 0 
    f1 = 2 * recall * precision / (precision + recall) if precision + recall != 0 else 0 
    return precision, recall, f1


def eval_chemprot(intPreds, intGolds, classes):
    # adopted from https://github.com/ncbi-nlp/BLUE_Benchmark/blob/master/blue/eval_rel.py
    assert len(intPreds) == len(intGolds)
    preds = intPreds
    golds = intGolds

    result = pmetrics.classification_report(golds, preds, macro=False, micro=True, classes_=classes)
    print(result.report)
    subindex = [i for i in range(len(classes)) if classes[i] != 'O']
    result = result.sub_report(subindex, macro=False, micro=True)
    print(result.report)
    return result.report
    
def eval_scibert_output(inFile, labelFile):
    classes = [] 
    with open(labelFile) as f:
        for line in f:
            line = line.strip() 
            if line == "": continue 
            classes.append(line)

    with open(inFile) as f:
        id2intGold = {}
        id2intPred = {}
        for line in f:
            obj = json.loads(line.strip())
            pairId, predIntLabel, orgIntLabel, classProbs = obj['metadata']['pairId'], np.argmax(obj['class_probs']), int(obj['orgLabel']), obj['class_probs']
            id2intGold[pairId] = orgIntLabel
            id2intPred[pairId] = predIntLabel
    intPreds = [] 
    intGolds = [] 
    for id_ in id2intGold.keys():
        intPreds.append(id2intPred[id_])
        intGolds.append(id2intGold[id_])
    eval_chemprot(intPreds, intGolds, classes)

def eval_linkbert_output(linkbertPredFile):
    intPreds = [] 
    intGolds = [] 
    class2id = None
    classes = None
    with open(linkbertPredFile) as f:
        for line in f:
            line = line.strip()
            if line == "": continue 
            obj = json.loads(line)
            label = obj['label']
            if label == '0': label = 'O'
            classProbs = obj['class_probs']
            if not class2id:
                class2id = {}
                classes = sorted(classProbs.keys())
                for class_ in classes: 
                    class2id[class_] = len(class2id)

            listProbs = [classProbs[class_] for class_ in classes]
            predIntLabel = np.argmax(listProbs)
            orgIntLabel = class2id[label]
            intPreds.append(predIntLabel)
            intGolds.append(orgIntLabel)
    eval_chemprot(intPreds, intGolds, classes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_scibert", action='store_true')
    parser.add_argument("--scibertPredFile", default = None)
    parser.add_argument("--scibertLabelFile", default = None, help = 'a file stores a list of labels used by scibert')
    parser.add_argument("--eval_linkbert", action='store_true')
    parser.add_argument("--linkbertPredFile", default = None)
    args = parser.parse_args()
    if args.eval_scibert: 
        eval_scibert_output(args.scibertPredFile, args.scibertLabelFile)
    if args.eval_linkbert:
        eval_linkbert_output(args.linkbertPredFile)
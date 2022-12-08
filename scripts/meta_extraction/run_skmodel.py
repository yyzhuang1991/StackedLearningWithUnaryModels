import os, sys, argparse, torch, pickle, random, json, numpy as np
from os.path import abspath, dirname, join, exists
from os import makedirs
from sys import stdout
from copy import deepcopy
from sklearn.svm import SVC
from scipy.special import softmax
from feature_utils import form_features
sys.path.append("../")
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
sys.path.insert(0, "../shared")
from structures import Relation, Span, MteDocument

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def eval(inferInstances, evalInstances, tupleLevel = False):

    def score(predSet, goldSet):
        numCorrect = len(predSet.intersection(goldSet))
        numGold = len(goldSet)
        numPred = len(predSet)
        pre = numCorrect/numPred if numPred else 0 
        rec = numCorrect/numGold if numGold else 0 
        f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0 
        return pre, rec, f1
    # tuple-level
    if tupleLevel:
        predIds = set([(ins.span1.docId, ins.span1.stdText, ins.span2.stdText, ins.predRelationLabel) for ins in inferInstances if ins.predRelationLabel != 'O'])
        goldIds = set([(ins.span1.docId, ins.span1.stdText, ins.span2.stdText, ins.relationLabel) for ins in evalInstances if ins.relationLabel != 'O'])
    else: 
        # instance-level 
        predIds = set([(ins.id, ins.predRelationLabel) for ins in inferInstances if ins.predRelationLabel != 'O'])
        goldIds = set([(ins.id, ins.relationLabel) for ins in evalInstances if ins.relationLabel != 'O'])

    pre, rec, f1 = score(predIds, goldIds) 
    return pre, rec, f1 

def predict(model, inferInstances, featureFig):
    X = form_features(inferInstances, featureFig)
    try:
        scores = model.decision_function(X)
    except:
        scores = [[-1] for _ in range(len(X))]
    Y = model.predict(X)
    for ins, score, y in zip(inferInstances, scores, Y):
        ins.binaryPredScore = ins.predScore
        ins.binaryPredRelationLabel = ins.predRelationLabel
        ins.predScore = score
        ins.intPredRelationLabel = y.item()
    return inferInstances

def train(inferInstances, args, testInferInstances = None):

    model = SVC(C = args.C, gamma = args.gamma, kernel = args.kernel, tol = 1e-3, max_iter = 100)
    X = form_features(inferInstances, args)
    Y = [args.label2ind[ins.relationLabel] for ins in inferInstances]
    print("Training ...") 
    model.fit(X, Y)
    if not exists(args.modelOutdir):
        makedirs(args.modelOutdir)
    modelFile = join(args.modelOutdir, 'model.pkl')
    print(f"Saving the trained model to {modelFile}")
    with open(modelFile, 'wb') as f:
        pickle.dump(model, f)
    configFile = join(args.modelOutdir, 'config.pkl')
    print(f"Saving the config to {configFile}")
    with open(configFile, 'wb') as f:
        pickle.dump(args, f)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action = 'store_true', help = "Whether to train")
    parser.add_argument("--test", action = 'store_true', help = "Whether to eval")
    parser.add_argument("--C",default = 0.05, type = float, help = "C of svm")
    parser.add_argument("--kernel",default = 'linear', help = "kernel of svm")
    parser.add_argument("--gamma",default = 'auto')
    parser.add_argument("--numDistQuantiles", default = 15, type = int, help = "number of quantiles to vectorize entities' distances")
    parser.add_argument("--trainDir", default = None)
    parser.add_argument("--testDir", default = None)
    parser.add_argument("--modelOutdir", default = None, help = 'dir to save the model during training')
    parser.add_argument("--trainedModelDir", default = None, help = 'dir where the trained model is saved, used when eval is True')
    parser.add_argument("--evalOutFile", default = None, help = 'outfile to save evaluation results')
    args = parser.parse_args()
    return args

def load_data(indir):
    with open(join(indir, 'inferInstances.pkl'), 'rb') as f:
        inferInstances = pickle.load(f)
    with open(join(indir, 'evalInstances.pkl'), 'rb') as f:
        evalInstances = pickle.load(f)
        return inferInstances, evalInstances

def main(args):
    if args.train: 
        trainInferInstances, trainEvalInstances = load_data(args.trainDir)

        label2ind = {'O':0}
        relations = sorted(set([ins.relationLabel for ins in trainInferInstances if ins.relationLabel != 'O']))
        for relation in relations:
            label2ind[relation] = len(label2ind)
        ind2label = {v:k for k, v in label2ind.items()}
        args.label2ind = label2ind
        args.ind2label = ind2label
        numPos = len([ins for ins in trainInferInstances if ins.relationLabel != 'O'])
        numNeg = len([ins for ins in trainInferInstances if ins.relationLabel == 'O'])
        print(f"In TRAIN, loaded {len(trainInferInstances)} training data, label2ind = {label2ind}, #POS = {numPos}, #NEG = {numNeg}.")
        modelFile = join(args.modelOutdir, 'model.pkl')
        if exists(modelFile):
            os.remove(modelFile)
        if args.test:
            testInferInstances, testEvalInstances = load_data(args.testDir)

        train(trainInferInstances, args, testInferInstances = testInferInstances)
        
        # eval over training set 
        print()
        print(">>> Eval over TRAIN")
        print(f"Loading model from {modelFile}")
        with open(modelFile, 'rb') as f:
            model = pickle.load(f)

    elif args.test:
        oldArgs = pickle.load(open(join(args.trainedModelDir, 'config.pkl'), 'rb')) 
        oldArgs.trainedModelDir = args.trainedModelDir
        oldArgs.testDir = args.testDir
        oldArgs.evalOutFile = args.evalOutFile
        args = oldArgs

        print(f"Loaded configs from {args.trainedModelDir}, with label2ind {args.label2ind}")
        testInferInstances, testEvalInstances = load_data(args.testDir)
        print(f'Loaded {len(testEvalInstances)} Eval data')
        with open(join(args.trainedModelDir, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        testInferInstances = predict(model, testInferInstances, args)
        for ins in testInferInstances:
            ins.predRelationLabel = args.ind2label[ins.intPredRelationLabel]

        # insPre, insRec, insF1 = eval(testInferInstances, testEvalInstances, tupleLevel = False)
        tupPre, tupRec, tupF1 = eval(testInferInstances, testEvalInstances, tupleLevel = True)
        # evalStr = f">>> Evaluation over EVAL set:\nInstance Level: Precision = {insPre*100:.2f}, Recall = {insRec*100:.2f}, F1 = {insF1*100:.2f}\nTuple Level: Precision = {tupPre*100:.2f}, Recall = {tupRec*100:.2f}, F1 = {tupF1*100:.2f}\n"
        evalStr = f"Precision = {tupPre*100:.2f}, Recall = {tupRec*100:.2f}, F1 = {tupF1*100:.2f}\n"
        print(evalStr)
        if args.evalOutFile is not None:
            print(f"Saving prediction to {args.evalOutFile}.predInstances")        
            with open(args.evalOutFile+".predInstances", 'wb') as f:
                pickle.dump(testInferInstances, f)

            evalOutdir = "/".join(args.evalOutFile.split("/")[:-1])
            if not exists(evalOutdir):
                makedirs(evalOutdir)
            print(f"Saving evaluation result to {args.evalOutFile}")
            print(evalStr)
            with open(args.evalOutFile, 'w') as f:
                f.write(evalStr)

if __name__ == "__main__":
    args = get_parser()
    main(args)

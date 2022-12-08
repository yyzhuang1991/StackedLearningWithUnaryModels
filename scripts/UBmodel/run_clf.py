import os, sys, argparse, torch, pickle, random, json, numpy as np
from os.path import abspath, dirname, join, exists
from os import makedirs
from transformers import *
from torch.utils.data import DataLoader
from sys import stdout
from copy import deepcopy

from model import UnaryModel
from dataset import MyDataset, collate

sys.path.append('../')

from eval import eval 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

shared_path = '../shared'
sys.path.insert(0, shared_path)
from other_utils import add_marker_tokens
from structures import Span

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def predict(model, dataloader):
    model = model.to(device)
    model.eval()
    predInstances = []
    soft = torch.nn.Softmax(dim = 1)
    with torch.no_grad():  
        for i, item in enumerate(dataloader):
            stdout.write(f"\rpredicting {i}/{len(dataloader)}")
            stdout.flush()
            if 'bert_starts' in item:
                logits =model.forward(item)
            else:
                logits = model(item['sent_inputids'].to(device), attention_mask = item['sent_attention_masks'].to(device))[0]

            scores = soft(logits).cpu().numpy()
            yPreds = np.argmax(scores,1)

            for ins, yPred, score in zip(item["instances"], yPreds, scores):
                ins.intPredRelationLabel = yPred.item()
                ins.predScore = score
                predInstances.append(ins)

    return predInstances



def eval_and_save(model, testLoader, testEvalInstances,  bestF1, args, tupleLevel = False, binary = False): 
    """
    Evaluate a trained model over validation set and save if achieves better performance 
    """

    print("\n\n---------------eval------------------\n")
    predInstances = predict(model, testLoader)
    for ins in predInstances:
        ins.predRelationLabel = args.ind2label[ins.intPredRelationLabel]
    if tupleLevel:
        if binary:
            predSigns = [(ins.docId, ins.span1.stdText,ins.span2.stdText) for ins in predInstances if ins.predRelationLabel != 'O']
            goldSigns = [(ins.docId, ins.span1.stdText, ins.span2.stdText) for ins in testEvalInstances if ins.relationLabel != 'O']
        else:
            predSigns = [(ins.docId, ins.stdText) for ins in predInstances if ins.predRelationLabel != 'O']
            goldSigns = [(ins.docId, ins.stdText) for ins in testEvalInstances if ins.relationLabel != 'O']
    else: 
        predSigns = [ins.id for ins in predInstances if ins.predRelationLabel != 'O']
        goldSigns = [ins.id for ins in testEvalInstances if ins.relationLabel != 'O']

    shouldSave = 0
    precision, recall, f1 = eval(predSigns, goldSigns)
    scoreStr = "TUPLE LEVEL: " if tupleLevel else "Instance Level: "
    scoreStr += f"precision: {precision*100:.2f}, recall: {recall*100:.2f}, f1: {f1*100:.2f}, best f1: {bestF1}\n"

    if bestF1 is None or f1 > bestF1:
        bestF1 = f1
        shouldSave = 1
    print()
    print(f"------------- Evaluation ------------\n\n{scoreStr}\n")
    print()
    if shouldSave and args.saveModel:

        if not os.path.exists(args.modelSaveDir):
            os.makedirs(args.modelSaveDir)

        print(f"\nsaving model to {args.modelSaveDir}\n")
   
        torch.save(model.state_dict(), join(args.modelSaveDir, "model.ckpt"))
        # write model setting
        with open(join(args.modelSaveDir, "args.pkl"), "wb") as f:
            pickle.dump(args, f)

    return predInstances, bestF1 


def train(model, trainInstances, trainEvalInstances, devInferInstances, devEvalInstances, args):
    model.to(device)
    trainLoader = create_dataloader(trainInferInstances, True, args, shuffle = args.shuffle)
    numBatches = len(trainLoader)
    devLoader = create_dataloader(devInferInstances, False, args, shuffle = False)
    """ optimizer """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
               {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.01},
               {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad ], 'weight_decay': 0.0}
          ]

    optimizer = AdamW(optimizer_grouped_parameters,
                              lr=args.lr, correct_bias=False)

    """ scheduler """
    numTrainingSteps = len(trainLoader) * args.epoch
    numWarmupSteps = int(0.1 * numTrainingSteps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=numWarmupSteps, num_training_steps=numTrainingSteps)  # PyTorch scheduler
    bestF1 = None
    lossFunct = torch.nn.CrossEntropyLoss()
    curPatience = args.trainPatience
    for epoch in range(args.epoch):
        avgLoss = 0
        model.train()
        
        for idx, item in enumerate(trainLoader):
            if 'bert_starts' in item:
                logits = model.forward(item)
            else:
                logits = model.forward(item['sent_inputids'].to(device), attention_mask = item['sent_attention_masks'].to(device))[0]
            loss = lossFunct(logits, item["labels"].to(device))
            stdout.write(f'\r{epoch}/{args.epoch} epoch: batch = {idx}/{numBatches} batch, batch loss = {loss.item():.2f}')
            stdout.flush()
            avgLoss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.maxGradNorm)
            optimizer.step()
            scheduler.step() 
            optimizer.zero_grad()

        print("\n\n---------------------------------\n")
        print(f"\n{epoch} epoch avg loss: {avgLoss/numBatches:.2f}\n\n")
        print("--- eval over val ... ---")
        args.saveModel = 1
        lastBestF1 = bestF1
        _, bestF1 = eval_and_save(model, devLoader, devEvalInstances, bestF1, args, tupleLevel = False, binary = args.mode == 'binary')
        if lastBestF1 is None or bestF1 > lastBestF1: 
            curPatience = args.trainPatience
        else:
            curPatience -= 1
        print(f"Current training patience: {curPatience}")
        lastBestF1 = bestF1
        if curPatience <= 0: 
            print(f"Running out of patience (started at {args.trainPatience}) at epoch {epoch}")
            break 
    print("Training ends ! \n")

def create_dataloader(inferInstances, isTraining, args, iterIdx=0, shuffle = True):
    for ins in inferInstances:
        ins.intRelationLabel = args.label2ind[ins.relationLabel]

    if isTraining:
        trainInferInstances = inferInstances
        trainPos = [] 
        trainNoisyNeg = [] 
        trainNegExcludingNoisy = []
        numPos = 0  
        numNeg = 0 
        for ins in trainInferInstances:
            if ins.relationLabel != 'O':
                trainPos.append(ins)
                numPos += 1 
            else:
                if hasattr(ins, 'haveNoisyLabel'):
                    if ins.haveNoisyLabel:
                        trainNoisyNeg.append(ins)
                    else:
                        trainNegExcludingNoisy.append(ins)
                        numNeg += 1 
                else:
                    trainNegExcludingNoisy.append(ins)
                    numNeg += 1 

        print(f"Originally {numPos + numNeg} in TRAIN dataset, {numPos}({numPos/(numPos + numNeg)*100:.2f}%) POS and {numNeg}({numNeg/(numPos + numNeg)*100:.2f}%) NEG")

        trainInstances = trainPos + trainNegExcludingNoisy
        random.Random(args.seed + iterIdx).shuffle(trainInstances)
        numPos = len([1 for ins in trainInstances if ins.relationLabel != 'O'])
        numNeg = len([1 for ins in trainInstances if ins.relationLabel == 'O'])
        numInstances = len(trainInstances)
        print(f"{len(trainInstances)} in TRAIN dataset, including {numPos}({numPos/numInstances*100:.2f}%) pos, {numNeg}({numNeg/numInstances*100:.2f}) neg.")
        trainDataloader = DataLoader(MyDataset(trainInstances), batch_size = args.batchSize, shuffle = args.shuffle, collate_fn = collate)
        return trainDataloader
    else:
        dataloader = DataLoader(MyDataset(inferInstances), batch_size = args.batchSize, shuffle = False, collate_fn = collate)
        return dataloader

def make_inputids(instances, tokenizer, maskNer = 0):
    for ins in instances:
        try:
            ins.insert_type_markers_by_offset(tokenizer, maskNer = maskNer)
        except:
            ins.insert_type_markers(tokenizer, maskNer = maskNer)
    print("----")
    print(f"Sampled output of input ids: {instances[0].inputIds} ")
    print(f"Decoded: {tokenizer.convert_ids_to_tokens(instances[0].inputIds)}")
    print(f"---")

def load_data(dataDir, load_eval = True):
    with open(join(dataDir, f"inferInstances.pkl"), "rb") as f:
        inferInstances = pickle.load(f)
    if load_eval:
        with open(join(dataDir, f"evalInstances.pkl"), "rb") as f:
            evalInstances = pickle.load(f)
        return inferInstances, evalInstances

    return inferInstances

if __name__ == "__main__":
    """ ================= parse =============== """
    parser = argparse.ArgumentParser()
    
    # I/O
    parser.add_argument("--mode", choices = ['binary', 'unary'], required =  True)
    parser.add_argument("--task", default = 'mte', choices = ['mte', 'chemprot'])
    parser.add_argument("--modelType", choices = ['bert-base-uncased','allenai/scibert_scivocab_uncased', 'michiyasunaga/BioLinkBERT-base','michiyasunaga/LinkBERT-base'], default = 'bert-base-uncased')
    parser.add_argument("--maskNer", default = 0, type = int, choices = [0, 1])
    parser.add_argument("--trainDir", default = None, help = 'directory where extracted span instances (made by make_train_dev_test.py) are stored')
    parser.add_argument("--testDir", default = None, help = 'directory where extracted span instances from DEV data are stored')
    parser.add_argument("--train", action = 'store_true')
    parser.add_argument("--trainPatience", default = 3, type = int)
    parser.add_argument("--test", action = 'store_true')
    parser.add_argument("--trainedModelDir", default = None)
    parser.add_argument("--evalOutdir", default = None)
    parser.add_argument("--dropout", default = 0.1, type = float)
    parser.add_argument("--seed", help = "seed", default = 100, type = int)
    parser.add_argument("--epoch", help = "number epochs to train", default = 5, type = int)
    parser.add_argument("--lr", default = 0.00001, type = float)
    parser.add_argument("--saveModel", help = "whether to save the model or not", default = 1, type = int, choices = [0,1])
    parser.add_argument("--modelSaveDir", default = "saved_model", help = "where to save the model to")
    parser.add_argument("--batchSize", type = int, default = 10, help = "batch size")
    parser.add_argument("--shuffle", type = int, default = 1, choices = [1,0], help = "whether to shuffle the training data")
    parser.add_argument("--maxGradNorm", default = 1.00, type = float, help = "max gradient norm to clip")

    args = parser.parse_args()
    seed = args.seed
    seed_everything(seed)
    print("Loading data ")
    if args.train:
        print(f"Using tokenizer type: {args.modelType}")
        tokenizer = BertTokenizerFast.from_pretrained(args.modelType)
        print(f"Loading data from {args.trainDir} and {args.testDir}")
        if args.task == 'mte':
            trainInferInstances, trainEvalInstances = load_data(args.trainDir, load_eval = True)
            devInferInstances, devEvalInstances = load_data(args.testDir, load_eval = True)
        elif args.task == 'chemprot':
            trainInferInstances = load_data(args.trainDir, load_eval = False)
            devInferInstances = load_data(args.testDir, load_eval = False)
            trainEvalInstances = deepcopy(trainInferInstances)
            devEvalInstances = deepcopy(devInferInstances)

        ners = set()
        relations = set()
        for ins in trainEvalInstances:
            if ins.relationLabel != 'O':
                relations.add(ins.relationLabel)
            if args.mode == 'unary':
                if ins.ner != 'O':
                    ners.add(ins.ner)
            else:
                for u in [ins.span1, ins.span2]:
                    if u.ner != 'O': 
                        ners.add(u.ner)

        if args.mode == 'unary':
            if len(ners) != 1:
                raise NameError(f"UNARY MODEL: NERs in the evluation data must have only 1 type! Got NERs = {ners}")
            if len(relations) != 1:
                raise NameError(f"UNARY: RELATIONs in the evluation data must have only 1 type! Got RELATIONs = {relations}")
            ner = list(ners)[0]
            relation = list(relations)[0]
            args.ner = ner 
            args.relation = relation 
            label2ind = {'O': 0, relation: 1}
            ind2label = {v:k for k, v in label2ind.items()}
            args.label2ind = label2ind
            args.ind2label = ind2label
            args.num_classes = len(label2ind)
            add_marker_tokens(tokenizer, [args.ner]) 
            print(f"Model would be trained for NER = {args.ner}, RELATION = {args.relation}")
            make_inputids(trainInferInstances, tokenizer, maskNer = args.maskNer)
            make_inputids(devInferInstances, tokenizer,maskNer = args.maskNer)

            model = UnaryModel(tokenizer, args)
            print("Start training ")
            train(model, trainInferInstances, trainEvalInstances, devInferInstances, devEvalInstances, args)
            del model 
        else:
            ners = sorted([n for n in ners])
            relations = sorted([r for r in relations])
            args.ners = ners 
            args.relations = relations 
            label2ind = {'O': 0}
            for r in args.relations:
                label2ind[r] = len(label2ind)
            ind2label = {v:k for k, v in label2ind.items()}
            args.label2ind = label2ind
            args.ind2label = ind2label
            args.num_classes = len(label2ind)
            add_marker_tokens(tokenizer, args.ners) 
            print(f"Model would be trained for NER = {args.ners}, RELATION = {args.relations}")
            make_inputids(trainInferInstances, tokenizer, maskNer = args.maskNer)
            make_inputids(devInferInstances, tokenizer, maskNer = args.maskNer)
            model = BertForSequenceClassification.from_pretrained(args.modelType, num_labels = args.num_classes)
            model.resize_token_embeddings(len(tokenizer))
            print("Start training ")
            train(model, trainInferInstances, trainEvalInstances, devInferInstances, devEvalInstances, args)
            del model 


    if args.test:
        if not exists(args.evalOutdir):
            makedirs(args.evalOutdir)
        configFile = join(args.trainedModelDir, "args.pkl")
        print(f"Loading config from {configFile}")
        oldArgs = pickle.load(open(configFile, 'rb'))
        args.modelType = oldArgs.modelType
        args.num_classes = oldArgs.num_classes
        args.task = oldArgs.task 
        args.maskNer = oldArgs.maskNer
        args.label2ind = oldArgs.label2ind
        args.ind2label = oldArgs.ind2label   
        print(f"Using tokenizer type: {args.modelType}")
        tokenizer = BertTokenizerFast.from_pretrained(args.modelType) 
        if args.mode == 'unary':
            args.ner = oldArgs.ner 
            args.relation = oldArgs.relation 
            add_marker_tokens(tokenizer, [args.ner]) 
            modelFile = join(args.trainedModelDir, "model.ckpt")
            print(f"Loading model from {modelFile}")
            model = UnaryModel(tokenizer, args)
            model.load_state_dict(torch.load(modelFile))
            print(f"Model would be tested for NER = {args.ner}, RELATION = {args.relation}")
            if args.task == 'mte':
                testInferInstances, testEvalInstances = load_data(args.testDir, load_eval = True)
            elif args.task == 'chemprot':
                testInferInstances = load_data(args.testDir, load_eval = False)
                testEvalInstances = deepcopy(testInferInstances)
            
            make_inputids(testInferInstances, tokenizer, maskNer = args.maskNer)
            testLoader = create_dataloader(testInferInstances,False, args,shuffle = False)
            args.saveModel = 0
            predInstances, _ = eval_and_save(model, testLoader, testEvalInstances, None, args, tupleLevel = False, binary = args.mode == 'binary')
            outfile = join(args.evalOutdir, 'predInstances.pkl') 
            print(f"Saving predicted instances to {outfile}")
            with open(outfile, 'wb') as f:
                pickle.dump(predInstances, f)
        else:
            args.ners = oldArgs.ners 
            args.relations = oldArgs.relations 
            add_marker_tokens(tokenizer, args.ners) 
            modelFile = join(args.trainedModelDir, "model.ckpt")
            print(f"Loading model from {modelFile}")
            model = BertForSequenceClassification.from_pretrained(args.modelType,num_labels = args.num_classes)
            model.resize_token_embeddings(len(tokenizer))
            model.load_state_dict(torch.load(modelFile))
            print(f"Model would be tested for NER = {args.ners}, RELATION = {args.relations}")
            testInferInstances, testEvalInstances = load_data(args.testDir)
            make_inputids(testInferInstances, tokenizer, maskNer = args.maskNer)
            testLoader = create_dataloader(testInferInstances,False, args, shuffle = False)
            args.saveModel = 0
            predInstances, _ = eval_and_save(model, testLoader, testEvalInstances, None, args, tupleLevel = False, binary = args.mode == 'binary')
            _, _ = eval_and_save(model, testLoader, testEvalInstances, None, args, tupleLevel = True, binary = args.mode == 'binary')
            outfile = join(args.evalOutdir, 'predInstances.pkl') 
            print(f"Saving predicted instances to {outfile}")
            with open(outfile, 'wb') as f:
                pickle.dump(predInstances, f)

    
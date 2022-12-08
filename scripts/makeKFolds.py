from os.path import abspath, dirname, join, exists
import sys , glob, argparse
from os import makedirs
sys.path.append('./shared/')
from extraction_utils import extract_gold_relations_from_ann, get_docid, extract_gold_entities_from_ann
import random
import numpy as np
from sklearn.model_selection import KFold 

def get_textfile(ann_file):
	return abspath(ann_file[:-4] + ".txt")

def get_corenlpfile(ann_file):
	filename = ann_file.split('corpus-LPSC/')[1][:-4]
	return abspath(f"../parse/{filename}.txt.json") 

def get_gold_annot_stats(docid2annFiles):
	relation2count = {}
	docid2relation2count = {}
	docid2entity2set = {}
	for i, docid in enumerate(docid2annFiles):
		seenGoldRelations = set()
		seenGoldEntities = set()
		if docid not in docid2relation2count:
			docid2relation2count[docid] = {}
			docid2entity2set[docid] = {}

		for annFile in docid2annFiles[docid]:
			tempGoldRelations = extract_gold_relations_from_ann(annFile, use_component = True)
			goldEntities = extract_gold_entities_from_ann(annFile, use_component = True)
			for e in goldEntities: 
				esign = f"{e['docid']}__{e['doc_start_char']}-{e['doc_end_char']}"
				if e['label'] not in docid2entity2set[docid]:
					docid2entity2set[docid][e['label']] = set()
				docid2entity2set[docid][e['label']].add(esign)


			for e1, e2, relation in tempGoldRelations:
				if e2['label'] == 'Target':
					e1, e2 = e2, e1 
				sign = f"{e1['docid']}__{e1['doc_start_char']}-{e1['doc_end_char']}|{e2['doc_start_char']}-{e2['doc_end_char']}||{relation}"
				if sign in seenGoldRelations:
					continue 
				
				seenGoldRelations.add(sign)
				relation2count[relation] = relation2count.get(relation, 0) + 1
				docid2relation2count[docid][relation] = docid2relation2count[docid].get(relation, 0) + 1  
	print("Stats in gold annotation: ")
	for r in relation2count:
		print(f"       {r}: {relation2count[r]}")
	eType2count = {}
	for did in docid2entity2set:
		for eType in docid2entity2set[did]:
			eType2count[eType] = eType2count.get(eType, 0) + len(docid2entity2set[did][eType])
	print("\nENTITY\n")
	for eType in eType2count:
		print(f"       {eType}: {eType2count[eType]}")

	return relation2count, docid2relation2count, docid2entity2set

def kfold(docids, k):
	x = list(range(len(docids)))
	kf = KFold(n_splits=k)
	train_indices = [] 
	test_indices = [] 
	for train_index, test_index in kf.split(x):
		train_indices.append(list(train_index))
		test_indices.append(list(test_index))
	dev_indices = [] 
	for i in range(k):
		if i == 0:
			dev_indices.append(list(test_indices[-1]))
		else:
			dev_indices.append(list(test_indices[i-1]))

	# fold files
	trainDocids = [] 
	devDocids = [] 
	testDocids = []

	for train, dev, test in zip(train_indices, dev_indices, test_indices):
		set_dev = set(list(dev))
		trainDocids.append([docids[i] for i in train if i not in set_dev])
		devDocids.append([docids[i] for i in dev]) 
		testDocids.append([docids[i] for i in test])

	return trainDocids, devDocids, testDocids


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--k", help = "number of folds", type = int, default = 5)
	parser.add_argument("--devPercent", help = 'size of the fixed development set. percentage', type = float, default = 0.25)
	parser.add_argument("--outdir", help = "output directory", required = True)
	
	args = parser.parse_args()

	docid2annFiles = {}
	dirs = ["lpsc15", "lpsc16", "phx", "mpf", "mer-a"]

	countAnnFiles = 0 
	for dir_ in dirs:
		annFiles = glob.glob(join("../corpus-LPSC",f"{dir_}/*.ann"))
		for annFile in annFiles:
			year, docname, docid = get_docid(annFile) # docid is year_docname
			if docid not in docid2annFiles:
				docid2annFiles[docid] = set()
			docid2annFiles[docid].add(annFile) 
			countAnnFiles += 1 

	print(f"Loaded {len(docid2annFiles)} documents (by unique docnames), {countAnnFiles} documents (by venue-docname, so it contains duplicated documents from mer-a)")
	
	docids = sorted(docid2annFiles.keys())
	random.Random(100).shuffle(docids)
	outdir = args.outdir
	if not exists(outdir):
		makedirs(outdir)
		
	devSize = int(args.devPercent * len(docids)) + 1
	devDocids = docids[:devSize] 
	docids = docids[devSize:]
	fixedDevDocids = devDocids
	
	trainDocids, devDocids, testDocids = kfold(docids, args.k)
	for fold in range(args.k):
		curoutdir = join(outdir, f"fold{fold}")
		if not exists(curoutdir):
			makedirs(curoutdir)
		print(f"Creating list of files for fold {fold} at {curoutdir}")
		trainFolds = trainDocids[fold] + devDocids[fold]
		random.Random(100).shuffle(trainFolds)
		devFolds = fixedDevDocids

		testFolds = testDocids[fold]
		for name, docids in zip(["train", "dev", 'test'], [trainFolds, devFolds, testFolds]):
			with open(join(curoutdir, f'{name}.list'), 'w') as f:
				for docid in docids:
					annFiles = docid2annFiles[docid]
					for annFile in annFiles:
						f.write(",".join((abspath(annFile), get_textfile(annFile), get_corenlpfile(annFile))) + "\n")

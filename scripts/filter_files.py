import sys, os, glob 
sys.path.append("shared")
from extraction_utils import extract_gold_entities_from_ann

def remove_file(annFile):
	entities = extract_gold_entities_from_ann(annFile, use_component = True)
	hasTarget = any([e['label'] == 'Target' for e in entities])
	hasComponent = any([e['label'] == 'Component' for e in entities])
	hasProperty = any([e['label'] == 'Property' for e in entities])
	# if it does not contain any pair of (Target, Component) or (Target, Property), we skip this file
	if not hasTarget or (not hasComponent and not hasProperty):
		return True 
	else:
		return False

for name in ['mpf', 'phx']:
	new_set = set()
	for annFile in glob.glob(f"../corpus-LPSC/{name}/*.ann"):
		if remove_file(annFile): 
			os.remove(annFile)
			txtFile = annFile.split(".ann")[0] + ".txt"
			os.remove(txtFile)
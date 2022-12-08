from os.path import dirname, abspath, exists, join 
import sys 
curpath = dirname(abspath(__file__))
extractionUtilsPath = join(dirname(dirname(curpath)), 'shared')
sys.path.append(extractionUtilsPath)
from extraction_utils import need_swap_entity

mte_validRelationSet = {
    'Contains': set([('Target', 'Component', 'Contains')]),
    'HasProperty': set([("Target", 'Property', 'HasProperty')])    
    }

# -----
# sort ner label in order 
tempValidRelationSet = {}
for k in mte_validRelationSet:
    tempSet = set()
    for a, b, r in mte_validRelationSet[k]:
        a = a
        b = b
        if need_swap_entity(a,b):
            b,a = a,b
        tempSet.add((a,b,r))
    tempValidRelationSet[k] = tempSet

mte_validRelationSet = tempValidRelationSet

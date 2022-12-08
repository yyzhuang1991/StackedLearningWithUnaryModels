import copy, re, json, os, glob, warnings, sys
from os.path import exists, join, abspath, dirname
from copy import deepcopy 

from my_name_utils import canonical_target_name, canonical_elemin_name, canonical_property_name, canonical_name


def add_entities(queue, e):
    # add entities and merge entities if possible. Merge entities when two words have the same ner label that is not 'O' and (adjacent or two words are separated by hyphens or underscores). Note that this method is not perfect since we always merge adjacent words with the same NER into an entity, thus will lose a lot of smaller entities. For example, we will get only "Iron - Feldspar" and miss "Iron" and "Feldspar"

    if not len(queue) or e['label'] == 'O':
        queue.append(deepcopy(e))
        return 
    last_e = queue[-1]
    if last_e['label'] == e['label']:
        # merge 
        last_e['text'] = f"{last_e['text']} {e['text']}"
        last_e['doc_end_char'] = e['doc_end_char']
        last_e['sent_end_idx'] = e['sent_end_idx']
    else:
        if len(queue) > 1 and queue[-1]['text'] in ["_", "-"] and queue[-2]['label'] == e['label']: # words that are splitted by hyphen or underscores
            queue[-2]['text'] = f"{queue[-2]['text']}{last_e['text']}{e['text']}"
            queue[-2]['doc_end_char'] = e['doc_end_char']
            queue[-2]['sent_end_idx'] = e['sent_end_idx']
            queue.pop(-1)
        else:
            queue.append(deepcopy(e))


def make_tree_from_ann_file(text_file, ann_file, be_quiet = True):
    with open(text_file) as text_file, open(ann_file) as ann_file:
        texts = text_file.read()
        text_file.close()

        anns = map(lambda x: x.strip().split('\t'), ann_file)
        anns = filter(lambda x: len(x) > 2, anns)
        anns = filter(lambda x: ';' not in x[1], anns)

        def __parse_ann(ann):
            spec = ann[1].split()
            name = spec[0]
            markers = list(map(lambda x: int(x), spec[1:]))
            t = texts[markers[0]:markers[1]]
            if not t == ann[2]:
                if not be_quiet:
                    print("Error: Annotation mis-match, file=%s, ann=%s" % (text_file, str(ann)))
                return None
            return (name, markers, t)
        anns = map(__parse_ann, anns) # format
        anns = filter(lambda x: x, anns) # skip None

        # building a tree index for easy accessing
        tree = {}
        for entity_type, pos, name in anns:
            begin, end = pos[0], pos[1]
            if begin not in tree:
                tree[begin] = {}
            node = tree[begin]
            if end not in node:
                node[end] = []
            node[end].append(entity_type)

        return tree


def collect_entity_at_offset(token, sentid, tokidx, charoffset, offset_entity_label, charoffset2label2entities):
        """ map character offset to label to entities """

        # if charoffset2entities is emtpy, create an entity 
        if charoffset not in charoffset2label2entities:
            charoffset2label2entities[charoffset] = {}
        if offset_entity_label not in  charoffset2label2entities[charoffset]:
            charoffset2label2entities[charoffset][offset_entity_label] = []

        entity = {
            "text": token["word"],
            "doc_start_char": token["characterOffsetBegin"],
            "doc_end_char": token["characterOffsetEnd"],
            "sent_start_idx": int(tokidx),
            "sent_end_idx": int(tokidx) + 1,
            "sentid": int(sentid)
        }
        charoffset2label2entities[charoffset][offset_entity_label].append(entity)

def merge_and_make_entities(charoffset2label2entities, text_file):
    entities = []
    for charoffset in charoffset2label2entities:
        for label in charoffset2label2entities[charoffset]:
            entity_parts = sorted(charoffset2label2entities[charoffset][label], key = lambda x: x["sent_start_idx"])

            # merge while make sure entity parts are sequential and do not have overlap 
            entity = deepcopy(entity_parts[0])
            last_entity_is_hypen_or_underscore = 0
            for i in range(1, len(entity_parts)):
                curentity = entity_parts[i]
                        

                if not (charoffset[0] <= entity["doc_start_char"] < entity["doc_end_char"] <= curentity["doc_start_char"] < curentity["doc_end_char"] <= charoffset[1]):

                    raise NameError(f"Entity parts are not able to form a valid entity due to incorrect boundaries to merge in {text_file}:\n OFFSET: {charoffset}\n   last entity part: {entity['doc_start_char']}, {entity['doc_end_char']}, {entity['text']}\n   cur entity part:  {curentity['doc_start_char']}, {curentity['doc_end_char']}, {curentity['text']}")
                if curentity['sentid'] != entity['sentid']:
                    raise NameError(f"Inconsistent sentence ID in {text_file}! last entity: {entity['sentid']}, current entity: {curentity['sentid']}")
                # merge 
                text = entity['text']
                if curentity['text'] in ["_", "-"]: # if current word is a hyphen or underscore, then append it to the text without any space 
                    text = entity['text'] + curentity['text']
                    last_entity_is_hypen_or_underscore = 1
                else: # if current word is not a hypen but last word is hypen, then we append it to the text without spance 
                    if last_entity_is_hypen_or_underscore:
                        text = entity['text'] + curentity['text']
                    else:
                        text = entity['text'] + " " + curentity['text'] 
                    last_entity_is_hypen_or_underscore = 0 


                entity = {
                    "text": text,
                    "doc_start_char": entity['doc_start_char'],
                    "doc_end_char": curentity['doc_end_char'],
                    "sent_start_idx": entity['sent_start_idx'],
                    "sent_end_idx": curentity['sent_end_idx'],
                    "sentid": entity['sentid']
                }
            entity['label'] = label 
            entities.append(entity)
    return entities


def extract_entities_from_text(text_file, ann_file, doc = None, corenlp_file = None, use_component = True, be_quiet = True):
    """
    this function extracts entities from text file 
    Argument:
        text_file: text_file that contains the journal/abstract 
        
        ann_file: file that contains annotations from brat. 

        doc:  a dictionary that stores corenlp's parsing for text_file. either doc or corenlp_file must not be none. 

        corenlp_file: a file that stores corenlp's parsing for text_file. 

        use_component: whether to map element and mineral to component 
    
        be_quiet: whether to print out warnings during extraction
    """
    
    year, docname, docid = get_docid(text_file)

    if doc is None and corenlp_file is None:
        raise NameError("Either doc_dict or corenlp_file must be provided ! ")
    if doc is None:
        # read corenlp file
        doc = json.load(open(corenlp_file, "r"))

    tree = make_tree_from_ann_file(text_file, ann_file)

    charoffset2label2entities = {}

    text = open(text_file, "r").read()

    for s in doc["sentences"]:
        tokens = [t for t in s["tokens"]]
        sentid = int(s["index"]) # starts from 0 
        for tokidx, token in enumerate(tokens):
            token_begin, token_end = token["characterOffsetBegin"], token["characterOffsetEnd"]

            if text[token_begin: token_end] != token["word"]:
                if not be_quiet:
                    warnings.warn(f"ERROR Mismatch text: ({token['word']})\n offset from corenlp: ({token['characterOffsetBegin']}, {token['characterOffsetEnd']})\ntext according to offset from corenlp: ({text[token_begin: token_end]})")
                continue

            for begin in tree:
                for end in tree[begin]:
                    if  begin <= token_begin < token_end <= end:

                        charoffset = (begin ,end)

                        for offset_entity_label in tree[begin][end]:
                            collect_entity_at_offset(token, sentid, tokidx, charoffset, offset_entity_label, charoffset2label2entities)
    
    entities = merge_and_make_entities(charoffset2label2entities, text_file)

    if use_component:
        for e in entities:
            if e['label'] in ['Element', 'Mineral']:
                e['label'] = 'Component'
    for e in entities:
        # e["venue"] = venue
        e["docname"] = docname
        e["year"] = year 
        e['docid'] = docid

        if e['label'] == 'Target':
            e['std_text'] = canonical_target_name(e['text'])
        elif e['label'] in ['Element', 'Mineral', 'Component']:
            e['std_text'] = canonical_elemin_name(e['text'])
        elif e['label'] == 'Property':
            e['std_text'] = canonical_property_name(e['text'])
        else:
            e['std_text'] =  canonical_name(e['text'])

    return entities
    
def get_docid(ann_file):
    if "lpsc" in ann_file:
        year = "20"+re.findall(r"lpsc(\d+)", ann_file)[0]
        docname = ann_file.split("/")[-1].split(".")[0]
    elif "mpf" in ann_file or "phx" in ann_file or "mer-a" in ann_file:
        year, docname = re.findall(r"(\d+)_(\d+)", ann_file.split("/")[-1])[0] 
    else:
        raise NameError(f"file must be from LPSC or MPF or PHX. Currently we have {ann_file}")
    doc_id = f"{year}_{docname}"
    return year, docname, doc_id

def extract_gold_entities_from_ann(ann_file, use_component = True):
    year, docname, docid = get_docid(ann_file)

    entities = []
    seen_annotids = set()
    with open(ann_file, "r") as f:
        for k in f.readlines():
            k = k.strip()
            if len(k.split("\t")) == 3:
                annot_id, label, span = k.split("\t")
                if annot_id in seen_annotids:
                    raise NameError(f"Duplicated Annotation IDs {annot_id} in {ann_file}")
                seen_annotids.add(annot_id)

                label, doc_start_char, doc_end_char = label.split()
                doc_start_char = int(doc_start_char)
                doc_end_char = int(doc_end_char)
                if annot_id[0] == "T":
                    entity = {
                        "text": span.lower(),
                        "annot_id": annot_id,
                        "doc_start_char": doc_start_char,
                        "doc_end_char": doc_end_char,
                        "label": label,
                        # "venue": venue,
                        "year": year,
                        "docname": docname,
                        "docid": docid
                    }
                    entities.append(entity)

    for e in entities:
        if e['label'] == 'Target':
            e['std_text'] = canonical_target_name(e['text'])
        elif e['label'] in ['Element', 'Mineral','Component']:
            e['std_text'] = canonical_elemin_name(e['text'])
        elif e['label'] == 'Property':
            e['std_text'] = canonical_property_name(e['text'])
        else: 
            e['std_text'] = canonical_name(e['text'])

    if use_component:
        for e in entities:
            if e['label'] in ['Element', 'Mineral']:
                e['label'] = 'Component'

    return entities

def need_swap_entity(ner1, ner2):
    # tell if we should swap two entities to construct a relation. this is to make sure that ners labels are put in order
    return not ner1 <= ner2 

def extract_gold_relations_from_ann(ann_file, use_component = True):
    """ This function extract relations from ann files

    Args:
        ann_file: .ann file 
        use_component: whether to map element and mineral to component 

    return : a list of gold relations (e1, e2, relation), where e1 and e2 are sorted by their ner labels
    """

    annotid_annotid_relation = []
    for annotation_line in open(ann_file).readlines():
        if annotation_line.strip() == "": continue
        splitline = annotation_line.strip().split('\t')
        annot_id = splitline[0]

        if splitline[0][0] == "R":
            args = splitline[1].split()
            if len(args) == 3:
                relation = args[0]
                arg1 = args[1].split(":")[1]
                arg2 = args[2].split(":")[1]
                annotid_annotid_relation.append((arg1, arg2, relation))

        elif splitline[0][0] == 'E': # event
            args         = splitline[1].split() 
            relation   = args[0].split(':')[0]
            
            anchor  = args[0].split(':')[1]
            args         = [a.split(':') for a in args[1:]]
            targets = [v for (t,v) in args if t.startswith('Targ')]
            cont    = [v for (t,v) in args if t.startswith('Cont')]

            for t in targets:
                for c in cont:
                    annotid_annotid_relation.append((t, c, relation))

    gold_entities = extract_gold_entities_from_ann(ann_file, use_component = use_component)
    annotid2entities = {e["annot_id"]: e for e in gold_entities}

    gold_relations = []
    for t, c, relation in annotid_annotid_relation:
        e1 = annotid2entities[t]
        e2 = annotid2entities[c]
        if need_swap_entity(e1['label'], e2['label']):
            gold_relations.append((deepcopy(e2), deepcopy(e1), relation))
        else:
            gold_relations.append((deepcopy(e1), deepcopy(e2), relation))

    return gold_relations

def extract_intrasent_goldrelations_from_ann(ann_file, corenlp_file = None, doc = None,use_component = True):
    """ This function extracts gold relations with entities in the same sentence

    Args:
        ann_file: .ann file 
    
        corenlp_file: file that stores corenlp's parsing in json file 
    
        doc: dictionary that stores corenlp's parsing
        
        use_component: whether to map element and mineral to component 
    """

    if doc is None and corenlp_file is None:
        raise NameError("Either doc_dict or corenlp_file must be provided ! ")
    if doc is None:
        # read corenlp file
        doc = json.load(open(corenlp_file, "r"))

    gold_relations = extract_gold_relations_from_ann(ann_file, use_component = use_component)

    offset2sentid = get_offset2sentid(doc = doc, corenlp_file = corenlp_file)

    intrasent_gold_relations = []
    for e1, e2, relation in gold_relations:
        sentid1 = get_sentid_from_offset(e1['doc_start_char'], e1['doc_end_char'], offset2sentid)
        sentid2 = get_sentid_from_offset(e2['doc_start_char'], e2['doc_end_char'], offset2sentid)

        if sentid1 is None or sentid2 is None: 
            continue 
        if sentid1 == sentid2:
            # new_e1 = deepcopy(e1)
            # new_e2 = deepcopy(e2)
            # new_e1["sentid"] = sentid1
            # new_e2["sentid"] = sentid2
            # intrasent_gold_relations.append((new_e1, new_e2, relation))
            e1['sentid'] = sentid1
            e2['sentid'] = sentid2 
            intrasent_gold_relations.append((e1, e2, relation))
    return intrasent_gold_relations

def get_offset2sentid(doc = None, corenlp_file = None):

    """ get a dictonary that maps character offset to sentid """

    if doc is None and corenlp_file is None:
        raise NameError("Either doc_dict or corenlp_file must be provided ! ")
    if doc is None:
        # read corenlp file
        doc = json.load(open(corenlp_file, "r"))

    offset2sentid = {}
    for sent in doc["sentences"]:
        offset = (sent["tokens"][0]["characterOffsetBegin"],  sent["tokens"][-1]["characterOffsetEnd"])
        sentid = int(sent["index"]) 

        assert offset not in offset2sentid
        offset2sentid[offset] = sentid
    return offset2sentid

def get_sentid_from_offset(doc_start_char, doc_end_char, offset2sentid):
    """ This function gets the sent index given the character offset of an entity in the document 
    
    Args:
        doc_start_char: starting character offset of the entity in the document 
       
        doc_end_char: ending character offset of the entity in the document 
       
        offset2sentid: a mapping from character offset to sent index in the document 
    """
    sentid = None
    for offset in offset2sentid:
        if offset[0] <= doc_start_char < doc_end_char <= offset[1]:
            return offset2sentid[offset]
    return sentid 

def get_offset2docidx(doc = None, corenlp_file = None):
    """ 
    This function gets the mapping from the document-level offset of a word to its document-level word index 
    """

    # get a dictonary of character offset to sentid
    if doc is None and corenlp_file is None:
        raise NameError("Either doc_dict or corenlp_file must be provided ! ")
    if doc is None:
        # read corenlp file
        doc = json.load(open(corenlp_file, "r"))

    offset2idx = {}
    for sent in doc["sentences"]:
        for tok in sent['tokens']:
            offset = (tok["characterOffsetBegin"],  tok["characterOffsetEnd"])
            offset2idx[offset] = len(offset2idx)

    return offset2idx

def get_docidx_from_offset(doc_start_char, doc_end_char, offset2docidx):
    """
        This function gets the starting and ending document-level word indices for an entity given the entity's document-level offset. 
    """
    begin_idx = None
    end_idx = None # exclusive
    for offset in offset2docidx:
        if offset[0] <= doc_start_char < offset[1]:
            begin_idx = offset2docidx[offset]
        if  offset[0] < doc_end_char <= offset[1]:
            end_idx = offset2docidx[offset]
    return (begin_idx, end_idx)

def get_sentid2start2endNer(entities):
    """
    this function maps sent_start_idx to entity's end, if there is anym and also the entity ner type

    """
    sentid2start2endNer = {}
    for e in entities:
        sentid = e['sentid']
        if sentid not in sentid2start2endNer:
            sentid2start2endNer[sentid] = {}
        start = e['sent_start_idx']
        end = e['sent_end_idx']
        ner = e['label']

        if start not in sentid2start2endNer[sentid]:
            sentid2start2endNer[sentid][start] = []

        sentid2start2endNer[sentid][start].append((end, ner))

def construct_intrasent_entity_pairs(entities):
    intrasent_entitypairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if entities[i]['sentid'] != entities[j]['sentid']:
                continue
            if need_swap_entity(entities[i]['label'], entities[j]['label']):
                intrasent_entitypairs.append((deepcopy(entities[j]), deepcopy(entities[i])))
            else:
                intrasent_entitypairs.append((deepcopy(entities[i]), deepcopy(entities[j])))
    return intrasent_entitypairs

def extract_intrasent_entitypairs_from_text_file(text_file, ann_file, doc = None, corenlp_file = None, use_component = True):
    
    """ 
    This function extract all pairs of entities from the same sentence as relation candidates. note that here we would NOT get duplicated entity pairs with reverse order, such as (t1, t2) and (t2, t1). Entity pairs would be sorted by their ner labels.
    
    Args: 
        text_file: text_file that contains the journal/abstract 
        
        ann_file: file that contains annotations from brat. used to extract ners 

        doc:  a dictionary that stores corenlp's parsing for text_file

        corenlp_file: is a file that stores corenlp's parsing for text_file
    """

    if doc is None and corenlp_file is None:
        raise NameError("Either doc_dict or corenlp_file must be provided ! ")
    if doc is None:
        # read corenlp file
        doc = json.load(open(corenlp_file, "r"))

    entities = extract_entities_from_text(text_file, ann_file, doc = doc, corenlp_file = corenlp_file, use_component = use_component)

    # get all possible entity pairs. 
    return construct_intrasent_entity_pairs(entities)


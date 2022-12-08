import re, sys, random, os 
from os.path import abspath, dirname
from copy import deepcopy
from extraction_utils import * 

def get_input_ids(tokenizer, toks, max_len = 512):
    tokidx2bertidx = [] #token idx to (leftmost wordpiece index in input_ids, rightmost wordpiece index in input_ids) (left = inclusive, right = exclusive)
    input_ids = [tokenizer.convert_tokens_to_ids([tokenizer.cls_token])[0]]
    for i, tok in enumerate(toks):
            
        ids = tokenizer.encode(tok, add_special_tokens = False)
        if len(input_ids) + len(ids) + 1 > max_len:
            break
        tokidx2bertidx.append([len(input_ids), len(input_ids) + len(ids)])
            
        input_ids.extend(ids)
    input_ids.append(tokenizer.convert_tokens_to_ids([tokenizer.sep_token])[0])
    return input_ids, tokidx2bertidx

def truncate(temp_prespan_ids, temp_posspan_ids, num_cut):
    # this function truncates previous and pos-context iteratively for  num_cut times . NOTE, the ids are assume to come with [CLS] and [SEP], and the truncation would not touch these two tokens
    prespan_ids = temp_prespan_ids[:]
    posspan_ids = temp_posspan_ids[:]

    while num_cut and (len(prespan_ids) > 1 or len(posspan_ids) > 1):

        if len(prespan_ids) > 1:
            prespan_ids.pop(1)
            num_cut -= 1

        if num_cut == 0:
            break
        if len(posspan_ids) > 1:
            posspan_ids.pop(-2)
            num_cut -= 1
        if num_cut == 0:
            break
    return prespan_ids, posspan_ids, num_cut


class Span:
    def __init__(self, docId, docStartChar, docEndChar, text, stdText, ner, sentToks = None, sentId = None, sentStartIdx = None, sentEndIdx = None, relationLabel = None, sentStr = None, startOffset = None, endOffset = None, sentLemmas = None, sentPos = None):
        """ 
        This class is designed to store information for an entity such as Target, Element and Component

        Args:
            docid: year_docname
            doc_start_char: 
                starting character offset of the entity in the document 
            doc_end_char:
                ending character offset of the entity in the document 
            text:
                text of the entity 
            ner_label: 
                ner label of the entity 
            sent_toks:
                list of words of the sentence that contains the entity 
            sentid:
                sent index of the sentence that contains the entity 
            sent_start_idx:
                the starting word index of the entity in the sentence
            sent_end_idx:
                the ending word index of the entity in the sentence
        """
        self.docId = docId
        self.id = f"{self.docId}-{docStartChar}-{docEndChar}"
        self.docStartChar = docStartChar
        self.docEndChar = docEndChar
        self.text = text
        self.ner = ner
        # self.relation_label_is_noisy = None
        self.stdText = stdText

        self.sentToks = sentToks
        self.sentLemmas = sentLemmas
        self.sentPos = sentPos
        self.sentId = sentId
        self.sentStartIdx = sentStartIdx
        self.sentEndIdx = sentEndIdx
        self.sentStr = sentStr
        self.startOffset = startOffset
        self.endOffset = endOffset
        if self.startOffset is not None and self.endOffset is not None and self.sentStr is not None:
            assert self.sentStr[self.startOffset:self.endOffset] == self.text 

        self.bertStartIdx = None # location of < of <e>
        self.bertEndIdx = None
        self.decoded = [] # this is the decoded sentence, which may not be used as input to bert 
        self.decodedInput = [] # this is the input to bert 
        self.relationLabel = relationLabel
        self.predRelationLabel = None 

    def keep_k_tokens_around(self, kTokens):
        start = max(self.sentStartIdx - kTokens, 0)
        end = min(self.sentEndIdx + kTokens, len(self.sentToks))
        self.sentToks = self.sentToks[start:end]
        self.sentStartIdx -= start 
        self.sentEndIdx -= start 

    def insert_type_markers(self, tokenizer, useStdText = True, maxLen = 512, maskNer = False):
        """
            This function inserts type markers such as <Target> around the entity in the sentence 

            use_std_text: whether to substitute the entity's text with its canonical name in the sentence. for example, 
            if use_std_text is true, then the sentence 'A contains K' would be turned into 'A contains <T>Potassium<\\T>'
        """
        assert self.sentToks is not None
        self.inputIds = []
        exceedLeng = 0 
        preSpans = tokenizer.tokenize(" ".join(["[CLS]"] + self.sentToks[:self.sentStartIdx]))
        startMarkers = [f"<ner_start={self.ner.lower()}>"]
        if useStdText:
            spans = tokenizer.tokenize(self.stdText)
        else:
            spans = tokenizer.tokenize(" ".join(self.sentToks[self.sentStartIdx:self.sentEndIdx]))
        
        if maskNer:
            spans = [f"<{self.ner.lower()}>"]

        endMarkers = [f"<ner_end={self.ner.lower()}>"]
        posSpans = tokenizer.tokenize(' '.join(self.sentToks[self.sentEndIdx:] + ["[SEP]"]))
        self.decoded = preSpans + startMarkers + spans + endMarkers + posSpans 
        self.decodedInput = preSpans + startMarkers + spans + endMarkers + posSpans
        if len(preSpans + startMarkers + spans + endMarkers + posSpans) > maxLen:
            # truncate now 
            diff = len(preSpans + startMarkers + spans + endMarkers + posSpans) - maxLen

            prepSans, posSpans, diff = truncate(preSpans, posSpans, diff)

        self.inputIds = tokenizer.convert_tokens_to_ids(preSpans + startMarkers + spans + endMarkers + posSpans)
        self.bertStartIdx = len(preSpans)
        self.bertEndIdx = len(preSpans + startMarkers + spans)


        assert tokenizer.convert_ids_to_tokens(self.inputIds)[self.bertStartIdx] == f"<ner_start={self.ner.lower()}>" and tokenizer.convert_ids_to_tokens(self.inputIds)[self.bertEndIdx] == f"<ner_end={self.ner.lower()}>"

        # if input_ids is longger than the maximum length, simply use the 0th vector to represent the entity 
        if len(self.inputIds) > maxLen:
            exceedLeng = 1
            self.inputIds = self.inputIds[: maxLen]
            
            if self.bertStartIdx >= maxLen:
                self.bertStartIdx = 0
            
            if self.bertEndIdx >= maxLen:
                self.bertEndIdx = 0
        
        return exceedLeng

    def insert_type_markers_by_offset(self, tokenizer, useStdText = True, maxLen = 512, maskNer = False):
        """
            This function inserts type markers such as <Target> around the entity in the sentence (un-tokenized)

            use_std_text: whether to substitute the entity's text with its canonical name in the sentence. for example, 
            if use_std_text is true, then the sentence 'A contains K' would be turned into 'A contains <T>Potassium<\\T>'
        """
        assert self.sentStr is not None and self.startOffset is not None and self.endOffset is not None 
        preSpans = tokenizer.tokenize(f"[CLS] {self.sentStr[:self.startOffset]}")
        startMarkers = [f"<ner_start={self.ner.lower()}>"]
        if maskNer:
            spans = tokenizer.tokenize(f"<{self.ner.lower()}>")
        else:
            spans = tokenizer.tokenize(self.text if not useStdText else self.stdText)
        endMarkers = [f"<ner_end={self.ner.lower()}>"]
        posSpans = tokenizer.tokenize(f"{self.sentStr[self.endOffset:]} [SEP]")


        self.inputIds = []
        exceedLeng = 0 
        self.decoded = preSpans + startMarkers + spans + endMarkers + posSpans 
        self.decodedInput = preSpans + startMarkers + spans + endMarkers + posSpans
        if len(preSpans + startMarkers + spans + endMarkers + posSpans) > maxLen:
            # truncate now 
            diff = len(preSpans + startMarkers + spans + endMarkers + posSpans) - maxLen

            prepSans, posSpans, diff = truncate(preSpans, posSpans, diff)

        self.inputIds = tokenizer.convert_tokens_to_ids(preSpans + startMarkers + spans + endMarkers + posSpans)
        self.bertStartIdx = len(preSpans)
        self.bertEndIdx = len(preSpans + startMarkers + spans)


        assert tokenizer.convert_ids_to_tokens(self.inputIds)[self.bertStartIdx] == f"<ner_start={self.ner.lower()}>" and tokenizer.convert_ids_to_tokens(self.inputIds)[self.bertEndIdx] == f"<ner_end={self.ner.lower()}>"

        # if input_ids is longger than the maximum length, simply use the 0th vector to represent the entity 
        if len(self.inputIds) > maxLen:
            exceedLeng = 1
            self.inputIds = self.inputIds[: maxLen]
            
            if self.bertStartIdx >= maxLen:
                self.bertStartIdx = 0
            
            if self.bertEndIdx >= maxLen:
                self.bertEndIdx = 0
        
        return exceedLeng



    def __str__(self): 
        return f"DOCID: {self.docId}\nTEXT: {self.text}, STDTEXT: {self.stdText}, NER:{self.ner}, Relation Label:{self.relationLabel}, ({self.docStartChar}, {self.docEndChar}), SENTID: {self.sentId}\nSENTENCE TOKENS:\n    {'' if self.sentToks is None else ' '.join(self.sentToks) }\nSENTENCE: {self.sentStr if self.sentStr else ''}\nDECODED: \n    {' '.join(self.decoded)}\nDECODED INPUT TO BERT: \n    {' '.join(self.decodedInput)}\nSTART, END OFFSET: ({self.docStartChar, self.docEndChar})\n"

class Relation:
    def __init__(self,span1, span2, relationLabel = None):

        """
            This is a class to store information of a relation. A relation instance contains two entities, denotd as span1 and span2 

            Args:
                span1: 
                    target span instance
                span2: 
                    component span instance 
                label_str: 
                    relation label such as 'Contains' and 'O'
        """
        self.span1 = span1
        self.span2 = span2
        self.sentToks = self.span1.sentToks
        self.relationLabel = relationLabel  
        self.decodedInput = None 
        self.docId = self.span1.docId 
        self.id = f"{self.span1.id}||{self.span2.id}"
         
    def insert_type_markers(self, tokenizer, useStdText = True, maxLen = 512, maskNer = False):
        self.sentToks = self.span1.sentToks
        self.inputIds = []
        exceedLeng = 0 
        span1, span2 = [self.span1, self.span2] if self.span1.sentStartIdx < self.span2.sentStartIdx else [self.span2, self.span1]
        if maskNer:
            span1Words = tokenizer.tokenize(f"<{span1.ner.lower()}>")
            span2Words = tokenizer.tokenize(f"<{span2.ner.lower()}>")

        else:
            span1Words = tokenizer.tokenize(span1.stdText) if useStdText else tokenizer.tokenize(span1.text)
            span2Words = tokenizer.tokenize(span2.stdText) if useStdText else tokenizer.tokenize(span2.text)

        preSpan1 = tokenizer.tokenize(" ".join(["[CLS]"] + self.sentToks[:span1.sentStartIdx]))
        startMarker1 = [f"<ner_start={span1.ner.lower()}>"]
        endMarker1 = [f"<ner_end={span1.ner.lower()}>"]
        postSpan1 = tokenizer.tokenize(" ".join(self.sentToks[span1.sentEndIdx:span2.sentStartIdx]))
        startMarker2 = [f"<ner_start={span2.ner.lower()}>"]
        endMarker2 = [f"<ner_end={span2.ner.lower()}>"]
        postSpan2 = tokenizer.tokenize(" ".join(self.sentToks[span2.sentEndIdx:]+ ["[SEP]"]))

        self.decodedInput = preSpan1 + startMarker1 + span1Words + endMarker1 + postSpan1 + startMarker2 + span2Words + endMarker2 + postSpan2 
        if len(self.decodedInput) > maxLen:
            # truncate now 
            diff = len(self.decodedInput) - maxLen

            preSpan1, postSpan2, diff = truncate(preSpan1, postSpan2, diff)

        self.inputIds = tokenizer.convert_tokens_to_ids(preSpan1 + startMarker1 + span1Words + endMarker1 + postSpan1 + startMarker2 + span2Words + endMarker2 + postSpan2)

        # if input_ids is longger than the maximum length, simply use the 0th vector to represent the entity 
        if len(self.inputIds) > maxLen:
            exceedLeng = 1
            self.inputIds = self.inputIds[: maxLen]    

        return exceedLeng



    def __str__(self):
        sentence = ""
        if self.span1.sentToks: 
            sentence = " ".join(self.span1.sentToks)
        elif self.span1.sentStr:
            sentence = self.span1.sentStr

        string = f"DOCID   :{self.span1.docId}\n  sentence:{sentence}\n  Marked Sentence: {'' if  self.decodedInput is None else ' '.join(self.decodedInput)}\n  Span1 = {self.span1.text}, Offset: {'' if self.span1.startOffset is None or self.span1.endOffset is None else (self.span1.startOffset, self.span1.endOffset)}, In Relation: {self.span1.relationLabel if self.span1.relationLabel else ''}\n  Span 2:{self.span2.text}, Offset: {'' if self.span2.startOffset is None or self.span2.endOffset is None else (self.span2.startOffset, self.span2.endOffset)}, In Relation: {self.span2.relationLabel if self.span2.relationLabel else ''}\n  Binary Relation: {self.relationLabel if self.relationLabel is not None else ''}\n\n"
        
        return string
"""
documents
"""
class Document:

    def __init__(self):
        self.docId = None
        self.sentences = [] 
        self.relation2evalBinaryRelations = {} # this includes O cases in chemprot 
        self.relation2inferBinaryRelations = {}
        self.entities = [] 
    
    """chemprot """
    def make_doc_from_chemprot(self,docId, docId2sentId2annots, validRelationSet = set(['CPR:3','CPR:4','CPR:5','CPR:6', 'CPR:9'])):
        """
        This function creates a document. sentence2relations could be produced by read_chemprot 
        """
        self.docId = docId
        # note that self.sentences is not really ordered. We creat fake sentIds.
        sentId2annots = docId2sentId2annots[docId]
        numSentences = len(sentId2annots)
        sentIds = sorted([sid for sid in sentId2annots.keys()])
        # NOTE: the list sentId may not be a contiguous sequence. for example, we could have [6] as the sentIds
        self.sentences = [None for _ in range(max(sentIds)+1)] 
        for sentId in sentIds:
            self.sentences[sentId] = sentId2annots[sentId]['sentence']

        binaryId2relation = {}
        binaryId2inRelations = {} # map binary relation id to relation
        unaryId2inRelations = {} # map unary id to relation annotation 
        seenEntityIds = set()
        for sentId in sentIds:
            sent = sentId2annots[sentId]['sentence']
            relations = sentId2annots[sentId]['relations']

            for (chemStartOffset, chemEndOffset), (geneStartOffset, geneEndOffset), pairId,     relation in relations:
                chemStartOffset, chemEndOffset, geneStartOffset, geneEndOffset = int(chemStartOffset), int(chemEndOffset), int(geneStartOffset), int(geneEndOffset)

                if relation not in self.relation2evalBinaryRelations:
                    self.relation2evalBinaryRelations[relation] = []
                chemText = sent[chemStartOffset:chemEndOffset]
                geneText = sent[geneStartOffset:geneEndOffset]
                
                chem = Span(self.docId, f"sent{sentId}|{chemStartOffset}", f"sent{sentId}|{chemEndOffset}", chemText, chemText, 'CHEMICAL', sentId = sentId, sentStr = sent, startOffset = chemStartOffset, endOffset = chemEndOffset)

                gene = Span(self.docId, f"sent{sentId}|{geneStartOffset}", f"sent{sentId}|{geneEndOffset}", geneText, geneText, 'GENE', sentId = sentId, sentStr = sent, startOffset = geneStartOffset,endOffset = geneEndOffset)

                if need_swap_entity("CHEMICAL", "GENE"):
                    u1, u2 = gene, chem
                else:
                    u1, u2 = chem, gene

                for u in [u1, u2]:
                    if u.id not in unaryId2inRelations:
                        unaryId2inRelations[u.id] = set()
                    unaryId2inRelations[u.id].add(relation)

                    if u.id not in seenEntityIds:
                        self.entities.append(u)
                        seenEntityIds.add(u.id)

                binaryRelation = Relation(u1, u2,relationLabel = relation)
                docId, sentId, eId1, eId2 = pairId.split(".")
                binaryRelation.taskId = f"{docId}.{eId1}.{eId2}" # taskId is different from id as id is manually defined by me (which makes some coding easier), while taskId is provided by the original dataset  
                bid = binaryRelation.id 
                if bid not in binaryId2inRelations:
                    binaryId2inRelations[bid] = set()
                binaryId2inRelations[bid].add(relation)

                if bid not in binaryId2relation:
                    binaryId2relation[bid] = binaryRelation
                self.relation2evalBinaryRelations[relation].append(binaryRelation)


        # construct relation2inferBinaryRelations. At this point, relation2evalBinaryRelations contains all positive cases for each relation. So we could make relation2inferBinaryRelations by adding negative cases into each relation 

        for relation in validRelationSet:
            self.relation2inferBinaryRelations[relation] = []
            # iterate through all possible relations
            for bid, tempBinaryRelation in  binaryId2relation.items(): 
                binaryRelation = deepcopy(tempBinaryRelation)
                if relation not in binaryId2inRelations[bid]:
                    binaryRelation.relationLabel = 'O'
                else:
                    binaryRelation.relationLabel = relation

                for u in [binaryRelation.span1, binaryRelation.span2]:
                    if relation in unaryId2inRelations[u.id]:
                        u.relationLabel = relation 
                    else:
                        u.relationLabel = 'O'
                self.relation2inferBinaryRelations[relation].append(binaryRelation)



    @staticmethod
    def read_chemprot(fileWithOrgText):
        """
            This function reads in the whole file (which keeps the entity names instead of masking them out), and aggregates annotation lines into the same sentences 
        """
        docId2sentId2annots = {}
        relationSet = set()
        def split_line(line):
            pairId, text, start1, end1, ner1, start2, end2, ner2, relation = line.strip().split("\t")
            text = text.strip()
            if relation == 'false':
                relation = 'O'
            docId, sentId, eId1, eId2 = pairId.split(".") # here docId is the sentence id
            sentId = int(sentId)
            if ner1 == 'CHEMICAL':
                chemId, chemStartOffset, chemEndOffset, geneId, geneStartOffset, geneEndOffset = eId1, start1, end1, eId2, start2, end2 
            else:
                chemId, chemStartOffset, chemEndOffset, geneId, geneStartOffset, geneEndOffset = eId2, start2, end2, eId1, start1, end1 

            return docId, sentId, pairId, chemId, geneId, text, chemStartOffset, chemEndOffset, geneStartOffset, geneEndOffset, relation
        
        with open(fileWithOrgText) as f:
            for i, annotLine in enumerate(f):
                annotLine = annotLine.strip() 
                docId, sentId, pairId, chemId, geneId, sentence, chemStartOffset, chemEndOffset, geneStartOffset, geneEndOffset, relation = split_line(annotLine)
                relationSet.add(relation)
                if docId not in docId2sentId2annots:
                    docId2sentId2annots[docId] = {}
                if sentId not in docId2sentId2annots[docId]:
                    docId2sentId2annots[docId][sentId] = {'sentence': sentence,
                        'relations': []}
                else:
                    assert sentence == docId2sentId2annots[docId][sentId]['sentence']
                docId2sentId2annots[docId][sentId]['relations'].append(((chemStartOffset, chemEndOffset), (geneStartOffset, geneEndOffset), pairId, relation))
        print(f"Getting {len(docId2sentId2annots)} unqiue documents")
        # print stats
        relationCounts = {}
        numPairs = 0 
        for _, sentId2annots in docId2sentId2annots.items():
            for _, annots in sentId2annots.items():
                for _, _, pairId, relation in annots['relations']:
                    relationCounts[relation] = relationCounts.get(relation, 0) + 1 
                    numPairs += 1 
        for relation in sorted(relationCounts):
            print(f"{relation}: {relationCounts[relation]}")
        print(f"Collects in total {numPairs} pairs of annotations in {fileWithOrgText}")
        return docId2sentId2annots

class MteDocument:

    def __init__(self, annFiles, textFiles, corenlpFiles, validRelationSet = set([('Target', 'Component', 'Contains')])):
        # annFiles should refer to the same document
        self.validRelationSet = set()
        self.validNerSet = set()
        for a, b, r in validRelationSet:
            if need_swap_entity(a,b):
                b,a = a,b
            self.validRelationSet.add((a,b,r))
            self.validNerSet.add(a)
            self.validNerSet.add(b)
        self.year, self.docName, self.docId = get_docid(annFiles[0])
        self.doc = json.load(open(corenlpFiles[0])) # we just use the first corenlpFile here, since all corenlpFiles are produced by parsing the same text file (parsing results produced by parse_texts.py)
        self.sentences = [[token['word'] for token in self.doc['sentences'][sentid]['tokens']] for sentid in range(len(self.doc['sentences']))] 
        self.sentLemmas = [[token['lemma'] for token in self.doc['sentences'][sentid]['tokens']] for sentid in range(len(self.doc['sentences']))] 
        self.sentPos = [[token['pos'] for token in self.doc['sentences'][sentid]['tokens']] for sentid in range(len(self.doc['sentences']))] 

        self.relation2evalBinaryRelations = self.load_relation2evalBinaryRelations(annFiles, textFiles[0], doc = self.doc, validRelationSet = self.validRelationSet) # relation for evaluation
        self.relation2evalUnaryInstances = self.load_relation2evalUnaryInstances(self.relation2evalBinaryRelations) # relation for evaluation
        self.entities = self.load_entities(annFiles, textFiles[0], doc = self.doc) # load entities from text files and parse trees
        self.relation2inferBinaryRelations = self.load_relation2inferBinaryRelations(annFiles, textFiles[0], self.entities, self.validRelationSet, self.relation2evalBinaryRelations, doc = self.doc) # load binary relation instances for inference, where each binary relation contains two unary instances


    def load_entities(self, annFiles, textFile, corenlpFile = None, doc = None):
        """
            This gets all possible entities for annFiles 
        """
        entities = []
        seenEntities = set()
        if doc is None:
            doc = json.load(open(corenlpFile))

        for annFile in annFiles:
            for e in extract_entities_from_text(textFile, annFile, doc = doc):
                sign = f"{e['doc_start_char']}_{e['doc_end_char']}||{e['label']}"
                if sign in seenEntities:
                    continue 
                seenEntities.add(sign)
                entities.append(e)
        return entities

    def load_relation2evalBinaryRelations(self, annFiles, textFile, corenlpFile = None, doc = None, validRelationSet = None):
        """
            This function creates dictionary mapping from different types of relations to its evaluation data, which is a deduplicated list of (entityDict1, entityDict2). entityDict1 and entityDict2 are sorted by their ner labels and they participate in a relation of the specific relation type
        """
        if doc is None:
            doc = json.load(open(corenlpFile))
            
        relation2evalBinaryRelations = {}
        seen = set() 
        for annFile in annFiles:
            for e1, e2, relation in extract_intrasent_goldrelations_from_ann(annFile, doc = doc): # e1 and e2 have been sorted by their ner labels
                if validRelationSet is not None and (e1['label'], e2['label'], relation) not in validRelationSet:
                    continue 
                relationSign = f"{e1['doc_start_char']}_{e1['doc_end_char']}_{e1['label']}||{e2['doc_start_char']}_{e2['doc_end_char']}_{e2['label']}||{relation}"
                if relationSign not in seen: 
                    seen.add(relationSign)
                else:
                    continue 
                if relation not in relation2evalBinaryRelations:
                    relation2evalBinaryRelations[relation] = []
                u1 = Span(self.docId, e1['doc_start_char'], e1['doc_end_char'], e1['text'], e1['std_text'], e1['label'], sentToks = self.sentences[e1['sentid']], sentId = e1['sentid'], relationLabel = relation, sentLemmas = self.sentLemmas[e1['sentid']], sentPos = self.sentPos[e1['sentid']])
                u2 = Span(self.docId, e2['doc_start_char'], e2['doc_end_char'], e2['text'], e2['std_text'], e2['label'], sentToks = self.sentences[e2['sentid']], sentId = e2['sentid'], relationLabel = relation, sentLemmas = self.sentLemmas[e2['sentid']], sentPos = self.sentPos[e2['sentid']])
                binaryRelation = Relation(u1, u2, relationLabel = relation)
                relation2evalBinaryRelations[relation].append(binaryRelation)

        return relation2evalBinaryRelations
            
    def load_relation2evalUnaryInstances(self,relation2evalBinaryRelations):
        relation2evalUnaryInstances = {}
        seenSigns = set()
        for relation in relation2evalBinaryRelations:
            if relation not in relation2evalUnaryInstances:
                relation2evalUnaryInstances[relation] = [] 
            for binaryRelation in relation2evalBinaryRelations[relation]:
                for u in [binaryRelation.span1,binaryRelation.span2]:
                    sign = f"{u.id}_{u.ner}||{relation}"
                    if sign in seenSigns:
                        continue 
                    seenSigns.add(sign)
                    relation2evalUnaryInstances[relation].append(u)
        return relation2evalUnaryInstances
   
    def load_relation2inferBinaryRelations(self, annFiles, textFile, entities, validRelationSet, relation2evalBinaryRelations, doc = None, corenlpFile = None):
        # This function extract all intrasent pairs of (e1,e2) that fit in validRelationSet from textFile, and form the binary relation instances. And these binary relation instances contains unary instances with assigned labels 
        goldBinarySigns = set() # keep track the binary relation instances that are in relations
        goldUnarySigns = set() # keep track the unary instances that are in relations
        for relation in relation2evalBinaryRelations:
            for binaryRelation in relation2evalBinaryRelations[relation]:
                u1, u2 = binaryRelation.span1, binaryRelation.span2 
                e1sign = f"{u1.id}_{u1.ner}||{relation}"
                e2sign = f"{u2.id}_{u2.ner}||{relation}"
                relationSign = f"{e1sign}<>{e2sign}"
                goldBinarySigns.add(relationSign)
                goldUnarySigns.add(e1sign)
                goldUnarySigns.add(e2sign)

        validNers2relations = {}
        for ner1, ner2, relation in validRelationSet:
            if (ner1,ner2) not in validNers2relations:
                validNers2relations[(ner1, ner2)] = set()
            validNers2relations[(ner1, ner2)].add(relation)

        relation2inferBinaryRelations = {}
        for e1, e2 in construct_intrasent_entity_pairs(entities):
            for relation in validNers2relations.get((e1['label'], e2['label']), []):
                if relation not in relation2inferBinaryRelations:
                    relation2inferBinaryRelations[relation] = [] 
                # relation2inferBinaryRelations[r].append((e1,e2)) 

                # form unary instance first 
                u1 = Span(self.docId, e1['doc_start_char'], e1['doc_end_char'], e1['text'], e1['std_text'], e1['label'], sentToks = self.sentences[e1['sentid']], sentId = e1['sentid'], sentStartIdx = e1['sent_start_idx'], sentEndIdx = e1['sent_end_idx'], sentLemmas = self.sentLemmas[e1['sentid']], sentPos = self.sentPos[e1['sentid']])
                u2 = Span(self.docId, e2['doc_start_char'], e2['doc_end_char'], e2['text'], e2['std_text'], e2['label'], sentToks = self.sentences[e2['sentid']], sentId = e2['sentid'], sentStartIdx = e2['sent_start_idx'], sentEndIdx = e2['sent_end_idx'], sentLemmas = self.sentLemmas[e2['sentid']], sentPos = self.sentPos[e2['sentid']])
                e1sign = f"{u1.id}_{u1.ner}||{relation}"
                e2sign = f"{u2.id}_{u2.ner}||{relation}"
                relationSign = f"{e1sign}<>{e2sign}"
                e1RelationLabel = relation if e1sign in goldUnarySigns else 'O'
                e2RelationLabel = relation if e2sign in goldUnarySigns else 'O'  
                u1.relationLabel = e1RelationLabel
                u2.relationLabel = e2RelationLabel
                
                binaryRelationLabel = relation if relationSign in goldBinarySigns else 'O'
                rel = Relation(u1, u2, relationLabel = binaryRelationLabel)
                relation2inferBinaryRelations[relation].append(rel)
        return relation2inferBinaryRelations


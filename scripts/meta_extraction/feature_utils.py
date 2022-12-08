import numpy as np 
from sklearn import preprocessing
def dist(s1, s2, useTokDist = True):
    min_ = None 
    if useTokDist:
        for a in [s1.sentStartIdx, s1.sentEndIdx]:
            for b in [s2.sentStartIdx, s2.sentEndIdx]:
                d = abs(a-b)
                if min_ is None or d < min_:
                    min_ = d 
    else:
        for a in [s1.docStartChar, s1.docEndChar]:
            for b in [s2.docStartChar, s2.docEndChar]:
                d = abs(a-b)
                if min_ is None or d < min_:
                    min_ = d 
    return min_

# extra features
def get_quantile_dist(instances,numQuantiles = None, dist2idx = None):
    assert numQuantiles is not None or dist2idx is not None 
    # if is training, then find the quantiles
    dists = [dist(ins.span1, ins.span2) for ins in instances]
    if dist2idx is None:
        quant = 1/numQuantiles
        quantiles = [quant * (i+1) for i in range(numQuantiles)]
        boundaries = [-1]
        for q in quantiles:
            boundaries.append(np.quantile(dists, q))
        dist2idx = {(boundaries[i-1], boundaries[i]): i-1 for i in range(1, len(boundaries))}

    distVecs = []
    size = len(dist2idx)
    for d in dists: 
        found = 0
        zeros = [0 for _ in range(size)]
        for (a,b), idx in dist2idx.items():
            if a < d <= b:
                zeros[idx] = 1 
                break 
        distVecs.append(zeros)

    return distVecs, dist2idx 

def bucketize_numbers_by_quantiles(numbers, numQuantiles = None, range2idx = None):
    assert numQuantiles is not None or range2idx is not None 
    # if is training, then find the quantiles
    if range2idx is None:
        quant = 1/numQuantiles
        quantiles = [quant * (i+1) for i in range(numQuantiles)]
        boundaries = [-1]
        for q in quantiles:
            boundaries.append(np.quantile(numbers, q))
        range2idx = {(boundaries[i-1], boundaries[i]): i-1 for i in range(1, len(boundaries))}

    vecs = []
    size = len(range2idx)
    for n in numbers: 
        found = 0
        zeros = [0 for _ in range(size)]
        for (a,b), idx in range2idx.items():
            if a < n <= b:
                zeros[idx] = 1 
                break 
        vecs.append(zeros)

    return vecs, range2idx 


def get_extra_feature(instances,featureFig, u1threshold, u2threshold):
    spanId2otherIdPred = {} # collect score between all pairs of instances predicted by the binary model 
    for instance in instances:
        span1, span2 = instance.span1, instance.span2
        for span, otherSpan in zip([span1, span2], [span2, span1]):
            if span.id not in spanId2otherIdPred:
                spanId2otherIdPred[span.id] = {}
            spanId2otherIdPred[span.id][otherSpan.id] = instance.predScore[1]

    e1Closests = [] # is e1 the entity with ner1 closest to e2
    e2Closests = [] # is e2 the entity with ner2 closest to e2 
    e1PrecedingE2s = [] # if e1 is at the left of e2 
    e1e2IsTheOnlyPairs = [] # if e1 and e2 is the only pair
    numPairs = [] # number of component and target pairs
    anyEntitiesAround = [] # if there are any target/component at the left/right of e1 and e2 
    negations = [] # if there is any negation word 'no' and 'not' between e1 and e2 

    # if e1 in another relation, is there any conjunction between e2 and the other e12
    canFormPairs = []  
    for instance in instances:
        span1, span2 = instance.span1, instance.span2 
        ner1 = span1.ner 
        ner2 = span2.ner 
        sortedSpans = sorted([span for span in instance.entities if span.ner in [ner1, ner2]], key = lambda x: (x.docStartChar, x.docEndChar))
        
        numNer1 = len([span.stdText for span in sortedSpans if span.ner == ner1])
        numNer2 = len([span.stdText for span in sortedSpans if span.ner == ner2])
        intermediateToks = set([ w.lower() for w in instance.sentToks[min(instance.span1.sentStartIdx, instance.span2.sentStartIdx):max(instance.span1.sentEndIdx, instance.span2.sentEndIdx)]])
        span1Idx = None 
        span2Idx = None 
        
        for i, span in enumerate(sortedSpans):
            if span.id == span1.id:
                span1Idx = i 
            if span.id == span2.id: 
                span2Idx = i 
        assert span1Idx is not None and span2Idx is not None 

        # if there is any negation between e1 and e2 
        wordSet = set(intermediateToks)
        negated = 0 
        for nw in ['no', 'not', 'none', 'nothing', 'never', 'nowhere','hardly', 'barely', 'scarcely']:
            if nw in wordSet:
                negated = 1
                break 
        negations.append([negated])
        
        # numPairs
        numPair = numNer1 * numNer2
        if numPair == 1:
            numPairs.append([1,0,0,0,0])
        elif numPair <= 2:
            numPairs.append([0,1,0,0,0])
        elif numPair <= 4:
            numPairs.append([0,0,1,0,0])
        elif numPair <= 10:
            numPairs.append([0,0,0,1,0])
        else:
            numPairs.append([0,0,0,0,1])

        # if there are entities of the other type at the left and right of the current entity
        e1HasLeft = 1 if len([1 for span in sortedSpans[:span1Idx] if span.ner == ner2]) else 0 
        e1HasRight = 1 if len([1 for span in sortedSpans[span1Idx+1:] if span.ner == ner2]) else 0 
        e2HasLeft = 1 if len([1 for span in sortedSpans[:span2Idx] if span.ner == ner1]) else 0 
        e2HasRight = 1 if len([1 for span in sortedSpans[span2Idx+1:] if span.ner == ner1]) else 0 
        anyEntitiesAround.append([e1HasLeft, e1HasRight, e2HasLeft, e2HasRight])

        e1Closest = 1 # is e1 the entity of the type that is closest to e2?
        e2Closest = 1
        charDist = dist(span1, span2,useTokDist = False)
        wordDist = dist(span1, span2, useTokDist = True)
        for i in range(len(sortedSpans)):
            if  sortedSpans[i].ner == ner1 and dist(sortedSpans[i], span2, useTokDist = False) < charDist:
                e1Closest = 0 
                break 
        for i in range(len(sortedSpans)):
            if sortedSpans[i].ner == ner2 and dist(sortedSpans[i], span1, useTokDist = False) < charDist:
                e2Closest = 0 
                break
        e1Closests.append([e1Closest])
        e2Closests.append([e2Closest])
        # can e1 and e2 form a relation pair based on the distance and unary model predictions 
        if e1Closest and span2.predScore[1] >= u2threshold:
            canFormFromE2 = 1 
        else:
            canFormFromE2 = 0 
        if e2Closest and span1.predScore[1] >= u1threshold:
            canFormFromE1 = 1 
        else:
            canFormFromE1 = 0 
        canFormPairs.append([canFormFromE1, canFormFromE2])

        e1PrecedingE2s.append([1 if span1.sentStartIdx < span2.sentStartIdx else 0])

        e1e2IsTheOnlyPairs.append([1 if len(set([span.stdText for span in sortedSpans if span.ner == ner1])) == 1 and len(set([span.stdText for span in sortedSpans if span.ner == ner2])) == 1 else 0])

    distVecs, distRange2idx = bucketize_numbers_by_quantiles([dist(ins.span1, ins.span2) for ins in instances],numQuantiles = featureFig.numDistQuantiles, range2idx = None if not hasattr(featureFig, 'distRange2idx') else featureFig.distRange2idx)

    otherFeatures = [
            distVecs, 
            negations, 
            anyEntitiesAround,
            numPairs, 
            e1e2IsTheOnlyPairs, 
            e1PrecedingE2s, 
            e1Closests,  
            e2Closests]
    unaryFeatures = canFormPairs

    if not hasattr(featureFig, 'distRange2idx'):
        featureFig.distRange2idx = distRange2idx

    otherFeatures.append(unaryFeatures)
    X = np.hstack(otherFeatures).tolist() 
    return X

def form_features(inferInstances, featureFig):

    u1threshold=0.5
    u2threshold=0.5
    scoreFeatures = [] 
    for ins in inferInstances:
        unaryScore1 = ins.span1.predScore 
        unaryScore2 = ins.span2.predScore 
        binaryScore = ins.predScore
        scoreFeature = [] 
        scoreFeature.extend([unaryScore1[1], unaryScore2[1]])
        scoreFeature.append(binaryScore[1])
        scoreFeatures.append(scoreFeature)

    extraFeatures = get_extra_feature(inferInstances, featureFig, u1threshold, u2threshold)
    features = [scoreFeature + extraFeature for scoreFeature, extraFeature in zip(scoreFeatures, extraFeatures)] 
    X = np.array(features)
    return X
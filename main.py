import sys, io, math
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import csv, re
from scipy.stats import entropy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import roc_auc_score, matthews_corrcoef, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from logistic import LogisticRegression
from linear import LinearRegression
from collections import Counter

'''
Label Distribution
    rows 	unrelated 	discuss 	agree 	disagree
    49972 	0.73131 	0.17828 	0.0736012 	0.0168094
'''
label_headers = ['unrelated','discuss','agree','disagree']

def main():
    bodies, stances, index, body_IDs, stance_IDs, labels = generateSentences()
    encoder = LabelEncoder()
    encoder.fit(label_headers)
    encoded_labels = encoder.transform(labels)

    combined = matchStance(bodies,stances,body_IDs,stance_IDs)
    sorted_bodies = linkBodies(body_IDs,stance_IDs,bodies)

    training_bodies = sorted_bodies[0:index+1]
    training_stances = stances[0:index+1]
    training_labels = encoded_labels[0:index+1]

    cv,tfidf = vectorise(training_bodies,training_stances,training_labels)

    b_cv = cv.transform(training_bodies)
    s_cv = cv.transform(training_stances)
    b_tf = tfidf.transform(b_cv)
    s_tf = tfidf.transform(s_cv)
    cosineSim(training_bodies,training_stances,cv,training_labels,'plots/CS-vect.png')
    cosineSim(b_tf,s_tf,tfidf,training_labels,'plots/CS-tfidf.png')

    kldivergence(b_cv.toarray(),s_cv.toarray(),training_labels,'plots/KL-vect.png')
    kldivergence(b_tf.toarray(),s_tf.toarray(),training_labels,'plots/KL-tfidf.png')

    # valid_bodies = sorted_bodies[index+1:len(sorted_bodies)]
    # valid_stances = stances[index+1:len(stances)]
    valid_labels = encoded_labels[index+1:len(encoded_labels)]
    valid_b_cv = cv.transform(sorted_bodies)
    valid_s_cv = cv.transform(stances)
    valid_b_tf = list(tfidf.transform(valid_b_cv).toarray())
    valid_s_tf = list(tfidf.transform(valid_s_cv).toarray())

    dists = calcDistances(valid_b_tf,valid_s_tf)
    distanceShow(dists[0:index+1],training_labels)
    logistic = LogisticRegression(lr=0.0095,steps=10000)
    logistic.fit(input=dists[0:index+1],labels=training_labels)
    y_pred = logistic.predict(dists[index+1:len(dists)])

    linear = LinearRegression(lr=0.095,steps=10000)
    linear.fit(input=dists[0:index+1],labels=training_labels)
    y_pred2 = linear.predict(dists[index+1:len(dists)])

    print len(y_pred2),list(y_pred2)
    print(classification_report(y_true=list(valid_labels),y_pred=list(y_pred)))
    # print(classification_report(y_true=list(valid_labels),y_pred=list(y_pred2)))



def vectorise(bodies,stances,labels):
    cv = CountVectorizer(max_features=1000,stop_words='english',ngram_range=(1, 7))
    combined = [str(stances[i] + bodies[i]) for i in range(0,len(stances))]
    c_train_count = cv.fit_transform(combined)
    print "Shape of Count Matrix: {}".format(c_train_count.shape)
    tfidf = TfidfTransformer(smooth_idf=True)
    c_train_tf = tfidf.fit_transform(c_train_count)
    print "Shape of TF-IDF Matrix: {}".format(c_train_tf.shape)
    return cv,tfidf

def kldivergence(bodies,stances,labels,save):
    divs = []
    eps = 0.0000001
    for i in range(0,len(stances)):
        modified_b = [x+eps for x in bodies[i]]
        modified_s = [x+eps for x in stances[i]]
        ent = entropy(modified_s,modified_b)
        # ent = entropy(bodies[i],stances[i])
        divs += [ent]
        # if (i % 10000 == 0): print i
    average_div = np.mean(divs)
    print "Average KL-Divergence (Total): {}, Max: {}, Min: {}".format(average_div, max(divs),min(divs))
    box_sets = [[y for x,y in sorted(zip(labels,divs)) if x == i] for i in range(0,4)]
    print "Average KL-Divergence (Per Class): {}".format([np.mean(box_sets[i]) for i in range(0,4)])
    pyplot.figure()
    box = pyplot.boxplot(box_sets,labels=label_headers,showmeans=True)
    pyplot.savefig(save)

def distanceShow(dists,labels):
    average_dists = np.mean(dists,axis=1)
    funLabels = ['SumAbsDiff','SumSqrDiff','MeanSAD','MeanSSD','Euclidean','Chebyschev','Minkowski','Canberra']
    print "Average Distances: {}".format(zip(average_dists,funLabels))
    new_dists = np.swapaxes(dists,0,1)
    for i in range(0,len(funLabels)):
        pyplot.figure()
        pyplot.hist(new_dists[i],color='green',bins=200)
        pyplot.xlabel('Distance')
        pyplot.ylabel('Probability')
        pyplot.grid(True)
        pyplot.savefig("plots/dist-{}-overall.png".format(funLabels[i]))
        pyplot.close("all")
        pyplot.figure()
        box_sets = [[y for x,y in sorted(zip(labels,new_dists[i])) if x == k] for k in range(0,4)]
        # bins = np.linspace(-10, 10, 100)
        for j in range(len(box_sets)-1,-1,-1):
            pyplot.hist(box_sets[j],bins=100,label=sorted(label_headers)[j],alpha=0.75)
            pyplot.xlabel('Distance')
            pyplot.ylabel('Probability')
            pyplot.grid(True)
        pyplot.legend(loc='upper right')
        pyplot.savefig("plots/dist-{}-class.png".format(funLabels[i],label_headers[j]))
        pyplot.close("all")

def calcDistances(bodies,stances):
    dists = []
    distFunc = [SumAbsDiff,SumSqrDiff,MeanSAD,MeanSSD,Euclidean,Chebyschev,Minkowski,Canberra]
    eps = 0.0000001
    for i in range(0,len(stances)):
        modified_b = [x+eps for x in bodies[i]]
        modified_s = [x+eps for x in stances[i]]
        dist = []
        for f in distFunc:
            dist += [f(modified_b,modified_s)]
        dists += [dist]
    return dists

def plotDists(dists):
    pyplot.figure()
    for i in range(0,len(dists[0])):
        curr_metric = []

# Distance Metrics
def SumAbsDiff(body,stance):
    return sum([abs(x-y) for x,y in zip(stance,body)])

def SumSqrDiff(body,stance):
    return sum([(x-y)**2 for x,y in zip(stance,body)])

def MeanSAD(body,stance):
    return SumAbsDiff(body,stance) / float(len(stance))

def MeanSSD(body,stance):
    return SumSqrDiff(body,stance) / float(len(stance))

def Euclidean(body,stance):
    return math.sqrt(SumSqrDiff(body,stance))

def Chebyschev(body,stance):
    return max([abs(x-y) for x,y in zip(stance,body)])

def Minkowski(body,stance,n=3):
    return sum([(abs(x-y)) ** n for x,y in zip(stance,body)]) ** (1.0 / n)

def Canberra(body,stance):
    return sum([abs(x-y)/(abs(x) + abs(y)) for x,y in zip(stance,body)])

def linkBodies(b_id,s_id,bodies):
    linked = []
    for i in range(0,len(s_id)):
        linked += [bodies[b_id.index(s_id[i])]]
    return linked

def cosineSim(bodies,stances,model,labels, save):
    length = 0
    if not (type(bodies) == list):
        ax1 = [bodies.getnnz(axis=1),stances.getnnz(axis=1)]
        assert(len(ax1[0]) == len(ax1[1]))
        length = len(ax1[0])
    else:
        assert(len(bodies) == len(stances))
        length = len(stances)
    b_vector = model.transform(bodies).todense()
    s_vector = model.transform(stances).todense()
    sims = []
    for i in range(0,length):
        sims += [cosine_similarity(s_vector[i],b_vector[i])[0][0]]
    average_sims = np.mean(sims)
    min_sims = min(sims)
    max_sims = max(sims)
    print "Average Cosine Similarity (Total): {}, Max: {}, Min: {}".format(average_sims,max_sims,min_sims)
    box_sets = [[y for x,y in sorted(zip(labels,sims)) if x == i] for i in range(0,4)]
    print "Average Cosine (Per Class): {}".format([np.mean(box_sets[i]) for i in range(0,4)])
    pyplot.figure()
    box = pyplot.boxplot(box_sets,labels=label_headers,showmeans=True)
    pyplot.savefig(save)

def matchStance(bodies,stances,body_IDs,stance_IDs):
    combined = []
    for i in range(0,len(stances)):
        combined += [stances[i] + ' ' + bodies[body_IDs.index(stance_IDs[i])]]
    return combined

def splitSet(training):
    validation_index = int(math.floor(0.9 * len(training)))
    print "Validation index is {}".format(validation_index)
    print "Distribution for training set: {}".format(countClass(training[0:validation_index]))
    print "Distribution for validation set: {}".format(countClass(training[validation_index+1:len(training)]))
    return validation_index

def countClass(data):
    # 0 = Unrelated, 1 = Discuss, 2 = Agree, 3 = Disagree
    classes = [0.0] * 4
    for row in data:
        if (row[2] == 'unrelated'):
            classes[0] += 1
        elif (row[2] == 'discuss'):
            classes[1] += 1
        elif (row[2] == 'agree'):
            classes[2] += 1
        elif (row[2] == 'disagree'):
            classes[3] += 1
    distr = [i*100/sum(classes) for i in classes]
    return distr

def generateSentences():
    bodies = readBody()
    stances = readStance()
    stance_sentences = [str(stances[k][0]) for k in range(len(stances))]
    body_sentences = [str(bodies[k][1]) for k in range(len(bodies))]
    index = splitSet(stances)
    print "Bodies Size: {}, Stances Size: {}, Split Index: {}".format(len(bodies),len(stances), index)
    return body_sentences, stance_sentences, index, [bodies[i][0] for i in range(0,len(bodies))], [stances[j][1] for j in range(0,len(stances))], [stances[k][2] for k in range(0,len(stances))]

def readBody():
    with open('train_bodies.csv','r') as body_file:
        content = list(csv.reader(body_file, delimiter=',',quotechar='"'))
        content = content[1:len(content)]
        for i in range(0,len(content)):
            content[i][1] = re.sub(r'[^\w\s]',' ',content[i][1]).replace('\n','').lower()
    return content

def readStance():
    with open('train_stances.csv','r') as stance_file:
        content = list(csv.reader(stance_file, delimiter=',',quotechar='"'))
        content = content[1:len(content)]
        for i in range(0,len(content)):
            content[i][0] = re.sub(r'[^\w ]',' ',content[i][0]).replace('\n','').lower()
    return content

# def activator(x):
# def logistic(X,y):


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
import string, os, time, sys, collections
import numpy as np
from numpy import linalg as LA
from math import *
from mpi4py import *
import os

STOPWORDS = ["a", "able", "about","1","2","3","4","5","6","7","8","9","0","above", "according", "accordingly", "across", "actually", "after", "afterwards", "again", "against", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any", "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear", "appreciate", "appropriate", "are", "around", "as", "aside", "ask", "asking", "associated", "at", "available", "away", "awfully", "b", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "both", "brief", "but", "by", "c", "came", "can", "cannot", "cant", "cause", "causes", "certain", "certainly", "changes", "clearly", "co", "com", "come", "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "course", "currently", "d", "definitely", "described", "despite", "did", "different", "do", "does", "doing", "done", "down", "downwards", "during", "e", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially", "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "f", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for", "former", "formerly", "forth", "four", "from", "further", "furthermore", "g", "get", "gets", "getting", "given", "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "h", "had", "happens", "hardly", "has", "have", "having", "he", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "hi", "him", "himself", "his", "hither", "hopefully", "how", "howbeit", "however", "i", "ie", "if", "ignored", "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar", "instead", "into", "inward", "is", "it", "its", "itself", "j", "just", "k", "keep", "keeps", "kept", "know", "knows", "known", "l", "last", "lately", "later", "latter", "latterly", "least", "less", "lest", "let", "like", "liked", "likely", "little", "ll", "look", "looking", "looks", "ltd", "m", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover", "most", "mostly", "much", "must", "my", "myself", "n", "name", "namely", "nd", "near", "nearly", "necessary", "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none", "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "o", "obviously", "of", "off", "often", "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "p", "particular", "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably", "provides", "q", "que", "quite", "qv", "r", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless", "regards", "relatively", "respectively", "right", "s", "said", "same", "saw", "say", "saying", "says", "second", "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "since", "six", "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure", "t", "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "theres", "thereupon", "these", "they", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "twice", "two", "u", "un", "under", "unfortunately", "unless", "unlikely", "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "uucp", "v", "value", "various", "ve", "very", "via", "viz", "vs", "w", "want", "wants", "was", "way", "we", "welcome", "well", "went", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "willing", "wish", "with", "within", "without", "wonder", "would", "would", "x", "y", "yes", "yet", "you", "your", "yours", "yourself", "yourselves", "z", "zero"]
MY_PY_SET_LIMIT= os.getenv('MY_PY_SET_LIMIT')



if MY_PY_SET_LIMIT != None:
     resource.setrlimit(resource.RLIMIT_STACK, (int(MY_PY_SET_LIMIT), int(MY_PY_SET_LIMIT)))
     
def jaccard_similarity(x,y):
    productoPunto = float(np.dot(list(x),list(y)))
    magnitudA = float(LA.norm(list(x)))
    magnitudB = float(LA.norm(list(y)))
    return productoPunto/((magnitudA**2)+(magnitudB**2)+ productoPunto)

def kMeans(X, K, maxIters = 10, plot_progress = None):

    centroids = X[np.random.choice(np.arange(len(X)), K), :]
    for i in range(maxIters):
        # Cluster Assignment step
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])
        # Move centroids step
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]
        if plot_progress != None: plot_progress(X, C, np.array(centroids))
    return np.array(centroids) , C
words = []
arr = []
mapa = dict()

def organizar(file):      
      content = open(str(sys.argv[1]+file))
      txt = content.read().lower()
      words = txt.replace("\r\n", " ").replace("\t"," ").replace("\n"," ").replace("\r","").replace("$","").replace("!","").replace("-","").replace("[","").replace("]","").replace(".","").replace(",","").replace(":","").replace(";","").replace("_","").replace("*","").replace("+","").replace("'","").replace("?","").replace("¿","").split()
      return words

lista = []
termino = False
terminoTodo = False
def leerArchivo():
      comm = MPI.COMM_WORLD
      size = comm.Get_size()
      rank = comm.Get_rank()
      
      if rank == 0:
            data = os.listdir(str(sys.argv[1]))
            chunks = [[] for _ in range(size)]
            for i, chunk in enumerate(data):
                  chunks[i % size].append(chunk)
      else:
            data = None
            chunks = None
      data = comm.scatter(chunks, root=0)
      for i in range(0,len(data)):
            print(data[i])
            data[i] = organizar(data[i])      
      newData = comm.gather(data,root=0)
      if rank==0:
            return newData
def main():
      palabras = []
      arrTemp = leerArchivo()
      if arrTemp:
            print(len(arrTemp))
            for item in arrTemp:
                  for fil in item:
                        arr.append(fil)
            print(len(arr))
            for i in range(len(arr)):
                  for j in range(len(arr[i])):
                        if arr[i][j] not in STOPWORDS:
                              palabras.append(arr[i][j])
            mapa = collections.Counter(palabras)
            tMayu = dict(mapa.most_common(40)).keys()
            fdt = []
            for doc in arr:
                  result = []
                  for i in range(len(tMayu)):
                        result.append(0)
                  for word in doc:
                        if word not in STOPWORDS:
                              if word in tMayu:
                                    result[tMayu.index(word)] += 1
                  fdt.append(result)
            mat = np.empty((len(fdt),len(fdt)))
            for i in range(len(fdt)):
                  for j in range(len(fdt)):
                        mat[i][j] = 1-jaccard_similarity(fdt[i],fdt[j])
            print(mat)
            cents, C = kMeans(mat, 3)
            print(C)
            print(cents)
timeIni = time.time()
main()
print(time.time() - timeIni)

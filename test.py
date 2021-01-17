# https://pymupdf.readthedocs.io/en/latest/faq.html

import fitz
import sys
from math import floor
import re

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


class Chapter:
    def __init__(self, number, name, pageStart, pageEnd=None):

        assert isinstance(number, int), ('Error: Type error: %s' % (type(number)))
        assert isinstance(pageStart, int), ('Error: Type error: %s' % (type(pageStart)))
        assert isinstance(name, str), ('Error: Type error: %s' % (type(name)))

        self.number = number
        self.name = name
        self.pageRange = [pageStart, pageEnd]

    def __repr__(self):        
        return " ".join(["[Chapter %d]"%self.number, self.name, str(self.pageRange)])

# function to remove stopwords
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def main():
    #opening the file
    filename = sys.argv[1]
    try:
        doc = fitz.open(filename)
    except Exception:
        exit("Invalid file. Please ensure that file is .pdf format.")    
    
    #test pdf file by checking for text
    pagenum = doc.pageCount
    #choosing 4 different pages
    sample = (floor(2/6*pagenum), floor(3/6*pagenum), floor(4/6*pagenum), floor(5/6*pagenum))
    blankPages= 0
    #checking if text is successfully extracted
    for page in sample:
        text = doc[page].getText()
        if text == "":
            blankPages+=1
    if blankPages==len(sample):
        exit("This pdf is not readable, sorry")
    
    #getting all the chapter information
    chts = []
    chapter_count=0
    toc = doc.get_toc()
    for n, items in enumerate(toc):
        if items[0]==1 and n < len(toc)-1 and toc[n+1][0]==2: # lvl 1 in document hierarchy AND search for section that has subsections
            chapter_count+=1
            if chapter_count>=2:
                chts[-1].pageRange[1] = items[2]-1
            c = Chapter(chapter_count, items[1], items[2])
            chts.append(c)
        if chapter_count>=1 and items[0]==1: # finds the end of last chapter
            chts[-1].pageRange[1] = items[2]-1
    
    #displaying all chapters
    [print(c) for c in chts]

    #asking user for specific chapter
    chapter = None
    while not chapter:
        try:
            x = int(input(f"Please enter a chapter (1-{chapter_count}): "))
            if x>=1 and x<=chapter_count:
                chapter = x
            else: raise Exception
        except Exception:
            print("Invalid chapter")

    #writing all the text to string from chapter
    output = ""
    for page in range(chts[chapter-1].pageRange[0]-1, chts[chapter-1].pageRange[1]):
        text = doc[page].getText()
        output+=text 
    print("Working on Summary...")
    
    #https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/ adapted the code
    #from lines 94 to 137
    
    #split big string into list of string sentences
    sentences = []
    sentences = sent_tokenize(str(output))
    
    # Extract word vectors from data
    word_embeddings = {}
    f = open('glove.6B.100d.txt',encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    
    #clean each sentence of unneeded chars
    clean_sentences = []
    for s in sentences:
        temp = re.sub('[^a-zA-Z]+', ' ', s)
        clean_sentences.append(temp)
    clean_sentences = [s.lower() for s in clean_sentences]
    # remove stopwords from the sentences
    stop_words = stopwords.words('english')
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences] 
    
    #calculate sentence vectors
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v) 
        
    #similarity matrix for scoring the text
    sim_mat = np.zeros([len(sentences), len(sentences)])  
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]  
                
    #run pagerank algorithm and get important sentences
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    
    #ask user for summary length
    sum_length = None
    while not sum_length:
        try:
            x = int(input(f"There are {len(sentences)} in this chapter. Please enter how many sentences you want in the summary: "))
            if x>=1 and x<=len(sentences):
                sum_length = x
            else: raise Exception
        except Exception:
            print("Invalid number")    

    #put summary to file
    chapter_sum = chts[chapter-1]
    output = open("chapter_summary" + ".txt", "wb")
    title_sum = "Chapter "+ str(chapter_sum.number)+ " Title: "+ chapter_sum.name+" " + str(chapter_sum.pageRange) + "\n"
    title_sum = bytes(title_sum, 'utf-8')
    output.write(title_sum)
    output.write(bytes("Number of Sentences: "+ str(sum_length) + "\n","utf-8"))
    for i in range(sum_length):
        output.write(bytes("\n"+ranked_sentences[i][1]+'\n','utf-8'))
    output.close()
    print("The summary has been printed to chapter_summary.txt!")
main()
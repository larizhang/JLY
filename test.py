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

#https://www.analyticsvidhya.com/blog/2018/11/introduction-text-summarization-textrank-python/ adapted the code

class Chapter:
    def __init__(self, number, name, pageStart, pageEnd=None):

        assert isinstance(number, int), ('Error: Type error: %s' % (type(number)))
        assert isinstance(pageStart, int), ('Error: Type error: %s' % (type(pageStart)))
        # assert isinstance(pageEnd, int), ('Error: Type error: %s' % (type(pageEnd)))
        assert isinstance(name, str), ('Error: Type error: %s' % (type(name)))

        self.number = number
        self.name = name
        self.pageRange = [pageStart, pageEnd]
        # self.sub = []

    def __repr__(self):        
        return " ".join(["[Chapter %d]"%self.number, self.name, str(self.pageRange)])

# function to remove stopwords
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new



def main():
    #filename = sys.argv[1]
    filename = "irbookonlinereading.pdf"
    doc = fitz.open(filename)
    pagenum = doc.pageCount
    sample = (floor(2/6*pagenum), floor(3/6*pagenum), floor(4/6*pagenum), floor(5/6*pagenum))
    blankPages= 0
    for page in sample:
        text = doc[page].getText()
        if text == "":
            blankPages+=1
    if blankPages==len(sample):
        raise Exception("This pdf is not readable, sorry")
    
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
    
    [print(c) for c in chts]

    chapter = None
    while not chapter:
        try:
            x = int(input(f"Please enter a chapter (1-{chapter_count}): "))
            if x>=1 and x<=chapter_count:
                chapter = x
            else: raise Exception
        except Exception:
            print("Invalid chapter")

    """
    output = open("string_contents" + ".txt", "wb")
    for page in range(chts[chapter-1].pageRange[0], chts[chapter-1].pageRange[1]+1):
        text = doc[page].getText().encode("utf8")
        output.write(text)

    output.close()
    """
    output = ""
    for page in range(chts[chapter-1].pageRange[0]-1, chts[chapter-1].pageRange[1]):
        text = doc[page].getText()
        output+=text  
    #to be removed

    test_string = """
The meaning of the term information retrieval can be very broad. Just getting a credit card out of your wallet so that you can type in the card number is a form of information retrieval. However, as an academic field of study, information retrieval might be defined thus:

Information retrieval (IR) is finding material (usually documents) of an unstructured nature (usually text) that satisfies an information need from within large collections (usually stored on computers).
As defined in this way, information retrieval used to be an activity that only a few people engaged in: reference librarians, paralegals, and similar professional searchers. Now the world has changed, and hundreds of millions of people engage in information retrieval every day when they use a web search engine or search their email.[*]Information retrieval is fast becoming the dominant form of information access, overtaking traditional database-style searching (the sort that is going on when a clerk says to you: ``I'm sorry, I can only look up your order if you can give me your Order ID'').
IR can also cover other kinds of data and information problems beyond that specified in the core definition above. The term ``unstructured data'' refers to data which does not have clear, semantically overt, easy-for-a-computer structure. It is the opposite of structured data, the canonical example of which is a relational database, of the sort companies usually use to maintain product inventories and personnel records. In reality, almost no data are truly ``unstructured''. This is definitely true of all text data if you count the latent linguistic structure of human languages. But even accepting that the intended notion of structure is overt structure, most text has structure, such as headings and paragraphs and footnotes, which is commonly represented in documents by explicit markup (such as the coding underlying web pages). IR is also used to facilitate ``semistructured'' search such as finding a document where the title contains Java and the body contains threading.

The field of information retrieval also covers supporting users in browsing or filtering document collections or further processing a set of retrieved documents. Given a set of documents, clustering is the task of coming up with a good grouping of the documents based on their contents. It is similar to arranging books on a bookshelf according to their topic. Given a set of topics, standing information needs, or other categories (such as suitability of texts for different age groups), classification is the task of deciding which class(es), if any, each of a set of documents belongs to. It is often approached by first manually classifying some documents and then hoping to be able to classify new documents automatically.

Information retrieval systems can also be distinguished by the scale at which they operate, and it is useful to distinguish three prominent scales. In web search , the system has to provide search over billions of documents stored on millions of computers. Distinctive issues are needing to gather documents for indexing, being able to build systems that work efficiently at this enormous scale, and handling particular aspects of the web, such as the exploitation of hypertext and not being fooled by site providers manipulating page content in an attempt to boost their search engine rankings, given the commercial importance of the web. We focus on all these issues in webcharlink. At the other extreme is personal information retrieval . In the last few years, consumer operating systems have integrated information retrieval (such as Apple's Mac OS X Spotlight or Windows Vista's Instant Search). Email programs usually not only provide search but also text classification: they at least provide a spam (junk mail) filter, and commonly also provide either manual or automatic means for classifying mail so that it can be placed directly into particular folders. Distinctive issues here include handling the broad range of document types on a typical personal computer, and making the search system maintenance free and sufficiently lightweight in terms of startup, processing, and disk space usage that it can run on one machine without annoying its owner. In between is the space of enterprise, institutional, and domain-specific search , where retrieval might be provided for collections such as a corporation's internal documents, a database of patents, or research articles on biochemistry. In this case, the documents will typically be stored on centralized file systems and one or a handful of dedicated machines will provide search over the collection. This book contains techniques of value over this whole spectrum, but our coverage of some aspects of parallel and distributed search in web-scale search systems is comparatively light owing to the relatively small published literature on the details of such systems. However, outside of a handful of web search companies, a software developer is most likely to encounter the personal search and enterprise scenarios.

In this chapter we begin with a very simple example of an information retrieval problem, and introduce the idea of a term-document matrix (Section 1.1 ) and the central inverted index data structure (Section 1.2 ). We will then examine the Boolean retrieval model and how Boolean queries are processed ( and 1.4 ).
"""
    sentences = []
    sentences = sent_tokenize(str(output))
    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt',encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()
    clean_sentences = []
    for s in sentences:
        temp = re.sub('[^a-zA-Z]+', ' ', s)
        clean_sentences.append(temp)
    clean_sentences = [s.lower() for s in clean_sentences]
    stop_words = stopwords.words('english') 
    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences] 
    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)   
    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])  
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]  
                
    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    # Extract top 10 sentences as the summary
    for i in range(10):
        print("Sentence" + str(i))
        print(ranked_sentences[i][1])     

main()
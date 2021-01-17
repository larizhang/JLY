# https://pymupdf.readthedocs.io/en/latest/faq.html


import fitz
import sys
from math import floor
import re

class Chapter:
    def __init__(self, number, name, pageStart, pageEnd=None):

        assert isinstance(number, int), ('Error: Type error: %s' % (type(number)))
        assert isinstance(pageStart, int), ('Error: Type error: %s' % (type(pageStart)))
        # assert isinstance(pageEnd, int), ('Error: Type error: %s' % (type(pageEnd)))
        assert isinstance(name, str), ('Error: Type error: %s' % (type(name)))

        self.number = number
        self.name = name
        self.pageRange = (pageStart, pageEnd)
        self.sub = []

    def __repr__(self):        
        return " ".join(["[Chapter %d]"%self.number, self.name, str(self.pageRange)])


def main():
    page = 15
    filename = sys.argv[1]
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
    
    output = open("summary" + ".txt", "wb")
    print(doc.get_toc())
    
    chts = []
    chapter_count=0
    toc = doc.get_toc()
    for n, items in enumerate(toc):
        if items[0]==1 and n < len(toc)-1 and toc[n+1][0]==2: # lvl 1 in document hierarchy AND search for section that has subsections
            chapter_count+=1
            c = Chapter(chapter_count, items[1], items[2])
            chts.append(c)
    [print(c) for c in chts]


    # text = doc[page].getText().encode("utf8")
    # output.write(text)
    output.close()

main()
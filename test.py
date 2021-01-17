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
        self.pageRange = [pageStart, pageEnd]
        # self.sub = []

    def __repr__(self):        
        return " ".join(["[Chapter %d]"%self.number, self.name, str(self.pageRange)])


def main():
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

    output = open("string_contents" + ".txt", "wb")
    for page in range(chts[chapter-1].pageRange[0], chts[chapter-1].pageRange[1]+1):
        text = doc[page].getText().encode("utf8")
        output.write(text)

    output.close()

main()
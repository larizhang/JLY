# https://pymupdf.readthedocs.io/en/latest/faq.html


import fitz

def main():
    page = 15
    filename = input("Enter filename: ")
    doc = fitz.open(filename)
    output = open("summary" + ".txt", "wb")
    text = doc[page].getText().encode("utf8")
    output.write(text)
    output.close()

main()
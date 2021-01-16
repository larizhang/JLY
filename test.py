# https://pymupdf.readthedocs.io/en/latest/faq.html


import fitz

def main():
    print("hello world")
    filename = input("Enter filename: ")
    doc = fitz.open(filename)
    output = open(filename[:10] + ".txt", "wb")
    text = doc[15].getText().encode("utf8")
    output.write(text)
    output.close()

main()
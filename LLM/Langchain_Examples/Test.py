import numpy as np
import docx2txt
import re

def remove_empty_rows(txt: str) -> str:
    return re.sub(r"(\n\s*){3,}", r"\n\n", txt)

def remove_bad_characters(txt: str) -> str:
    pattern = r"[ \n]*—[ \n]*"
    txt = re.sub(pattern, "", txt)
    pattern = r" {1,}"
    txt = re.sub(pattern, " ", txt)
    pattern = r" {2,}"
    txt = re.sub(pattern, " ", txt)
    return txt

if __name__ == '__main__':
#    np.show_runtime()
    txt = docx2txt.process(
    "/Users/shriniwasiyengar/Documents/Personal/Resumes/Shriniwas_Iyengar_Vita-Full Basic VJTI v14.docx")
    txt = remove_empty_rows(txt)
    print(txt)


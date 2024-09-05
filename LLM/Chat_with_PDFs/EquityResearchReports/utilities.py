import os
import sys
from pdf_loaders import PdfToTextLoader
from docx_loaders import DocxToTextLoader

PDF_FILES= []
COMPANY_NAMES = []
TEXT_FILES = []
DOCX_FILES = []

def extract_name(file_name):

    #Extract name of the company from research filename and add to NAMES list
    #the name of the file has the format <Company Name>_<YYYY>_<some text>.pdf
    #the extracted company name may have multiple words separated by hyphen.
    #The company name is extracted from the filename by splitting the filename using '_' and taking the first element   
    try:
        #create a regex pattern to extract the company name from the filename
        #the filename has the pattern <Company Name>-<YYYY>-sometext.pdf
        #For example, if the filename is "3p-learning-2015-db.pdf", the company name is "3p-learning"
        regex_pattern = r"(.+?)-\d{4}-.*"
        import re
        match = re.match(regex_pattern, file_name)
        if match:
            name = match.group(1)
            print("Found research report for company: "+name)
            return name
        else:
            print("No match found for the filename pattern.")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def process_files(directory):
    global COMPANY_NAMES  # Declare CANDIDATE_NAMES as a global variable
    global PDF_FILES  # Declare PDF_FILES as a global variable
    global TEXT_FILES  # Declare TEXT_FILES as a global variable
    global DOCX_FILES  # Declare DOCX_FILES as a global variable

    # Get list of all files in the directory which match the pattern *.pdf
    file_match = ".pdf"
    if os.path.exists(directory):
        file_names = [f for f in os.listdir(directory) if f.endswith(file_match)]
        if not file_names:
            print("No PDF files found in the directory| "+directory)
    else:
        print("The directory does not exist: "+directory)
        sys.exit()

    # Create a list of PDF files
    for file_name in file_names:
        name = extract_name(file_name)
        if name is not None:
            full_path_and_file_name = directory + file_name
            COMPANY_NAMES.append(name)
            PDF_FILES.append(full_path_and_file_name)
            #invoke load_pdf method to load the PDF file
            pdf_loader = PdfToTextLoader(full_path_and_file_name, full_path_and_file_name.replace(".pdf", ".txt"))
            pdf_loader.load_pdf()
            
        else:
            print(f"Name could not be extracted from the filename: {file_name}")

    # Get list of all files in the directory which match the pattern *.docx
    file_match = ".docx"
    if os.path.exists(directory):
        file_names = [f for f in os.listdir(directory) if f.endswith(file_match)]
        if not file_names:
            print("No DOCX files found in the directory| "+directory)
    else:
        print("The directory does not exist: "+directory)
        sys.exit()

    # Create a list of DOCX files
    for file_name in file_names:
        name = extract_name(file_name)
        if name is not None:
            full_path_and_file_name = directory + file_name
            COMPANY_NAMES.append(name)
            DOCX_FILES.append(full_path_and_file_name)
            #invoke load_docx method to load the DOCX file
            docx_loader = DocxToTextLoader(full_path_and_file_name, full_path_and_file_name.replace(".docx", ".txt"))
            docx_loader.load_docx()

        else:
            print(f"Name could not be extracted from the filename: {file_name}")

    # Get list of all files in the directory which match the pattern *.txt
    file_match = ".txt"
    if os.path.exists(directory):
        file_names = [f for f in os.listdir(directory) if f.endswith(file_match)]
        if not file_names:
            print("No TXT files found in the directory| "+directory)
    else:
        print("The directory does not exist: "+directory)
        sys.exit()

    # Create a list of TXT files
    for file_name in file_names:
        name = extract_name(file_name)
        if name is not None:
            full_path_and_file_name = directory + file_name
            COMPANY_NAMES.append(name)
            TEXT_FILES.append(full_path_and_file_name)
        else:
            print(f"Name could not be extracted from the filename: {file_name}")
    
    #remove duplicates from the list of candidate names
    COMPANY_NAMES = list(set(COMPANY_NAMES))

    #remove duplicates from the list of PDF files
    PDF_FILES = list(set(PDF_FILES))

    #remove duplicates from the list of TXT files
    TEXT_FILES = list(set(TEXT_FILES))

    #remove duplicates from the list of DOCX files
    DOCX_FILES = list(set(DOCX_FILES))
                          
    return COMPANY_NAMES, PDF_FILES, DOCX_FILES, TEXT_FILES

if __name__ == '__main__':
    # Specify the directory for data files
    home_dir = os.path.expanduser("~")
    data_dir = "/Users/shriniwasiyengar/git/python_ML/LLM/Chat_with_PDFs/EquityResearchReports/pdf-data/"
    directory = data_dir

    COMPANY_NAMES, PDF_FILES, DOCX_FILES, TEXT_FILES = process_files(directory)
    print("Companies found in the directory: "+str(COMPANY_NAMES))
    print("PDF files found in the directory: "+str(PDF_FILES))
    print("DOCX files found in the directory: "+str(DOCX_FILES))
    print("TXT files found in the directory: "+str(TEXT_FILES)
    )
    print("Processing of files completed successfully")
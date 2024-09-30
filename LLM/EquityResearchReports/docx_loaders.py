from docx import Document

class DocxToTextLoader:
    """
    Class for loading DOCX files and saving them as texts
    """

    def __init__(self, docx_path, output_path):
        """
        Args:
            docx_path (str): path to DOCX file
            output_path (str): path to save text file
        """
        self.docx_path = docx_path
        self.output_path = output_path

    def load_docx(self):
        """
        Loads DOCX file and saves it as text file
        """
        # Open the DOCX file
        doc = Document(self.docx_path)
        # Initialize an empty string to hold the text
        text = ''
        # Iterate through each paragraph in the document
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        
        # Save the text to the output file
        with open(self.output_path, 'w') as text_file:
            text_file.write(text)

# Example usage
# loader = DocxToTextLoader('path/to/your/document.docx', 'path/to/output.txt')
# loader.load_docx()
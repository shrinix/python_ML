import scispacy
import spacy
#Core models
import en_core_sci_sm #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
import en_core_sci_md #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
#NER specific models
import en_ner_bc5cdr_md #pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz

import webbrowser

#Tools for extracting & displaying data
from spacy import displacy
import pandas as pd

mtsample_df=pd.read_csv("./data/mtsamples.csv")

print(mtsample_df.shape)
print(mtsample_df.info())

print(mtsample_df.head(2))

text = mtsample_df.loc[10, "transcription"]
print(text)

# nlp_sm = en_core_sci_sm.load()
# doc = nlp_sm(text)
# #Display resulting entity extraction
# displacy_image = displacy.serve(doc, style='ent', port=5050)

# nlp_md = en_core_sci_md.load()
# doc = nlp_md(text)
# displacy_image = displacy.serve(doc, style='ent', port=5050)

nlp_bc = en_ner_bc5cdr_md.load()
doc = nlp_bc(text)

#get the entities
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(entities)

#parse the text and extract the entities
parsed_text = []
for token in doc:
    parsed_text.append([token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.ent_type_])

print(pd.DataFrame(parsed_text, columns=["Text", "Lemma", "POS", "Tag", "Dep", "EntType"]))

#extract medical conditions, drugs, procedures, body parts, diseases, and symptoms
medical_conditions = [ent.text for ent in doc.ents if ent.label_ == "MEDICAL_CONDITION"]
drugs = [ent.text for ent in doc.ents if ent.label_ == "CHEMICAL"]
procedures = [ent.text for ent in doc.ents if ent.label_ == "PROCEDURE"]
body_parts = [ent.text for ent in doc.ents if ent.label_ == "BODY_PART"]
diseases = [ent.text for ent in doc.ents if ent.label_ == "DISEASE"]
symptoms = [ent.text for ent in doc.ents if ent.label_ == "SYMPTOM"]

print("Medical Conditions: ", medical_conditions)
print("Drugs: ", drugs)
print("Procedures: ", procedures)
print("Body Parts: ", body_parts)
print("Diseases: ", diseases)
print("Symptoms: ", symptoms)


#Display resulting entity extraction
# displacy_image = displacy.serve(doc, style='ent', port=5050)
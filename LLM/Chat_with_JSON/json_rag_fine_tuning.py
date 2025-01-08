from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import login
import os
import json

import torch
from datasets import load_dataset

#HuggingFace transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel , prepare_model_for_kbit_training , get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from trl import SFTTrainer

def metadata_func(record, *args, **kwargs):
    metadata = {}
    metadata["id"] = record.get("id", "")
    metadata["firstname"] = record.get("firstname", "")
    metadata["surname"] = record.get("surname", "")
    metadata["share"] = record.get("share", 0)
    return metadata

#Function to flatten the JSON data which is in nested format.
#The function will take a JSON object as input and return a list of dictionaries.
def flatten_json(json_obj):
    texts = []
    flat_list = []
    for prize in json_obj["prizes"]:
        print("Processing prize: ", prize)
        if "laureates" not in prize:
            flat_dict["year"] = prize["year"]
            flat_dict["category"] = prize["category"]
            flat_dict["overallMotivation"] = prize["overallMotivation"]
            continue
        for laureate in prize["laureates"]:
            print("Processing laureate: ", laureate)
            flat_dict = {}
            flat_dict["year"] = prize["year"]
            flat_dict["category"] = prize["category"]
            flat_dict["id"] = laureate["id"]
            flat_dict["firstname"] = laureate["firstname"]
            #add error handling for missing surname
            if "surname" not in laureate:
                flat_dict["surname"] = ""
            else:
                flat_dict["surname"] = laureate["surname"]
            flat_dict["motivation"] = laureate["motivation"]
            flat_dict["share"] = laureate["share"]
            flat_list.append(flat_dict)

    # # Prepare text for embedding
    #     # text_to_embed = f"{name} is the {office_name}, {division_id}, {party}, {address_text}, {' '.join(phones)}, {' '.join(urls)}"
    #     text_to_embed = f"{flat_dict['firstname']} {flat_dict['surname']} won the {flat_dict['category']} prize in {flat_dict['year']} for {flat_dict['motivation']}"
    #     texts.append(text_to_embed)

    #     documents = [Document(page_content=text) for text in texts]

    # return documents
    return flat_list                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

def load_and_flatten_json(data_folder, file_name):
    print("Loading data...")
    file_path = data_folder + "/" + file_name
    with open(file_path, "r") as f:
        data = json.load(f)

    # data = JSONLoader(file_path=file_path, jq_schema=".prizes[].laureates[]?", content_key="motivation", metadata_func=metadata_func).load()
    flat_list = flatten_json(data)
    # loader = JSONLoader(file_path=data_folder+"/prize.json", jq_schema=".prizes[].laureates[]?", content_key="motivation", metadata_func=metadata_func)
    # documents = loader.load()
    print("Data loaded")
    return flat_list

def create_documents(flat_list):
    documents = []
    for record in flat_list:
        metadata = metadata_func(record)
        content = record["motivation"]
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents

def embed_documents(documents):
    print("Embedding documents...")

    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002",
    #     openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # embedding_function = OllamaEmbeddings(model="llama3.1",)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_function
    )

    retriever = vectorstore.as_retriever()
    print("Documents embedded")
    return retriever

def initialize_chain(template, retriever, use_ollama=False):

    prompt = ChatPromptTemplate.from_template(template)
    # Instantiate the LLM        

    if use_ollama:
        model = ChatOllama(model='llama3.1', config={'max_new_tokens': 20000, 'temperature': 0.0, 'top_p':1.0, 'seed':42, 'context_length': 10000})
    else:
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1000, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0,seed=42)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # rag_chain_from_docs = (
    # RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    # | prompt
    # | model
    # | StrOutputParser()
    # )

    # rag_chain_with_source = RunnableParallel(
    #     {"context": retriever, "question": RunnablePassthrough()}
    # ).assign(answer=rag_chain_from_docs)

    return chain

def get_similar_docs(retriever, query):
    similar_docs = retriever.similarity_search(query)
    return similar_docs

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def parse_documents(documents):
    parsed_docs = []
    for doc in documents:
        metadata = doc.metadata
        page_content = doc.page_content
        print("Page content: ", page_content)
        print("Metadata: ", metadata)
        parsed_doc = f"ID: {metadata['id']}, Name: {metadata['firstname']} {metadata['surname']}, Share: {metadata['share']}\nMotivation: {page_content}"
        parsed_docs.append(parsed_doc)
    return "\n\n".join(parsed_docs)

def fine_tune_model(peft_model_path,new_model):

    print("Starting fine-tuning...")
    # LOAD AND STRUCTURE DATA
    #load data from train.json into a datframe
    # folder = 
    train_ds = load_dataset("json" , data_files = './train.jsonl' , field = "train")
    
    test_ds = load_dataset('json' , data_files = './test.jsonl' , field = "test")
    
    model_id = 'NousResearch/Llama-2-7b-chat-hf'
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_use_double_quant=True,
        bnb_8bit_compute_dtype=torch.bfloat16, #if your gpu supports it 
        bnb_8bit_quant_type = "nf8",
        # bnb_8bit_use_double_quant = False #this quantises the quantised weights
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id)#, device_map="cpu") #quantization_config=bnb_config, device_map="auto"
    
    # Training_tokenizer (https://huggingface.co/docs/transformers/v4.37.2/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)

    # https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        # truncation_side = "right",
        # padding_side="right",
        # add_eos_token=True,
        # add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = 200

    base_model.gradient_checkpointing_enable() #this to checkpoint grads 
    model = prepare_model_for_kbit_training(base_model) #quantising the model (due to compute limits)

    # https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/nn/modules.py#L271
    print(model)

    def printParameters(model):
        trainable_param = 0
        total_params = 0
        for name , param in model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_param += param.numel()
                        
        print(f"Total params : {total_params} , trainable_params : {trainable_param} , trainable % : {100 * trainable_param / total_params} ")

    #LoraConfig : https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py#L44

    # Its better to pass the values to the target_modules either "all-linear" or specific modules you need to quantise !!

    # You can change these parameters depending on your use case
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1, 
        bias="none",
        target_modules=[ 
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
        ],
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model , peft_config)
    printParameters(model)

    # https://github.com/huggingface/transformers/blob/v4.37.2/src/transformers/training_args.py#L161

    # max_steps and num_train_epochs : 
    # 1 epoch = [ training_examples / (no_of_gpu * batch_size_per_device) ] steps

    args = TrainingArguments(
        output_dir = "LLama-2 7b", #Specifies the directory where the model checkpoints will be saved.
        # num_train_epochs=10, #1000
        max_steps = 50, #1000, # comment out this line if you want to train in epochs
        per_device_train_batch_size = 4,
        warmup_steps = 1, #0.03,
        gradient_accumulation_steps = 1,
        logging_steps=10,
        logging_strategy= "steps",
        save_strategy="steps",
        save_steps = 10,
        evaluation_strategy="steps",
        eval_steps=10, # comment out this line if you want to evaluate at the end of each epoch
        learning_rate=2.5e-5,
        # weight_decay = 1e14, #Helps to prevent overfitting by adding a penalty for larger weights. A typical value is 1e-4.
        bf16=True, #if your gpus supports this 
        logging_nan_inf_filter = False, #this helps to see if your loss values is coming out to be nan or inf and if that is the case then you may have ran into some problem 
        # lr_scheduler_type='constant',
        save_safetensors = True,
    )    

    def generate_dataset_prompt(example):
        print("Example: ", example)
        bos_token = "<s>[INST] <<SYS>>"
        user_inst = "You are a custom AI model and your role is to output correct answers for the questions asked by the user regarding Nobel Prizes"
        eoi_token = "<</SYS>>"
        before_prompt = ''  # add your instruction before the actual input (e.g., this is the medical coding, tell me the ..., what is the ...)
        input_prompt = example['Input']
        output_prompt = example['Output']
        eos_token = "</s>"

        data_set = []
        #get each 'Input' string from input_prompt and concatenate it with the bos_token, user_inst, eoi_token, before_prompt, input_prompt, output_prompt, eos_token
        if len(input_prompt) > 0:
            #get input, output string pairs from input_prompt and output_prompt lists
            for input, output in zip(input_prompt, output_prompt):
                input = input + '[/INST]'
                # Ensure all parts are strings before concatenation
                dataset_value = bos_token + user_inst + eoi_token + before_prompt + input + output + eos_token
                data_set.append(dataset_value)

        print("Size of training dataset: ", len(data_set))
        
        return data_set

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        # max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        # packing=True,
        formatting_func=generate_dataset_prompt, # this will apply the generate_dataset_prompt to all training and test dataset mentioned above !!
        args=args,
        train_dataset=train_ds["train"],
        eval_dataset=test_ds["train"]
    )

    model.config.use_cache = False
    trainer.train()
    print("Model fine-tuned successfully")

    #load the trained model and generate some outputs from it 

    # access_token = os.environ["HUGGING_FACE_HUB_TOKEN"]
    # login(token=access_token)

    # ft_model = PeftModel.from_pretrained(base_model)# , 'Checkpoint/base-checkpoint-10') #replace with the actual checkpoint name

    # Fine-tuned model
    # peft_model = PeftModel.from_pretrained(base_model, 
    #                                   peft_model_path+new_model,
    #                                   torch_dtype=torch.bfloat16,
    #                                   is_trainable=False) ## is_trainable mean just a forward pass jsut to get a sumamry
    
    trainer.model.save_pretrained(new_model)
    trainer.tokenizer.save_pretrained(new_model)

    print("Saved fine-tuned model to folder ", new_model)
    return peft_model, tokenizer

if __name__ == "__main__":
    # get_data_from_api()

    #get current working directory
    # import os
    # cwd = os.getcwd()

    # # data_folder = cwd+"/data/"
    # data_folder = "/Users/shriniwasiyengar/git/python_ML/LLM/Chat_with_JSON/data"
    # file_name = "prize.json"
    # flat_list = load_and_flatten_json(data_folder, file_name)

    # # Create Document objects with metadata
    # documents = create_documents(flat_list)
    # print("# of documents: ", len(documents))

    # retriever = embed_documents(documents)

    # template = """Answer the question based only on the following context.
    #              If you don't know the answer, just say that you don't know. 
    #              Use two sentences maximum and keep the answer concise.
    #             Question: {question}
    #             Context: {context} 
    #             Answer:
    #             """
    # use_ollama = False
    # chain = initialize_chain(template, retriever, use_ollama)
    # # query = "What year did albert einstein win the nobel prize?"
    # # query = "Who won the chemistry prize in 2024?"
    # query = "Who won the Nobel prize for peace in 1972?"

    # # similar_docs = get_similar_docs(retriever, query)
    # # print(similar_docs)
    
    # result = chain.invoke(query)
    # print("Result: ", result)   
    # # print(f'Context:\n{parse_documents(result["context"])}')
    # # print(f'Answer: {result["answer"]}')

    # from datasets import Dataset

    # def extract_page_content(documents):
    #     return [doc.page_content for doc in documents]

    # def extract_metadata(documents):
    #     return [doc.metadata for doc in documents]

    # questions = ["What year did Albert Einstein win the nobel prize?", 
    #             "Who won the chemistry prize in 2024?",
    #             "Who won the Nobel prize for peace in 1972?"
    #             ]
    # reference = ["Albert Einstein won the Nobel Prize in 1921.",
    #                 "David Baker, Demis Hassabis and John Jumper were joint winners of the Nobel prize for Chemistry in 2024.",
    #                 "No Nobel Prize was awarded in 1972"]
    # answers = []
    # contexts = []
    # metadata_list = []

    # # Inference
    # for query in questions:
    #     answers.append(chain.invoke(query))
    #     relevant_docs = retriever.get_relevant_documents(query)
    #     contexts.append(extract_page_content(relevant_docs))
    #     metadata_list.append(extract_metadata(relevant_docs))

    # # To dict
    # data = {
    #     "question": questions,
    #     "answer": answers,
    #     "contexts": contexts,
    #     "reference": reference
    # }

    # # Convert dict to dataset
    # dataset = Dataset.from_dict(data)

    # from ragas import evaluate
    # from ragas.metrics import (
    #     faithfulness,
    #     answer_relevancy,
    #     context_recall,
    #     context_precision,
    # )

    # result = evaluate(
    #     dataset = dataset, 
    #     metrics=[
    #         context_precision,
    #         context_recall,
    #         faithfulness,
    #         answer_relevancy,
    #     ],
    # )

    # df = result.to_pandas()
    # print(df)

    # #write df to csv
    # df.to_csv("ragas_test_prizes_before_ft.csv", index=False)

    if torch.backends.mps.is_built():
        # MPS = Metal Performance Shaders, used with Apple Silicon chips like M1, M1 Pro, M1 Max etc.
        device = torch.device("mps")  # for mac use. 
        print("MPS is available")
    else:
        device = torch.device("cpu")  # fallback to CPU if MPS is not available
        print("CPU is available")

    peft_model = []
    tokenizer = []
    peft_model_path = "/Users/shriniwasiyengar/git/python_ML/LLM/Chat_with_JSON/"#'./'
    new_model_name = "llama-2-7b-chat-nobel"
    #check if the fine-tuned model exists in the local folder "./" with the name "llama-2-7b-chat-nobel"    
    try:
        # peft_model = PeftModel.from_pretrained(peft_model_path+"llama-2-7b-chat-nobel")
        model_name = 'NousResearch/Llama-2-7b-chat-hf' #"decapoda-research/llama-7b-hf"
        
        adapters_name = peft_model_path+new_model_name #"lucas0/empath-llama-7b"

        print("Current working directory: ", os.getcwd())

        print(f"Starting to load the model {model_name} into memory")

        m = AutoModelForCausalLM.from_pretrained(
            model_name,
            #load_in_4bit=True,
            torch_dtype=torch.bfloat16,
            # device_map={device: 0},
        )
        peft_model = PeftModel.from_pretrained(m, adapters_name, offload_folder = "offload/")
        # # peft_model = m.merge_and_unload()
        # tok = LlamaTokenizer.from_pretrained(model_name)
        # tok.bos_token_id = 1

        # stop_token_ids = [0]

        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            # truncation_side = "right",
            # padding_side="right",
            # add_eos_token=True,
            # add_bos_token=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        print(f"Successfully loaded the model {model_name} into memory")
    except Exception as e:
        print("Error is: ", e)
        print("Fine-tuned model not found. Starting fine-tuning...")

    # Ensure the device is set to MPS if available
    device = torch.device("mps") if torch.has_mps else torch.device("cpu")

    if not peft_model:
        peft_model,tokenizer = fine_tune_model(peft_model_path, new_model_name)
    else:
       # Move the model to the device
        peft_model.to(device)

    with torch.no_grad():

        print(peft_model.eval())

        queries = ["Who won the Nobel Prize for Physics in 2002?", 
                    "What year did Médecins Sans Frontières win the Nobel Peace Prize?", 
                    "Who won the Nobel Prize for Medicine in 2001?", 
                    "What year did Wisława Szymborska win the Nobel Prize for Literature?", 
                    "Who won the Nobel Prize for Chemistry in 2000?", 
                    "Who won the Nobel Prize for Physics in 2001?", 
                    "What year did Gao Xingjian win the Nobel Prize for Literature?", 
                    "Who won the Nobel Prize for Medicine in 2000?", 
                    "Who won the Nobel Prize for Chemistry in 1999?"]
     
        answers = ["Raymond Davis Jr., Masatoshi Koshiba, and Riccardo Giacconi won the Nobel Prize for Physics in 2002.",
                "1999.",
                "Leland H. Hartwell, Tim Hunt, and Paul Nurse won the Nobel Prize for Medicine in 2001.",
                "Wisława Szymborska won the Nobel Prize for Literature in 1996.",
                "Alan J. Heeger, Alan MacDiarmid, and Hideki Shirakawa won the Nobel Prize for Chemistry in 2000.",
                "Eric A. Cornell, Wolfgang Ketterle, and Carl E. Wieman won the Nobel Prize for Physics in 2001.",
                "2000.",
                "Arvid Carlsson, Paul Greengard, and Eric Kandel won the Nobel Prize for Medicine in 2000.",
                "Ahmed Zewail won the Nobel Prize for Chemistry in 1999."
                ]

        bos_token = "<s>[INST]"
        user_inst = "<<SYS>> You are a custom AI assistant model and your goal is to provide correct answers to \
                    user's questions on Nobel Prizes based on their queries. Provide terse answers. Do not ask questions. \
                    Do not make small conversation. Do not make up answers or use the internet. <</SYS>>"
        eos_token = " [/INST]</s>"

        for query in queries:
            # eval_prompt = "<s>[INST] <<SYS>> You are a custom AI assistant model and your goal is to provide correct answers to user's questions on Nobel Prizes based on the prompt they have entered and you get rewarded for correct output <</SYS>> When did Albert Einstein win the Nobel Prize [/INST]"
            # query = "When did Albert Einstein win the Nobel Prize [/INST]"
            # query = "What year did V. S. Naipaul win the Nobel Prize for Literature?"

            eval_prompt = bos_token + user_inst + query + eos_token
            
            model_input = tokenizer(eval_prompt, return_tensors="pt").to(device)
            # print("Model input: ", model_input)
            # Generate the output
            # generated_ids = peft_model.generate(**model_input, max_new_tokens=500, repetition_penalty=1.15).to(device)
            # generated_ids = peft_model.generate(**model_input, max_new_tokens=500, repetition_penalty=1.15)
            #attention_mask=model_input['attention_mask']
            model_inputs = model_input["input_ids"].to(device)
            # print("Type of model inputs: ", type(model_inputs))
            # print("Shape of model inputs: ", model_inputs.shape)
            #`temperature` (=0.0) has to be a strictly positive float, otherwise next token scores will be invalid. 
            # For greedy decoding strategies, set `do_sample=False`
            # Either set `do_sample=True` or unset `top_p
            # Set temperature only when do_sample=True, temperature=0.0 is default when do_sample=False
            generated_ids = peft_model.generate(model_inputs, max_new_tokens=500, repetition_penalty=1.15,
                do_sample=False)
            # generated_ids = generated_ids.to(device)
            # print("Generated IDs: ", generated_ids)
            # Decode the generated output
            decoded_output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # print("Decoded Output: ", decoded_output)
            #extract response from the decoded output. The response is the text appearing after the eos_token
            response = decoded_output.split(eos_token)[1]
            #trim leading and trailing whitespaces
            response = response.strip()
            print("Response: ", response)
            print("\n")

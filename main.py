
import argparse

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders import AsyncChromiumLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype="bfloat16",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
    )


    if args.load_from_scratch:
        # Loading sources
        print("'--load-from-scratch' passed, scraping sources.")
        import nest_asyncio

        nest_asyncio.apply()

        with open("sources.txt") as fp:
            articles = fp.readlines()
            loader = AsyncChromiumLoader(articles)
            docs = loader.load()

        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        text_splitter = CharacterTextSplitter(chunk_size=args.chunk_size, 
                                            chunk_overlap=0)
        chunked_documents = text_splitter.split_documents(docs_transformed)


        db = FAISS.from_documents(chunked_documents, 
                                HuggingFaceEmbeddings(model_name='NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers'))
        
        print(f"Saving vectors to '{args.vectors_db}'.")
        db.save_local(folder_path=args.vectors_db)

    else:
        db = FAISS.load_local(args.vectors_db,  HuggingFaceEmbeddings(model_name='NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers'))

    retriever = db.as_retriever()



    # Create prompt template
    prompt_template = """
    ### [INST] Je bent een expert in Oost-Vlaamse stad Oudenaarde. Antwoord enkel in het Nederlands. Gebruik de volgende context voor vragen te beantwoorden:

    {context}
    
    [/INST]
    {chat_history}

    [INST]
    {question} 
    [/INST]"""

    # Create prompt from prompt template 
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1000)
    llm = HuggingFacePipeline(pipeline=pipe)

    def format_docs(docs):
        return "\n\n- ".join([d.page_content for d in docs])

    def format_history(chat_history):
        print(chat_history)
        if len(chat_history) > 0:
            return "[/INST]" + "\n".join(f"{human}\n[INST]{ai}[/INST]" for human, ai in chat_history) + "\n[INST]\n"


    prompt = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
        template=prompt_template,
        )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix="\n[INST]\n", ai_prefix="[/INST]\n")


    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)

    rag_chain = ( 
    {"context": retriever | format_docs, "question":  RunnablePassthrough()}
        | llm_chain
    )

    with gr.Blocks(css=CSS) as demo:

        def respond(message, chat_history):
            yield rag_chain.invoke(message)['text']

        chatbot = gr.ChatInterface(respond, title=args.title)

        demo.launch(share=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Dutch RAG Toolkit',
                    description='A Dutch retrieval-augmented chat toolkit',
                    epilog='Developed by Pieter Delobelle (KU Leuven)')
    
    parser.add_argument('--load-from-scratch', action='store_true')
    parser.add_argument('--model-name', default='mistralai/Mistral-7B-Instruct-v0.1')
    parser.add_argument('--vectors-db', default='vectors/')
    parser.add_argument('--chunk-size', default=1024)
    parser.add_argument('--title', default="OudenaardeGPT")
    
    args = parser.parse_args()
    
    main(args)

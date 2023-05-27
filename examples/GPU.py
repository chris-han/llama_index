from llama_index import load_index_from_storage
from llama_index.llm_predictor import HuggingFaceLLMPredictor
import torch
from llama_index.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms import GPT4All
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index import (
    GPTVectorStoreIndex,
    LangchainEmbedding,
    LLMPredictor,
    ServiceContext,
    StorageContext,
    download_loader,
    PromptHelper
)

# from PyMuPDF import PyMuPDFReader
PyMuPDFReader = download_loader("PyMuPDFReader")

# documents = PyMuPDFReader().load(
documents = PyMuPDFReader().load(
    file_path='./content/Efficient Methods for NLP.pdf', metadata=True)
# documents = PyMuPDFReader().load(file_path='/content/What_We_Know_GenZ.pdf', metadata=True)
# ensure document texts are not bytes objects
for doc in documents:
    doc.text = doc.text.decode()

# print a document to test. Each document is a single page from the pdf, with appropriate metadata
documents[10]

# setup prompts - specific to Camel

# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
query_wrapper_prompt = SimpleInputPrompt(
    "A continuación hay una instrucción que describe una tarea. "
    "Escribe una respuesta que complete adecuadamente la solicitud.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)


# NOTE: the first run of this will download/cache the weights, ~20GB
hf_predictor = HuggingFaceLLMPredictor(
    max_input_size=2048,
    max_new_tokens=256,
    temperature=0.25,
    do_sample=False,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Writer/camel-5b-hf",
    model_name="Writer/camel-5b-hf",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    model_kwargs={"torch_dtype": torch.bfloat16, "cache_dir": "./cache"},
)

embed_model = LangchainEmbedding(HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"))

service_context = ServiceContext.from_defaults(
    chunk_size_limit=512, llm_predictor=hf_predictor, embed_model=embed_model)

# create the index
index = GPTVectorStoreIndex.from_documents(
    documents, service_context=service_context)
index.storage_context.persist(persist_dir="./storage")


storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(
    storage_context, service_context=service_context)

query_engine = index.as_query_engine(
    streaming=True, similarity_top_k=3, service_context=service_context)
response_stream = query_engine.query(
    "What are the stages of efficient NLP method?")
response_stream.print_response_stream()


print(response_stream.source_nodes)

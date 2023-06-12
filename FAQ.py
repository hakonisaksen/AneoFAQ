from gtp_index import SimpleDirectoryReader, GTPListIndex, GTPSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os

os.environ["OPEN_API_KEY"] = "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"

def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

#define LLM 
llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_tokens=tokens))

#load data
docs = SimpleDirectoryReader(path).load_data()

#create vector index
vectorIndex = GTPSimpleVectorIndex(documents=docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
vectorIndex.save_to_disk('vectorIndex.json')
return vectorIndex

vectorIndex = createVectorIndex('knowledge')
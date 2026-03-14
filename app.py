from flask import Flask, render_template, request, jsonify
from src.helper import download_hugging_face_embeddings
import pinecone
from dotenv import load_dotenv
from src.prompt import *
import os

from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name="medchat"
index = pc.Index(index_name)

docsearch = PineconeVectorStore(
    index=index,
    embedding=embeddings
)

PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

rag_chain = (
    {"context": docsearch.as_retriever(), "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=rag_chain.invoke(input)
    print("Response : ", result ) #result["result"])
    return result #str(result["result"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
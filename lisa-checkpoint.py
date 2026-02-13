from openai import AzureOpenAI
import os
import pandas as pd
import ast
import tiktoken
from langchain_openai import AzureOpenAIEmbeddings   
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_chroma import Chroma
from langchain.schema.document import Document
import streamlit as st

os.environ["AZURE_OPENAI_KEY"] = "b6b4fa7db9864e2f87f13334d7676f42"
os.environ["AZURE_OPENAI_API_KEY"] = "b6b4fa7db9864e2f87f13334d7676f42"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://novaimsplayground.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
model="gpt4o"
client = AzureOpenAI()
today=str(pd.to_datetime('today'))
tokenizer="cl100k_base"

@st.cache_resource
def setup():
    embedding_function=AzureOpenAIEmbeddings(deployment="TextEmbeddings",
    #request_timeout=60,
    #max_retries=50,
    #chunk_size=1,
    model="text-embedding-ada-002")

    db_openai_altered_2 = Chroma(persist_directory="./chroma_db_openai_altered_2_cos", embedding_function=embedding_function)

    df_altered_2=pd.read_csv('knowledge_base_3.csv')
    df_altered_2.drop(columns="Unnamed: 0",inplace=True)

    documents=[]
    for i in range(df_altered_2.shape[0]):
        documents.append(Document(page_content=df_altered_2["page_content"].iloc[i],metadata=ast.literal_eval(df_altered_2["metadata"].iloc[i])))

    bm25_retriever = BM25Retriever.from_documents(documents)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, db_openai_altered_2.as_retriever(search_kwargs={'k':10})], weights=[0.5, 0.5])
    bm25_retriever.k = 10

    return ensemble_retriever

def num_tokens(text: str, model: str = tokenizer) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.get_encoding(model)
    return len(encoding.encode(text))

def query_message(
    query: str,
    answer_history: str,
    token_budget: int
) -> str:
    """Return a message for GPT, with relevant source texts pulled from a dataframe."""
    strings=ensemble_retriever.invoke(query)[:(4)]
    answer_history = "# Previous Answers" +answer_history
    information="\n\n# Context"
    query = f"\n\n# Current Query: {query}"
    for string in strings:
        string=string.page_content
        while (
            num_tokens(answer_history + information + string + query)
            > token_budget
        ):
            answer_history="# Previous Answers"+answer_history[answer_history.find('Query:',answer_history.find('Query:')+1):]
        else:
            information+="\n\nDocument:\n\n"
            information+=string
    message=answer_history
    message+=information
    message+= query
    return message

def generate_response(prompt):
    message = query_message(prompt, st.session_state.answer_history, 10500-1754)
    messages = [
        {"role": "system", "content": f"You are an AI assistant called Lisa that helps tourists visiting Lisbon using both your general knowledge and the provided context (which may or may not be useful). You give engaging and kind responses. Example of a query and the answer you would give: Query - 'I am planning a trip to Lisbon, Portugal. Can you help me with some information about popular tourist attractions?' Response - 'Of course! Your Lisbon journey is about to be amazing. Do not miss out on Belem Tower, the awe-inspiring Jeronimos Monastery, and the picturesque Alfama district. If you have any questions or need more information, just let me know!'. With each user message, you get access to the previous answers (query given by the user and the answer you have given), documents containing context associated to the query and the actual query of the user. Whenever the question is about pricing (or a price is included in your answer), always be conservative in your answer and recommend checking the official website. Always remember the current date is {today}. Do not make recommendations that occur before this date, unless the user specifically requests historical information!"},
        {"role": "user", "content": message},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.5,
        max_tokens=1500,
        stream=True
    )
    return response

ensemble_retriever = setup()

st.title("Lisa")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "answer_history" not in st.session_state:
    st.session_state.answer_history = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Lisa about Lisbon!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = generate_response(prompt)
        response = st.write_stream(stream)
    
    st.session_state.answer_history += f"\n\nQuery: {prompt}\n\nAnswer: {response}"
    st.session_state.messages.append({"role": "assistant", "content": response})
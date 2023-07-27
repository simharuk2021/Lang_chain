from langchain.document_loaders import YoutubeLoader
import os
from apikey import api_key
import streamlit as st  
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

os.environ["OPENAI_API_KEY"] = api_key


# If the history is in th session state, then clear the history stored in the session state.
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

st.title('Chat with You Tube')
youTube_url = st.text_input('Input your Youtube URL')

if youTube_url:
    with st.spinner('Reading file...'):
        

        loader = YoutubeLoader.from_youtube_url(youTube_url)
        documents =loader.load()

        # splits the document into semantically related chunks with default values of size 1000 and overlap 200
        # this should though be based on use case scenarios
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)


        # can use st.write to view the application on the streamlit webpage 
        # st.write(chunks[0])
        # st.write(chunks[1])

        # measures the relatedness of text strings (e.g man - king = 0.5, man to royalty = -0.1, king - royalty = 0.9)
        embeddings = OpenAIEmbeddings()
        # collects the chunks and their relationships into a database
        vector_store = Chroma.from_documents(chunks, embeddings)

        # set the accuracy to complete truth
        # llm=OpenAI(temperature=0)

        # use the 3.5 ChatGPT model which has more functionality (including maths)
        llm=ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=1)

        # use the vector database and use QA (question and answer)
        retriever = vector_store.as_retriever()
        # The below line of code deals with singular Q and A - there is no stored history with which to base further questions on
        # chain = RetrievalQA.from_chain_type(llm, retriever=retriever )
        crc = ConversationalRetrievalChain.from_llm(llm, retriever)
        st.session_state.crc=crc
        st.success('File uploaded, chunked and embedded successfully')

question = st.text_input('Input your question')

if question:
    if 'crc' in st.session_state:
        crc = st.session_state.crc
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        response = crc.run({'question': question, 'chat_history': st.session_state['history']})
        
        st.session_state['history'].append((question, response))
        st.write(response)
        
        for prompts in st.session_state['history']:
            st.write("Question: " + prompts[0])
            st.write("Answer: " + prompts[1])
import os
from apikey import api_key

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = api_key


st.title('Medium Article Generator')
topic = st.text_input('Input your area of interest')

title_template = PromptTemplate(
    input_variables= ['topic'],
    template='Give me a medium article about {topic}'
)

article_template = PromptTemplate(
    input_variables= ['title'],
    template='Give me a medium article for title {title}'
)

# temperature determines the level of creativity for the article
llm = OpenAI(temperature=0.9)
title_chain=LLMChain(llm=llm, prompt = title_template, verbose =True)

llm2 = ChatOpenAI(model_name = "gpt-3.5-turbo", temperature=0.9)
article_chain=LLMChain(llm=llm2, prompt = article_template, verbose =True)

overall_chain = SimpleSequentialChain(chains = [title_chain, article_chain], verbose=True)


if topic:
    response = overall_chain.run(topic)
    st.write(response)
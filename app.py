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

# Add a temperature input widget with custom labels for min and max values
temperature_options = {
    0.1: "Specific",
    0.9: "Default",
    1.0: "Random"
}
temperature = st.selectbox('Select Temperature', options=list(temperature_options.keys()), format_func=lambda x: temperature_options[x])


title_template = PromptTemplate(
    input_variables=['topic'],
    template='Give me a medium article about {topic}'
)

article_template = PromptTemplate(
    input_variables=['title'],
    template='Give me a medium article for title {title}'
)

# temperature determines the level of creativity for the article
llm = OpenAI(temperature=temperature)  # Use the user-selected temperature
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)  # Use the user-selected temperature
article_chain = LLMChain(llm=llm2, prompt=article_template, verbose=True)

overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)

if topic:
    # Generate the title first
    title_response = title_chain.run(topic)
    title = title_response

    # Generate the article content based on the generated title
    article_response = article_chain.run(title)
    article_content = article_response

    # Combine the title and article content and display the result
    full_article = f"{title}\n\n{article_content}"
    st.write(full_article)

import os
from apikey import api_key

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

# Set the OpenAI API key from the imported API key
os.environ["OPENAI_API_KEY"] = api_key

# Set the title of the Streamlit app
st.title('Medium Article Generator')

# Input field for the user to specify the topic of interest
topic = st.text_input('Input your area of interest')

# Define options for different temperature levels with custom labels
temperature_options = {
    0.1: "Specific",
    0.9: "Default",
    1.0: "Random"
}

# Selectbox widget for choosing the temperature level
temperature = st.selectbox('Select Temperature', options=list(temperature_options.keys()), format_func=lambda x: temperature_options[x])

# Define a template for generating the title based on the topic
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Give me a medium article about {topic}'
)

# Define a template for generating the article content based on the title
article_template = PromptTemplate(
    input_variables=['title'],
    template='Give me a medium article for title {title}'
)

# Create an instance of the OpenAI language model with the chosen temperature
llm = OpenAI(temperature=temperature)
# Create a chain for generating the title using the specified template
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)

# Create an instance of the ChatOpenAI language model with the chosen temperature
llm2 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
# Create a chain for generating the article content using the specified template
article_chain = LLMChain(llm=llm2, prompt=article_template, verbose=True)

# Combine the title and article content generation chains into an overall sequential chain
overall_chain = SimpleSequentialChain(chains=[title_chain, article_chain], verbose=True)

# Check if a topic has been provided by the user
if topic:
    # Generate the title using the title_chain
    title_response = title_chain.run(topic)
    title = title_response

    # Generate the article content based on the generated title using the article_chain
    article_response = article_chain.run(title)
    article_content = article_response

    # Combine the title and article content into a full article and display it
    full_article = f"{title}\n\n{article_content}"
    st.write(full_article)


# Medium Article Generator

This Streamlit app generates Medium articles based on user input. It utilizes the OpenAI API to generate article titles and content.

## Prerequisites

- Python 3.x

## Setup

1. Clone this repository to your local machine.

2. Install the required dependencies using the following command:

```bash

python.exe -m pip install --upgrade pip
pip install streamlit openai wikipedia chromadb tiktoken 
pip install python-dotenv
streamlit run app.py
OPENAI_API_KEY

```
The proceess involves creating an OPENAI_API_KEY and then obfuscating this using a dotenv [https://pypi.org/project/python-dotenv/] file (which is hidden via .gitignore).
The streamlit deployment options are available once the streamlit run app.py command has been run and the project is run at a local level.
This then allows for the deployment to be accessed via the committed github repository and the OPEN_AI_API key added via the advanced Streamlit deployment options.

## Acknowledgments

This app uses the OpenAI GPT-3.5 language model for article generation.

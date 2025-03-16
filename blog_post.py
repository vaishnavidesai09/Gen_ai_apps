import streamlit as st
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
llm=ChatOllama(model="llama3:latest")
outline_prompt= PromptTemplate(
    input_variables=["topic"],
    template="""You are a professional blogger. 
    Create an outline for a blog post on the following topic: {topic} 
    The outline should include: 
    - Introduction 
    - 3 main points with subpoints 
    - Conclusion    
    """
)
introduction_prompt= PromptTemplate(
    input_variables=["outline"],
    template="""You are a professional blogger. 
    Write an engaging introduction paragraph based on the following 
    outline:{outline} 
    The introduction should hook the reader and provide a brief 
    overview of the topic.  
    """
)


first_chain = outline_prompt | llm | StrOutputParser() | (lambda outline: (st.write(outline),outline)[1])
second_chain = introduction_prompt |llm 
final_chain = first_chain | second_chain

st.title("Blog Post generator")

topic = st.text_input("Enter the topic for the blog:")

if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)
import streamlit as st
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
llm=ChatOllama(model="llama3:latest")
emotion = ["Happy", "Sad", "Angry", "Surprised", "Fearful", "Disgusted","Inspiring"]
title_prompt= PromptTemplate(
    input_variables=["topic"],
    template="""You are an experienced speech writer. 
    You need to craft an impactful title for a speech  
    on the following topic: {topic} 
    Answer exactly with one title.   
    """
)
speech_prompt= PromptTemplate(
    input_variables=["title","emotion"],
    template="""You need to write a powerful {emotion} speech of 350 words 
     for the following title: {title}  
     Format the output with 2 keys: 'title', 'speech'  and fill thm with the respective values.
    """
)


first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])
second_chain = speech_prompt |llm | JsonOutputParser()
final_chain = first_chain | (lambda title :{ "title":title,"emotion":emotion }) |second_chain

st.title("Speech generator")

topic = st.text_input("Enter the topic for the speech:")
topic = st.text_input("Enter the emotion for the speech:")
if topic and emotion:
    response = final_chain.invoke({"topic":topic})
    st.write(response)
    st.write(response['title'])
   
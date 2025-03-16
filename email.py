import streamlit as st
from langchain.prompts import PromptTemplate

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
llm=ChatOllama(model="llama3:latest")

prompt1= PromptTemplate(
    input_variables=["product_name","features"],
    template="""You are an experienced marketing specialist.  
    Create a catchy subject line for a marketing  
    email promoting the following product: {product_name}.  
    Highlight these features: {features}.  
    Respond with only the subject line.  
    """
)
prompt2= PromptTemplate(
    input_variables=["subject_line","product_name","target_audience"],
    template="""Write a marketing email of 300 words for the  
    product: {product_name}. Use the subject line: 
     {subject_line}. Tailor the message for the  
     following target audience: {target_audience}. 
      Format the output as a JSON object with three  
      keys: 'subject', 'audience', 'email' and fill  
      them with respective values.
    """
)


first_chain = prompt1 | llm | StrOutputParser() 
second_chain = prompt2 |llm | JsonOutputParser()
final_chain = first_chain | (lambda subject_line :{ "subject_line":subject_line,"product_name":product_name,"target_audience":target_audience }) |second_chain

st.title("Marketing Email Generator")

product_name = st.text_input("Enter the product name:")
features = st.text_input("Enter the product features:")
target_audience = st.text_input("Enter the target audience:")

if product_name and features and target_audience:
    response = final_chain.invoke({"product_name":product_name,"features":features})
    st.write(response)
    
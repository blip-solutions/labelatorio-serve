import openai
from typing import List,Dict
from string import Template
from ..models.responses import (SimilarQuestionAnswer)

def generate_answer(api_key, question:str, answered_examples:List[SimilarQuestionAnswer], context_data:Dict[str,str]=None, additional_instructions:str=None, additional_instruction_on_no_data:str=None):
    openai.api_key = api_key


    if answered_examples:
            prompt_template="""Here are good examples of questions and answers:$answered_examples

    answer this question $additional_instructions:
    $context
    Q:$question
    A:"""
       
    else:
        prompt_template=f"""Try to answer this question $additional_instructions:
    $context
    Q:$question
    A:"""


    answered_examples_interpolated=""
    for example in answered_examples:
        answered_examples_interpolated+=f"""
    Q:{example.text}
    A:{example.answer}
    """
    #we dont want to exceed max_tokens
        if len(answered_examples_interpolated)+len(question)>4000:
            break
    _context_interpolated=""
    if context_data:
        _context_interpolated="Context:"
        for key,val in context_data.items():
            _context_interpolated+=f"{key.replace('_',' ').replace('-',' ')}: {val}\n"
        
    if additional_instructions and answered_examples:
        instructions=f"({additional_instructions})"
    elif not answered_examples:
        instructions=f"({additional_instruction_on_no_data})"
    else:
        instructions=""



    params={
        "answered_examples": answered_examples_interpolated,
        "question": question.strip(),
        "additional_instructions": instructions,
        "context":_context_interpolated
    }
    
    prompt=Template(prompt_template).safe_substitute(**params)

  
    response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.5,
            max_tokens=4097-int(len(prompt)/2),
            top_p=1,
            frequency_penalty=0.1,
            presence_penalty=0.1,
        )
    choices = response["choices"][0]
    res =  choices["text"].lstrip() 
    return res
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template = "Generate short and simple notes from the following \n {text}",
    input_variables = ['text']
)

prompt2 = PromptTemplate(
    template = "Generate 5 short questions and answers from the following \n {text}",
    input_variables = ['text']
)

prompt3 = PromptTemplate(
    template = "Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz ->{quiz}",
    input_variables = ['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    notes = prompt1 | model | parser,
    quiz = prompt2 | model | parser
)

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
Football is one of the most popular sports in the world and is played by two teams of eleven players each. 
The objective of the game is to score goals by getting the ball into the opposing team's net. Players mainly 
use their feet to control, pass, and shoot the ball, although they can also use their head or chest. 
Only the goalkeeper is allowed to use their hands, and only within the penalty area.

A standard football match lasts 90 minutes and is divided into two halves of 45 minutes each. 
If the score is tied in knockout competitions, extra time or penalty shootouts may be used to 
determine the winner.

Football requires teamwork, strategy, and physical fitness. Players take on different roles 
such as defenders, midfielders, and forwards. Defenders focus on stopping the opposing team 
from scoring, midfielders control the flow of the game, and forwards try to score goals.

Major international tournaments such as the FIFA World Cup bring together the best national 
teams from around the world and are watched by billions of fans.
"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()


from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler

chat = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ]
)

# t = PromptTemplate(
#     template="What is the capital of {country}",
#     input_variables=["country"]
# )

# t = ChatPromptTemplate.from_messages("What is the capital of {country}")

# t.format(country = "Korea")

examples = [
{
"country": "France",
"answer": """
Here is what I know:
Capital: Paris
Language: French
Food: Wine and Cheese
Currency: Euro
""",
},
{
"country": "Italy",
"answer": """
I know this:
Capital: Rome
Language: Italian
Food: Pizza and Pasta
Currency: Euro
""",
},
{
"country": "Greece",
"answer": """
I know this:
Capital: Athens
Language: Greek
Food: Souvlaki and Feta Cheese
Currency: Euro
""",
},
]

# examples_template = """
#     Humane : {question}
#     AI: {answer}
# """

examples_prompt = ChatPromptTemplate.from_messages([
    ("human", "What do you know about {country} and translate it into Korean without English result"),
    ("ai", "{answer}")
])

example_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=examples_prompt,
    examples=examples,

)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Geography expert, You give short answers. You always answer into Korean"),
    example_prompt,
    ("human", "What do you know about {country} and translate it into Korean without English result")
])

# prompt.format(country="Korea")


chain = final_prompt | chat
chain.invoke(
    {
        "country" : "US"
    }
)

# chat.predict("What do you know about Korea? and translate it into Korean withour English result")
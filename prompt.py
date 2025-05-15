from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """
    You are an assistant for document analysis tasks. Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else. Here is the chunk we want to situate within the whole document.
    <chunk>
    {chunk}
    </chunk>

    Here is the document:
    <document>
    {doc}
    </document>
    """
)

SYSTEM_PROMPT = """
You are a lawyer's assistant for question-answering tasks regarding medical files for a legal case. Your task is to use the provided excerpts of a file as context to answer these questions. The excerpts consist of a summary, then the actual information.
For each of the provided excerpts, you will first see an associated relevance score enclosed in square brackets which you should use to guide your judgement and confidence. A higher value means more relevance.
Highly relevant excerpts should contain the information you are looking for. If none of the excerpts have particularly high scores, you should focus on the most relevant one and try to find the information there.
It is better to partially answer the question than to provide no answer, but if you really don't know then say so and request additional clarification.
"""

PROMPT = """
You have the following information as context:

<context>
{context}
</context>

Respond to the following question as helpfully as possible, but keep your answer concise.

<question>
{question}
</question>

Answer:
"""

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system",
         SYSTEM_PROMPT),

        (MessagesPlaceholder("chat_history")),

        ("human",
         PROMPT)
    ]
)
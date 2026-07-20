import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

# Improt .env variables
load_dotenv()


def main():
    information = """
    Elon Reeve Musk (/ˈiːlɒn mʌsk/ Pretoria, June 28, 1971) is a businessman, investor, political activist conservative[3][4] and tycoon South african naturalized american.[note 1] He is the founder, CEO and «engineer» in chief of the company SpaceX; angel investor, general director and product architect Tesla, Inc; founder of The Boring Company; and co-founder of Neuralink and OpenAI.[note 2] Additionally, he is the chief technology officer of X Corp[5] Between January and May 2025, he served as administrator de facto del Department of Government Efficiency of the White House under the second presidency of Donald Trump.[6][7]As of June 2026, Musk is the richest person in the world according to Forbes. After the IPO of SpaceX- his net worth exceeded US$1.1 trillion, making him the only and first trillionaire in US dollar terms in history.[8] On June 16, 2026, his assets would reach a record high of $1.4 trillion.[9]

Musk was born and raised in a wealthy family of Pretoria (South Africa). His mother is Canadian and his father a white south african. He studied briefly at the University Pretoria before moving to Canada at age 17. He enrolled in the Queen's University and he moved to the Pennsylvania University two years later, where graduated in Economics and Physics. In 1995 he moved to California to attend the Stanford University, but instead decided to pursue a business career, co-founding the web software company Zip2 with his brother Kimbal. Zip2 it was acquired by Compaq for 307 million dollars in 1999. That same year, Musk co-founded the online bank X.com, which merged with Confinity in 2000 to form PayPal. The company was purchased by eBay in 2002 for one and a half billion dollars.

In 2002, Musk founded SpaceX, an aerospace manufacturer and space transportation services company, of which he is CEO and chief engineer. In 2003, it joined electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc) as president and product architect, becoming its CEO in 2008. In 2006, he helped create SolarCity, now Tesla Energy, a solar energy services company that was later acquired by Tesla and became Tesla Energy. In 2015, he co-founded OpenAI, a nonprofit research company that promotes friendly artificial intelligence. In 2016, he co-founded Neuralink, a neurotechnology company focused on developing brain-computer interfaces, and founded The Boring Company, a tunnel construction company. It also agreed to purchase the important American social network Twitter in 2022 by 44 000 million dollars. Musk has also proposed the hyperloop. In November 2021, Tesla's CEO was the first person in history to accumulate a fortune of three hundred billion dollars.[10]

He has been criticized for making unscientific and controversial statements. In 2018, he was sued by the Securities and Exchange Commission from the United States (SEC) for falsely tweeting that it had obtained financing for a private acquisition of Tesla. He reached a settlement with the SEC, but did not admit guilt, temporarily resigning from his presidency and accepting limitations on the use of his personal Twitter. In 2019, he won a defamation lawsuit brought against him by a British speleologist who advised on the tham Luang Cave Rescue. Musk has also been criticized for spreading the word misinformation about the COVID-19 pandemicand conspiracy theories; and for his controversial opinions on issues such as artificial intelligence, cryptocurrencies and public transportation.

    """
    summary_template = """
    Given the information {information} about a person I want you to create:
    1. A short summary
    2. Two interesting fact about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    #llm = ChatAnthropic(temperature=0, model="claude-haiku-4-5-20251001") # type: ignore
    llm = ChatOllama(temperature=0, model="llama3.2")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()

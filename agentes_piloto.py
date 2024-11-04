from crewai import Crew, Task, Agent, Process
from langchain_openai import ChatOpenAI
import os

# Configuração do LLM
ollama = ChatOpenAI(
    model_name="ollama/gemma2:27b",
    base_url="http://localhost:11434", 
    api_key="N/A", 
)


gereral_estrategista = Agent(
    role="General Estrategista.",
    goal="General tem um exercito com Jatos de reconhecimento, jatos de ataque, e infrantaria a sua dispocição para tomar uma região.",
    llm=ollama,
    memory=True,
    allow_delegation=True,
    backstory="Você é um general especializado em estratégia de combates aereos e terrestre. e precisa tomar as melhores decisões para tomar uma região. toda saida em português pt-br"
)

piloto1 = Agent(
    role="Piloto de Jato reconhecimento.",
    goal="Voar sobre território inimigo coletando o máximo de informações do terreno inimigo que ajude sua tropa.",
    llm=ollama,
    memory=True,
    backstory="Você é um experiente piloto de caça que tem como melhor qualidade mapear terreno inimigo coletando informações. toda saida em português pt-br"
)

infrantaria = Agent(
    role="Tropas Infantaria.",
    goal="Vai cumprir a missão que será delegada pelo general estrategista, vai machar e lutar na batalha.",
    llm=ollama,
    memory=True,
    backstory="É um grupo de soldados treinadas e armados que vai cumprir a missão, delegada pelo general estrategista. toda saida em português pt-br"
)

jatos_de_combate = Agent(
    role="Jatos de combate.",
    goal="Vai cumprir a missão atacando as areas determinadas pelo general.",
    llm=ollama,
    memory=True,
    backstory="É um grupo de jatos de ataque que estão a disposição para cumprir a missão delegada pelo general estrategista.  toda saida em português pt-br"
)





estrategia_general = Task(
    description="Após receber a analise de voo irá criar melhor estratégia de combate.",
    expected_output="Vai decidir qual estrégia será tomada se será ataque aereo ou ataque terrestre baseado na analise de voo",
    agent=gereral_estrategista
)

missao1 = Task(
    description="Vai sobrevoar territorio inimigo coletando informações para ajudar sua tropa de infrantária a dominar a região.",
    expected_output="Relatório das informações que foram coletadas em voo.",
    agent=piloto1
)

batalha_infrantaria = Task(
    description="Após receber a ordem de missão do general vai cumprir com eficiência.",
    expected_output="Vai executar e missão delegada pelo general e vai fazer um relatório do resultado da missão",
    agent=infrantaria
)

ataque_com_jatos = Task(
    description="Após receber a ordem de missão do general vai cumprir com eficiência.",
    expected_output="Vai executar e missão delegada pelo general e vai fazer um relatório do resultado da missão",
    agent=jatos_de_combate
)



crew = Crew(
    agents=[gereral_estrategista, piloto1, infrantaria, jatos_de_combate],
    tasks=[estrategia_general, missao1, batalha_infrantaria, ataque_com_jatos],
    hierarchical=True,
    verbose=True
)

result = crew.kickoff()

print(result)

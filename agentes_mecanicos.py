from crewai import Crew, Task, Agent, Process
from langchain_openai import ChatOpenAI
import os

# Configuração do LLM
ollama = ChatOpenAI(
    model_name="ollama/llama3.2",
    base_url="http://localhost:11434", 
    api_key="sua-api-key", 
)

relato_defeito = """
Moto: XRE-300.
Ano: 2023.
KM-Rodado: 30.000.
Problema: barulho estranho no motor somente quando reduz a velocidade.
Tempo de defeito: O problema começou tem 2 dias.
Última manutenção: a última manutenção foi a mais de 6 meses.
"""

mecanico_1 = Agent(
    role="Mecânico especialista em motores de moto",
    goal="""Com base no relato fornecido sobre o defeito, você vai usar sua experiência para 
    identificar o possível defeito no motor da moto. Seja preciso e evite diagnósticos 
    que possam gerar custos desnecessários para o cliente.""",
    verbose=True,
    llm=ollama,
    backstory="""Você tem 30 anos de experiência em consertar motores de moto. 
    Seu objetivo é ouvir atentamente o relato e fazer um diagnóstico preciso 
    com base em sua experiência."""
)

mecanico_2 = Agent(
    role="Mecânico especialista em motores de moto",
    goal="""Com base no relato fornecido sobre o defeito, você vai usar sua experiência para 
    identificar o possível defeito no motor da moto. Seja preciso e evite diagnósticos 
    que possam gerar custos desnecessários para o cliente.""",
    verbose=True,
    llm=ollama,
    backstory="""Você tem 25 anos de experiência em consertar motores de moto. 
    Seu objetivo é ouvir atentamente o relato e fazer um diagnóstico preciso 
    com base em sua experiência."""
)

analise_moto_1 = Task(
    description=f"""Com base no seguinte relato do cliente, identifique os possíveis defeitos no motor:
    '{relato_defeito}'""",
    expected_output="""Após ouvir o relato do cliente, o mecânico vai fazer um relatório identificando 
    as possíveis peças com defeito. O diagnóstico precisa ser preciso para evitar custos desnecessários.""",
    agent=mecanico_1,
    output_file="src/result/relatorio_mecanico_11.txt"
)

analise_moto_2 = Task(
    description=f"""Com base no seguinte relato do cliente, identifique os possíveis defeitos no motor:
    '{relato_defeito}'""",
    expected_output="""Após ouvir o relato do cliente, o mecânico vai fazer um relatório identificando 
    as possíveis peças com defeito. O diagnóstico precisa ser preciso para evitar custos desnecessários.""",
    agent=mecanico_2,
    output_file="src/result/relatorio_mecanico_22.txt"
)



crew = Crew(
    agents=[mecanico_1, mecanico_2],
    tasks=[analise_moto_1, analise_moto_2],
    process=Process.sequential,
    verbose=True,
)

Process.sequential
result = crew.kickoff()

print(result)

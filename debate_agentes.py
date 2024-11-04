from crewai import Crew, Task, Agent, Process
from langchain_openai import ChatOpenAI
import os

# Configuração do LLM
ollama = ChatOpenAI(
    model_name="ollama/gemma2:27b",
    base_url="http://localhost:11434", 
    api_key="na", 
)

# Função para criar a equipe de debate
def criar_crew_debate(qtd_agentes, qtd_rodadas, problema):
    # Agente Gerente: Responsável pela coordenação e consolidação da solução final
    gerente = Agent(
        role="Gerente do Debate",
        goal="Orquestrar o debate e consolidar a solução final para o problema.",
        memory=True,
        llm=ollama,
        backstory="Responsável por consolidar as soluções dos agentes e gerenciar o debate."
    )

    # Lista de agentes debatedores
    debatedores = []
    for i in range(qtd_agentes):
        debatedor = Agent(
            role=f"Agente{i+1}",
            goal="Propor uma solução para o problema e refiná-la com base nas contribuições dos demais agentes.",
            memory=True,
            llm=ollama,
            backstory=f"Especialista em analisar o problema e aprimorar a solução durante o debate."
        )

        debatedores.append(debatedor)


    editor = Agent(
        role=f"Editor texto para Linkedin",
        goal="Vai pegar o resumo final feito pelo gerente e vai criar uma postagem para Linkedin."
        "O post pode ter um comunicação não formal apontando problemas e soluções baseado informado no relatório do gerente",
        memory=True,
        llm=ollama,
        backstory=f"Especialista em criar assuntos que chamam atenção e gerar engajamento no Linkedin."
    )

    # Tarefa de Debate
    debate_task = Task(
        description=(
            f"Debate sobre o problema: {problema}. Cada agente deve propor e atualizar "
            f"sua solução em {qtd_rodadas} rodadas, considerando as sugestões dos demais agentes."
        ),
        expected_output="Solução final consolidada do problema após o debate.",
        agent=gerente,
        tools=[],
        parameters={
            "qtd_agentes": qtd_agentes,
            "qtd_rodadas": qtd_rodadas,
            "problema": problema
        }
    )


    potexto_paraPostagem = Task(
        description = f"Após o gerente consolidar todo o debate o editor vai receber os principais pontos de discursão e vai criar um post para Linkedin.",
        expected_output="Um texto bem elaborado e informal para uma postagem no Linkedin não use sem os bulets points.",
        agent=editor,
        output_file="post3.txt"
    )

    crew = Crew(
        agents=[gerente] + debatedores, # + [editor],
        tasks=[debate_task], # , potexto_paraPostagem],
        process=Process.sequential
    )


    resultado = crew.kickoff(inputs={"problema": problema})
    return resultado

# Exemplo de uso
qtd_agentes = 4
qtd_rodadas = 3  
problema = "Como podemos resolver, problema sociais causados com o uso de IA, substituindo pessoas por agentes autonomos?"

resultado_final = criar_crew_debate(qtd_agentes, qtd_rodadas, problema)
print(resultado_final)

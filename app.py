"""
Agente IA do Professor - Protótipo Fase 1
Aplicativo web com CrewAI + Llama via Groq
Inclui agente revisor para evitar alucinações.
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

# Configuração da chave de API do Groq
# A chave é lida de variável de ambiente para segurança no deploy
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)

# Modelo: GPT OSS 120B (substituto recomendado do Llama 3.3 70B, mais rápido e potente)
MODEL = "openai/gpt-oss-120b"

# ============================================================
# DEFINIÇÃO DOS PROMPTS DOS AGENTES
# ============================================================

PROMPT_ASSISTENTE = """Você é um Assistente Educacional especializado em ajudar professores e alunos do ensino médio técnico em escolas públicas estaduais de São Paulo, Brasil.

REGRAS FUNDAMENTAIS:
1. Sempre responda em português brasileiro.
2. Seja claro, didático e use linguagem acessível.
3. NUNCA invente informações, dados, datas, nomes ou referências bibliográficas.
4. Se não souber a resposta com certeza absoluta, diga EXPLICITAMENTE: "Não tenho certeza sobre isso" ou "Recomendo verificar esta informação em uma fonte oficial".
5. Quando citar dados específicos (datas, estatísticas, nomes), indique que o usuário deve confirmar em fontes oficiais se a precisão for crítica.
6. Prefira dizer "não sei" a inventar qualquer coisa.

Responda a seguinte pergunta/solicitação do usuário:
"""

PROMPT_REVISOR = """Você é um Revisor de Qualidade e Veracidade. Sua ÚNICA função é revisar a resposta abaixo e garantir que ela NÃO contém:

1. Informações inventadas (alucinações) - dados, datas, nomes, estatísticas que parecem fabricados.
2. Afirmações apresentadas como fatos absolutos quando deveriam ter ressalvas.
3. Referências bibliográficas ou fontes que possam ser fictícias.
4. Linguagem inadequada para o público (professores e alunos do ensino médio).

INSTRUÇÕES:
- Se a resposta estiver CORRETA e SEGURA, entregue-a EXATAMENTE como está, sem alterações.
- Se encontrar algo SUSPEITO, adicione um aviso entre colchetes no final: [AVISO: ...]
- Se a resposta contiver dados muito específicos que você não pode confirmar, adicione: [NOTA: Recomenda-se verificar os dados específicos mencionados em fontes oficiais.]
- NÃO adicione comentários sobre seu processo de revisão.
- NÃO mude o estilo ou tom da resposta original se ela estiver correta.
- Responda APENAS com a resposta final revisada, em português brasileiro.

RESPOSTA A SER REVISADA:
"""


def processar_pergunta(pergunta_usuario: str) -> str:
    """
    Processa uma pergunta do usuário usando o sistema de dois agentes:
    1. O Assistente responde a pergunta.
    2. O Revisor confere a resposta e a aprova ou corrige.
    """

    # Passo 1: O Assistente responde
    resposta_assistente = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_ASSISTENTE},
            {"role": "user", "content": pergunta_usuario}
        ],
        temperature=0.3,
        max_tokens=2048,
    )
    texto_assistente = resposta_assistente.choices[0].message.content

    # Passo 2: O Revisor confere
    resposta_revisor = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": PROMPT_REVISOR},
            {"role": "user", "content": texto_assistente}
        ],
        temperature=0.1,  # Temperatura muito baixa para o revisor ser conservador
        max_tokens=2048,
    )
    texto_final = resposta_revisor.choices[0].message.content

    return texto_final


# ============================================================
# APLICATIVO WEB (Flask)
# ============================================================

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    """Serve a página principal do chat."""
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint da API que recebe a pergunta e retorna a resposta do agente."""
    dados = request.get_json()
    pergunta = dados.get("pergunta", "").strip()

    if not pergunta:
        return jsonify({"erro": "Pergunta vazia."}), 400

    try:
        resposta = processar_pergunta(pergunta)
        return jsonify({"resposta": resposta})
    except Exception as e:
        return jsonify({"erro": f"Ocorreu um erro: {str(e)}"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  AGENTE IA DO PROFESSOR - Protótipo Fase 1")
    print("  Acesse: http://localhost:5000")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

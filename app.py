"""
Agente IA do Professor - Versão 2.0
- Autenticação com login/registro
- Histórico de conversas salvas
- Memória de contexto (lembra o que foi dito)
- Tom natural e humano
- Revisor de qualidade
"""

import os
import json
import hashlib
import secrets
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, session
from groq import Groq

# ============================================================
# CONFIGURAÇÃO
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Banco de dados simples com JSON (para o plano gratuito do Render)
# Em produção futura, migrar para PostgreSQL
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/agente_data")
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


# ============================================================
# FUNÇÕES DE BANCO DE DADOS (JSON)
# ============================================================

def load_users():
    """Carrega os usuários do arquivo JSON."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_users(users):
    """Salva os usuários no arquivo JSON."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def hash_password(password):
    """Gera hash seguro da senha."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(stored_hash, password):
    """Verifica se a senha confere com o hash armazenado."""
    salt, hashed = stored_hash.split(":")
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed


def get_user_conversations_file(username):
    """Retorna o caminho do arquivo de conversas do usuário."""
    safe_name = hashlib.md5(username.encode()).hexdigest()
    return os.path.join(CONVERSATIONS_DIR, f"{safe_name}.json")


def load_conversations(username):
    """Carrega as conversas de um usuário."""
    filepath = get_user_conversations_file(username)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_conversations(username, conversations):
    """Salva as conversas de um usuário."""
    filepath = get_user_conversations_file(username)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)


# ============================================================
# AUTENTICAÇÃO
# ============================================================

def login_required(f):
    """Decorator que exige login para acessar a rota."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "username" not in session:
            return jsonify({"erro": "Não autenticado. Faça login."}), 401
        return f(*args, **kwargs)
    return decorated_function


# ============================================================
# PROMPTS DOS AGENTES (v2.0 - Tom Natural)
# ============================================================

PROMPT_ASSISTENTE = """Você é o Agente IA do Professor — um assistente inteligente, amigável e direto.

PERSONALIDADE:
- Fale de forma natural, como um colega inteligente e bem-informado falaria.
- Seja direto e objetivo, mas sem ser frio ou robótico.
- Use linguagem clara e acessível, sem jargões desnecessários.
- Pode usar expressões naturais como "Olha", "Bom", "Na verdade", etc.
- Seja didático quando a pergunta pedir explicação.
- Tenha personalidade: seja simpático sem ser bajulador.

REGRAS INEGOCIÁVEIS:
1. NUNCA invente informações, dados, datas, nomes ou referências.
2. Se não souber algo com certeza, diga naturalmente: "Não tenho certeza sobre isso" ou "Seria bom confirmar esse detalhe".
3. Prefira dizer "não sei" a inventar qualquer coisa.
4. Responda sempre em português brasileiro.

CONTEXTO: Você atende principalmente professores e alunos do ensino médio técnico de São Paulo, mas pode ajudar com qualquer assunto.
"""

PROMPT_REVISOR = """Você é um revisor silencioso. Sua função é verificar a resposta abaixo e:

1. Se estiver CORRETA e NATURAL: entregue EXATAMENTE como está. Não mude nada.
2. Se contiver informação claramente ERRADA ou INVENTADA: corrija discretamente ou adicione "[Atenção: este dado precisa ser verificado]" ao lado do trecho suspeito.
3. Se contiver dados muito específicos (datas exatas, números, citações): deixe passar se parecerem plausíveis, mas adicione ao final apenas: "*Para dados específicos, vale confirmar em fontes oficiais.*" — MAS SOMENTE se realmente houver dados numéricos ou datas específicas na resposta.

IMPORTANTE:
- NÃO adicione notas ou avisos em respostas sobre fatos amplamente conhecidos (capitais, autores famosos, conceitos básicos de ciência, etc.)
- NÃO mude o tom ou estilo da resposta
- NÃO seja mais conservador do que necessário
- NÃO adicione comentários sobre seu processo de revisão
- Responda APENAS com a resposta final

RESPOSTA A SER REVISADA:
"""


# ============================================================
# PROCESSAMENTO DE PERGUNTAS COM MEMÓRIA
# ============================================================

def processar_pergunta(pergunta_usuario: str, historico: list) -> str:
    """
    Processa uma pergunta com memória de contexto.
    O histórico das últimas mensagens é enviado junto para manter o contexto.
    """

    # Montar mensagens com contexto (últimas 10 mensagens para não estourar tokens)
    mensagens = [{"role": "system", "content": PROMPT_ASSISTENTE}]

    # Adicionar histórico recente (últimas 10 trocas)
    historico_recente = historico[-20:]  # 20 mensagens = 10 trocas (user + assistant)
    for msg in historico_recente:
        mensagens.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # Adicionar a pergunta atual
    mensagens.append({"role": "user", "content": pergunta_usuario})

    # Passo 1: O Assistente responde com contexto
    resposta_assistente = client.chat.completions.create(
        model=MODEL,
        messages=mensagens,
        temperature=0.4,
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
        temperature=0.1,
        max_tokens=2048,
    )
    texto_final = resposta_revisor.choices[0].message.content

    return texto_final


# ============================================================
# APLICATIVO WEB (Flask)
# ============================================================

app = Flask(__name__, static_folder="static")
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # Mudar para True com HTTPS


# --- Rotas de páginas ---

@app.route("/")
def index():
    """Serve a página principal."""
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve arquivos estáticos."""
    return send_from_directory("static", filename)


# --- Rotas de autenticação ---

@app.route("/api/register", methods=["POST"])
def register():
    """Registra um novo usuário."""
    dados = request.get_json()
    username = dados.get("username", "").strip().lower()
    password = dados.get("password", "").strip()
    nome = dados.get("nome", "").strip()

    if not username or not password:
        return jsonify({"erro": "Usuário e senha são obrigatórios."}), 400

    if len(username) < 3:
        return jsonify({"erro": "Usuário deve ter pelo menos 3 caracteres."}), 400

    if len(password) < 4:
        return jsonify({"erro": "Senha deve ter pelo menos 4 caracteres."}), 400

    users = load_users()

    if username in users:
        return jsonify({"erro": "Este usuário já existe. Escolha outro."}), 409

    users[username] = {
        "password_hash": hash_password(password),
        "nome": nome or username,
        "criado_em": datetime.now().isoformat(),
    }
    save_users(users)

    session["username"] = username
    session["nome"] = nome or username

    return jsonify({"sucesso": True, "nome": nome or username})


@app.route("/api/login", methods=["POST"])
def login():
    """Faz login do usuário."""
    dados = request.get_json()
    username = dados.get("username", "").strip().lower()
    password = dados.get("password", "").strip()

    if not username or not password:
        return jsonify({"erro": "Usuário e senha são obrigatórios."}), 400

    users = load_users()

    if username not in users:
        return jsonify({"erro": "Usuário ou senha incorretos."}), 401

    if not verify_password(users[username]["password_hash"], password):
        return jsonify({"erro": "Usuário ou senha incorretos."}), 401

    session["username"] = username
    session["nome"] = users[username]["nome"]

    return jsonify({"sucesso": True, "nome": users[username]["nome"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    """Faz logout do usuário."""
    session.clear()
    return jsonify({"sucesso": True})


@app.route("/api/me", methods=["GET"])
def me():
    """Retorna dados do usuário logado."""
    if "username" not in session:
        return jsonify({"logado": False}), 200
    return jsonify({
        "logado": True,
        "username": session["username"],
        "nome": session["nome"]
    })


# --- Rotas de conversas ---

@app.route("/api/conversations", methods=["GET"])
@login_required
def list_conversations():
    """Lista todas as conversas do usuário."""
    conversations = load_conversations(session["username"])
    # Retorna resumo (sem mensagens completas)
    resumo = []
    for conv in conversations:
        resumo.append({
            "id": conv["id"],
            "titulo": conv["titulo"],
            "criada_em": conv["criada_em"],
            "atualizada_em": conv.get("atualizada_em", conv["criada_em"]),
            "mensagens_count": len(conv["mensagens"])
        })
    # Ordenar por mais recente
    resumo.sort(key=lambda x: x["atualizada_em"], reverse=True)
    return jsonify(resumo)


@app.route("/api/conversations", methods=["POST"])
@login_required
def create_conversation():
    """Cria uma nova conversa."""
    conversations = load_conversations(session["username"])
    nova_id = secrets.token_hex(8)
    nova_conversa = {
        "id": nova_id,
        "titulo": "Nova conversa",
        "criada_em": datetime.now().isoformat(),
        "atualizada_em": datetime.now().isoformat(),
        "mensagens": []
    }
    conversations.append(nova_conversa)
    save_conversations(session["username"], conversations)
    return jsonify(nova_conversa)


@app.route("/api/conversations/<conv_id>", methods=["GET"])
@login_required
def get_conversation(conv_id):
    """Retorna uma conversa completa com mensagens."""
    conversations = load_conversations(session["username"])
    for conv in conversations:
        if conv["id"] == conv_id:
            return jsonify(conv)
    return jsonify({"erro": "Conversa não encontrada."}), 404


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@login_required
def delete_conversation(conv_id):
    """Exclui uma conversa."""
    conversations = load_conversations(session["username"])
    conversations = [c for c in conversations if c["id"] != conv_id]
    save_conversations(session["username"], conversations)
    return jsonify({"sucesso": True})


# --- Rota principal do chat ---

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    """Endpoint do chat com memória de contexto."""
    dados = request.get_json()
    pergunta = dados.get("pergunta", "").strip()
    conv_id = dados.get("conversa_id", "").strip()

    if not pergunta:
        return jsonify({"erro": "Pergunta vazia."}), 400

    if not conv_id:
        return jsonify({"erro": "ID da conversa não informado."}), 400

    # Carregar conversa
    conversations = load_conversations(session["username"])
    conversa = None
    for conv in conversations:
        if conv["id"] == conv_id:
            conversa = conv
            break

    if not conversa:
        return jsonify({"erro": "Conversa não encontrada."}), 404

    # Montar histórico para contexto
    historico = []
    for msg in conversa["mensagens"]:
        historico.append({
            "role": "user" if msg["tipo"] == "user" else "assistant",
            "content": msg["texto"]
        })

    try:
        resposta = processar_pergunta(pergunta, historico)

        # Salvar mensagens na conversa
        agora = datetime.now().isoformat()
        conversa["mensagens"].append({
            "tipo": "user",
            "texto": pergunta,
            "timestamp": agora
        })
        conversa["mensagens"].append({
            "tipo": "agent",
            "texto": resposta,
            "timestamp": agora
        })
        conversa["atualizada_em"] = agora

        # Atualizar título se for a primeira mensagem
        if len(conversa["mensagens"]) == 2:
            # Usar as primeiras palavras da pergunta como título
            titulo = pergunta[:50]
            if len(pergunta) > 50:
                titulo += "..."
            conversa["titulo"] = titulo

        save_conversations(session["username"], conversations)

        return jsonify({"resposta": resposta})

    except Exception as e:
        return jsonify({"erro": f"Ocorreu um erro: {str(e)}"}), 500


# ============================================================
# INICIALIZAÇÃO
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  AGENTE IA DO PROFESSOR - Versão 2.0")
    print("  Acesse: http://localhost:5000")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

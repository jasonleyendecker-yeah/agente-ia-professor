"""
Agente IA do Professor - Versão 2.1
- Autenticação com login/registro
- Histórico de conversas salvas (PostgreSQL - Neon)
- Memória de contexto (lembra o que foi dito)
- Tom natural e humano
- Revisor de qualidade
- Dados persistentes (não perde ao reiniciar)
"""

import os
import json
import hashlib
import secrets
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, session
from groq import Groq
import psycopg2
from psycopg2.extras import RealDictCursor

# ============================================================
# CONFIGURAÇÃO
# ============================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=GROQ_API_KEY)
MODEL = "openai/gpt-oss-120b"
SECRET_KEY = os.environ.get("SECRET_KEY", secrets.token_hex(32))
DATABASE_URL = os.environ.get("DATABASE_URL", "")


# ============================================================
# BANCO DE DADOS (PostgreSQL via Neon)
# ============================================================

def get_db():
    """Retorna uma conexão com o banco de dados."""
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn


def init_db():
    """Cria as tabelas se não existirem."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(200) NOT NULL,
            nome VARCHAR(200) NOT NULL,
            criado_em TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id VARCHAR(20) PRIMARY KEY,
            username VARCHAR(100) NOT NULL REFERENCES users(username),
            titulo VARCHAR(200) NOT NULL DEFAULT 'Nova conversa',
            criada_em TIMESTAMP DEFAULT NOW(),
            atualizada_em TIMESTAMP DEFAULT NOW()
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversa_id VARCHAR(20) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            tipo VARCHAR(10) NOT NULL,
            texto TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


# ============================================================
# FUNÇÕES DE BANCO DE DADOS
# ============================================================

def hash_password(password):
    """Gera hash seguro da senha."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(stored_hash, password):
    """Verifica se a senha confere com o hash armazenado."""
    salt, hashed = stored_hash.split(":")
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed


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

    # Adicionar histórico recente (últimas 20 mensagens = 10 trocas user + assistant)
    historico_recente = historico[-20:]
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

app = Flask(__name__, static_folder=".")
app.secret_key = SECRET_KEY
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False


# --- Rotas de páginas ---

@app.route("/")
def index():
    """Serve a página principal."""
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Serve arquivos estáticos."""
    return send_from_directory(".", filename)


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

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("SELECT username FROM users WHERE username = %s", (username,))
        if cur.fetchone():
            return jsonify({"erro": "Este usuário já existe. Escolha outro."}), 409

        cur.execute(
            "INSERT INTO users (username, password_hash, nome) VALUES (%s, %s, %s)",
            (username, hash_password(password), nome or username)
        )
        conn.commit()

        session["username"] = username
        session["nome"] = nome or username

        return jsonify({"sucesso": True, "nome": nome or username})
    except Exception as e:
        conn.rollback()
        return jsonify({"erro": f"Erro ao registrar: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/api/login", methods=["POST"])
def login():
    """Faz login do usuário."""
    dados = request.get_json()
    username = dados.get("username", "").strip().lower()
    password = dados.get("password", "").strip()

    if not username or not password:
        return jsonify({"erro": "Usuário e senha são obrigatórios."}), 400

    conn = get_db()
    cur = conn.cursor()

    try:
        cur.execute("SELECT username, password_hash, nome FROM users WHERE username = %s", (username,))
        user = cur.fetchone()

        if not user:
            return jsonify({"erro": "Usuário ou senha incorretos."}), 401

        if not verify_password(user["password_hash"], password):
            return jsonify({"erro": "Usuário ou senha incorretos."}), 401

        session["username"] = username
        session["nome"] = user["nome"]

        return jsonify({"sucesso": True, "nome": user["nome"]})
    finally:
        cur.close()
        conn.close()


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
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT c.id, c.titulo, c.criada_em, c.atualizada_em,
                   COUNT(m.id) as mensagens_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversa_id = c.id
            WHERE c.username = %s
            GROUP BY c.id
            ORDER BY c.atualizada_em DESC
        """, (session["username"],))
        conversas = cur.fetchall()
        # Converter datetime para string
        resultado = []
        for conv in conversas:
            resultado.append({
                "id": conv["id"],
                "titulo": conv["titulo"],
                "criada_em": conv["criada_em"].isoformat() if conv["criada_em"] else "",
                "atualizada_em": conv["atualizada_em"].isoformat() if conv["atualizada_em"] else "",
                "mensagens_count": conv["mensagens_count"]
            })
        return jsonify(resultado)
    finally:
        cur.close()
        conn.close()


@app.route("/api/conversations", methods=["POST"])
@login_required
def create_conversation():
    """Cria uma nova conversa."""
    conn = get_db()
    cur = conn.cursor()
    try:
        nova_id = secrets.token_hex(8)
        agora = datetime.now()
        cur.execute(
            "INSERT INTO conversations (id, username, titulo, criada_em, atualizada_em) VALUES (%s, %s, %s, %s, %s)",
            (nova_id, session["username"], "Nova conversa", agora, agora)
        )
        conn.commit()
        return jsonify({
            "id": nova_id,
            "titulo": "Nova conversa",
            "criada_em": agora.isoformat(),
            "atualizada_em": agora.isoformat(),
            "mensagens": []
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"erro": f"Erro ao criar conversa: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()


@app.route("/api/conversations/<conv_id>", methods=["GET"])
@login_required
def get_conversation(conv_id):
    """Retorna uma conversa completa com mensagens."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, titulo, criada_em, atualizada_em FROM conversations WHERE id = %s AND username = %s",
            (conv_id, session["username"])
        )
        conv = cur.fetchone()
        if not conv:
            return jsonify({"erro": "Conversa não encontrada."}), 404

        cur.execute(
            "SELECT tipo, texto, timestamp FROM messages WHERE conversa_id = %s ORDER BY timestamp ASC",
            (conv_id,)
        )
        mensagens = cur.fetchall()

        return jsonify({
            "id": conv["id"],
            "titulo": conv["titulo"],
            "criada_em": conv["criada_em"].isoformat() if conv["criada_em"] else "",
            "atualizada_em": conv["atualizada_em"].isoformat() if conv["atualizada_em"] else "",
            "mensagens": [{"tipo": m["tipo"], "texto": m["texto"], "timestamp": m["timestamp"].isoformat()} for m in mensagens]
        })
    finally:
        cur.close()
        conn.close()


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@login_required
def delete_conversation(conv_id):
    """Exclui uma conversa."""
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("DELETE FROM conversations WHERE id = %s AND username = %s", (conv_id, session["username"]))
        conn.commit()
        return jsonify({"sucesso": True})
    except Exception as e:
        conn.rollback()
        return jsonify({"erro": f"Erro ao excluir: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()


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

    conn = get_db()
    cur = conn.cursor()

    try:
        # Verificar se a conversa existe e pertence ao usuário
        cur.execute(
            "SELECT id, titulo FROM conversations WHERE id = %s AND username = %s",
            (conv_id, session["username"])
        )
        conversa = cur.fetchone()
        if not conversa:
            return jsonify({"erro": "Conversa não encontrada."}), 404

        # Carregar histórico para contexto
        cur.execute(
            "SELECT tipo, texto FROM messages WHERE conversa_id = %s ORDER BY timestamp ASC",
            (conv_id,)
        )
        msgs = cur.fetchall()
        historico = []
        for msg in msgs:
            historico.append({
                "role": "user" if msg["tipo"] == "user" else "assistant",
                "content": msg["texto"]
            })

        # Processar a pergunta
        resposta = processar_pergunta(pergunta, historico)

        # Salvar mensagens no banco
        agora = datetime.now()
        cur.execute(
            "INSERT INTO messages (conversa_id, tipo, texto, timestamp) VALUES (%s, %s, %s, %s)",
            (conv_id, "user", pergunta, agora)
        )
        cur.execute(
            "INSERT INTO messages (conversa_id, tipo, texto, timestamp) VALUES (%s, %s, %s, %s)",
            (conv_id, "agent", resposta, agora)
        )

        # Atualizar título se for a primeira mensagem
        cur.execute("SELECT COUNT(*) as cnt FROM messages WHERE conversa_id = %s", (conv_id,))
        count = cur.fetchone()["cnt"]
        if count <= 2:
            titulo = pergunta[:50]
            if len(pergunta) > 50:
                titulo += "..."
            cur.execute(
                "UPDATE conversations SET titulo = %s, atualizada_em = %s WHERE id = %s",
                (titulo, agora, conv_id)
            )
        else:
            cur.execute(
                "UPDATE conversations SET atualizada_em = %s WHERE id = %s",
                (agora, conv_id)
            )

        conn.commit()
        return jsonify({"resposta": resposta})

    except Exception as e:
        conn.rollback()
        return jsonify({"erro": f"Ocorreu um erro: {str(e)}"}), 500
    finally:
        cur.close()
        conn.close()


# ============================================================
# INICIALIZAÇÃO
# ============================================================

# Criar tabelas ao iniciar
if DATABASE_URL:
    try:
        init_db()
        print("Banco de dados inicializado com sucesso!")
    except Exception as e:
        print(f"Erro ao inicializar banco: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("  AGENTE IA DO PROFESSOR - Versão 2.1")
    print("  Acesse: http://localhost:5000")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

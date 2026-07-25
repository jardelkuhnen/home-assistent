# AGENTS.md — Diretrizes de Arquitetura e Padrões Python

Este documento estabelece as diretrizes estritas, padrões de código Python e arquitetura para o desenvolvimento do projeto **`home-assistent-brain`** a partir do zero em um novo repositório.

---

## 📐 1. Visão Geral e Arquitetura
O sistema é um orquestrador multi-agentes de automação residencial e inteligência artificial baseado na arquitetura **Cérebro-Satélite**:
* **Cérebro (Orquestração & Core):** API FastAPI rodando **LangGraph** em Python, utilizando **Google Gemini 1.5** como motor cognitivo e de ferramentas (`@tool`).
* O motor de decisao é um repositório, que poderá ser trocado por qualquer outro LLM. 
* **Satélite (Borda Local):** Script de captação e conversão de áudio local utilizando **`faster-whisper`** (CPU/int8) para Speech-to-Text (STT) 100% offline.
* **Integração Física:** Comunicação REST com o **Home Assistant** para controle de dispositivos IoT (ex: Tomadas Wi-Fi MOES) e síntese de voz via **Alexa Media Player**.

---

## 📁 2. Estrutura de Diretórios Obrigatória

O repositório deve seguir rigidamente esta árvore de arquivos:

```text
home-assistent-brain/
├── .env.example            # Modelo de variáveis de ambiente obrigatórias
├── .gitignore              # Ignora .env, .venv, __pycache__, .mypy_cache, etc.
├── .pre-commit-config.yaml # Hooks de linting e typechecking automático
├── Makefile                # Automação de comandos (format, lint, typecheck, run)
├── pyproject.toml          # Configurações do Ruff, Mypy e dependências do projeto
├── AGENTS.md               # Este arquivo de diretrizes para o agente/LLM
├── api.py                  # Ponto de entrada do Cérebro (Servidor FastAPI)
├── satelite.py             # Ponto de entrada do Ouvido (Faster-Whisper STT)
└── src/
    ├── __init__.py
    ├── config.py           # Configurações tipadas via pydantic-settings
    ├── graph/              # Lógica do LangGraph
    │   ├── __init__.py
    │   ├── state.py        # TypedDict com o AgentState
    │   ├── nodes.py        # Nós do grafo (chatbot, tool_node)
    │   └── workflow.py     # Montagem e compilação do StateGraph
    ├── tools/              # Ferramentas expostas ao LLM (@tool)
    │   ├── __init__.py
    │   ├── weather.py      # Previsão do tempo
    │   ├── search.py       # Busca na web
    │   └── home.py         # Controle de dispositivos (Home Assistant)
    └── services/           # Clientes HTTP externos (Sem lógica de negócio)
        ├── __init__.py
        └── ha_client.py    # Abstração da API REST do Home Assistant
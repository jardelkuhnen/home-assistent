# Roadmap de Implementação — `home-assistent-brain`

Plano sequencial, modular e incremental para reconstruir o projeto do zero,
aderindo estritamente ao `AGENT.md` e às decisões registradas na sessão de
grilling (ver `CONTEXT.md` e `docs/adr/`).

> **Convenções do roadmap**
> - Raiz do repositório rotulada como `home-assistent-brain/` (o `AGENT.md` ainda
>   contém o rótulo residual `web-search-workflow/` herdado da referência; tratado
>   como correção).
> - `src` é o pacote (layout flat, `packages=["src"]`). Imports via `src.*`.
> - Critério de aceitação **comum a toda task que toca código**:
>   `make lint` e `make typecheck` sem erros; `make format` aplicado.
> - Critério de aceitação **comum a toda task que cria/altera tool ou service**:
>   testes unitários da unidade passam (`pytest tests/<unidade>`).
> - Segurança (baseline de 7 itens) é codificada como critério onde aplica:
>   1. Zero secrets no código (só `.env` + `.env.example`).
>   2. Timeout explícito em todo cliente HTTP.
>   3. Validação de input via Pydantic com constraints em toda `@tool`.
>   4. Auth `X-API-Key` no `POST /chat`.
>   5. HA token de longa duração com permissão mínima (documentado no `.env.example`).
>   6. `gitleaks` no pre-commit (além de ruff/mypy).
>   7. `.gitignore` cobre `.env`, `.venv`, `__pycache__`, `.mypy_cache`,
>      `.ruff_cache`, modelos Whisper baixados.

---

## Fase 1 — Governança, Infraestrutura e Ferramentaria

### Task 1.1 — Inicialização do repositório e `.gitignore`
**Objetivo:** Criar a base de versionamento ignorando secrets, caches e artefatos.
**Arquivos criados:**
- `home-assistent-brain/.gitignore`
**Critérios de aceitação:**
- Cobertura dos 7 itens da baseline de segurança relevantes a esta task
  (`.env`, `.venv`, `__pycache__`, `.mypy_cache`, `.ruff_cache`, modelos Whisper).
- `git status` limpo após criar `.env` e `.venv` locais (ambos ignorados).

### Task 1.2 — `pyproject.toml` (deps, Ruff, Mypy, pytest)
**Objetivo:** Declarar dependências, tooling e tipagem estrita.
**Arquivos criados:**
- `home-assistent-brain/pyproject.toml`
**Conteúdo decisivo:**
- `[project] dependencies`: `fastapi`, `uvicorn`, `langgraph`, `langchain`,
  `langchain-core`, `langchain-google-genai`, `langchain-openai`,
  `tavily-python`, `httpx`, `pydantic`, `pydantic-settings`, `faster-whisper`,
  `sounddevice`, `numpy`.
- `[project.optional-dependencies] dev`: `ruff`, `mypy`, `pre-commit`, `pytest`,
  `pytest-asyncio`, `respx`, `httpx` (test client).
- `[tool.ruff]`: `select = ["E","F","I","UP","B","SIM","N"]`, `line-length = 100`.
- `[tool.mypy]`: `strict = true`, `packages = ["src"]`.
- `[tool.pytest.ini_options]`: `testpaths = ["tests"]`.
- `[tool.setuptools]`: `packages = ["src"]` (e subpacotes `src.graph`,
  `src.tools`, `src.services`).
**Critérios de aceitação:**
- `pip install -e .[dev]` instala sem erros.
- `mypy --strict src` roda (pode reportar "no files" até a Task 1.4, sem erro de config).
- `ruff check .` roda (sem arquivos ainda, sem erro de config).

### Task 1.3 — `Makefile` e `.pre-commit-config.yaml`
**Objetivo:** Automação de comandos e hooks de lint/typecheck/secrets.
**Arquivos criados:**
- `home-assistent-brain/Makefile`
- `home-assistent-brain/.pre-commit-config.yaml`
**Alvos do Makefile:** `setup`, `install`, `format`, `lint`, `typecheck`, `test`,
`run-brain` (uvicorn api:app), `run-satellite` (python satelite.py), `check`
(lint + typecheck + test).
**Hooks do pre-commit:** `ruff check`, `ruff format`, `mypy`, `gitleaks`.
**Critérios de aceitação:**
- `make format`, `make lint`, `make typecheck`, `make test` executam sem erro
  de sintaxe do Makefile (podem não ter alvos ainda).
- `pre-commit run --all-files` instala e executa os hooks (gitleaks deve
  escanear o repositório sem encontrar secrets).

### Task 1.4 — `src/config.py` (Settings via pydantic-settings + `get_llm()`)
**Objetivo:** Base de configuração estrita tipada + factory do motor cognitivo
(ADR-0001).
**Arquivos criados:**
- `home-assistent-brain/src/__init__.py`
- `home-assistent-brain/src/config.py`
**Conteúdo decisivo:**
- `Settings(BaseSettings)` com `model_config = SettingsConfigDict(env_file=".env", extra="forbid")`.
  Campos: `llm_provider: Literal["gemini","openai"]`, `gemini_api_key: SecretStr`,
  `openai_api_key: SecretStr`, `openai_api_base: str`,
  `ha_url: AnyHttpUrl`, `ha_token: SecretStr`, `tavily_api_key: SecretStr`,
  `brain_api_key: SecretStr`, `alexa_media_entity: str`,
  `whisper_model: str = "small"`, `ha_timeout_s: float = 5.0`,
  `llm_timeout_s: float = 30.0`.
- `get_settings()` com cache (lru_cache).
- Protocol `CognitiveMotor` (método/bind compatível com `BaseChatModel`).
- `get_llm() -> CognitiveMotor`: retorna `ChatGoogleGenerativeAI` quando
  `llm_provider == "gemini"`, `ChatOpenAI` quando `"openai"`, com timeouts.
**Critérios de aceitação:**
- `make typecheck` (mypy strict) passa em `src/config.py`.
- `make lint` passa.
- `Settings` rejeita variáveis desconhecidas (`extra="forbid"`).
- Teste unitário: carregar `Settings` a partir de um `.env` de teste valida
  campos obrigatórios e rejeita extras.

### Task 1.5 — `.env.example` e `AGENT.md` corrigido
**Objetivo:** Template de variáveis + correção do rótulo residual da raiz.
**Arquivos criados/modificados:**
- `home-assistent-brain/.env.example` (criado)
- `home-assistent-brain/AGENT.md` (modificado: rótulo da árvore
  `web-search-workflow/` → `home-assistent-brain/`)
**Conteúdo do `.env.example`:**
- Todas as variáveis de `Settings`, com comentários explicativos.
- Nota sobre o `HA_TOKEN`: token de longa duração do Home Assistant com
  **permissão mínima** (apenas `services: call_service` para
  `notify.alexa_media`, `homeassistant`, e `switch`/`light` domains relevantes).
**Critérios de aceitação:**
- `.env.example` lista todas as chaves de `Settings` (paridade 1:1).
- `AGENT.md` com rótulo corrigido; árvore de diretórios inalterada nos nomes.
- Nenhum valor real de secret no `.env.example`.

---

## Fase 2 — Camada de Serviços Externos

### Task 2.1 — `src/services/ha_client.py` (Home Assistant + Alexa TTS)
**Objetivo:** Cliente HTTP para o HA: controle de dispositivos e síntese de voz
via Alexa Media Player (nó terminal do grafo — ADR-0002).
**Arquivos criados:**
- `home-assistent-brain/src/services/__init__.py`
- `home-assistent-brain/src/services/ha_client.py`
**Conteúdo decisivo:**
- `HomeAssistantClient` com `httpx.AsyncClient` configurado a partir de `Settings`
  (`base_url=ha_url`, `headers={"Authorization": f"Bearer {ha_token}"}`,
  `timeout=ha_timeout_s`).
- `async def call_service(domain, service, service_data) -> dict` — wrapper
  POST `/api/services/{domain}/{service}`.
- `async def toggle(entity_id) -> dict` — `homeassistant.toggle` (ou
  `switch.turn_on/off` conforme o domínio do `entity_id`).
- `async def speak(text: str) -> dict` — chama `notify.alexa_media` com
  `media_player_entity = alexa_media_entity`, `message = text`. Retorna a
  resposta do HA; **não levanta** em falha de TTS — captura `httpx.HTTPError`
  e retorna `{"ok": False, "error": str}` (o nó do grafo decide o que fazer).
- `async def get_state(entity_id) -> dict`.
- `async def close()`.
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- Timeout explícito (`ha_timeout_s`) em todas as chamadas.
- Testes com `respx`: mock de `/api/services/...` cobre sucesso, timeout
  (`httpx.TimeoutException`), e erro 4xx/5xx — `speak()` retorna
  `{"ok": False, ...}` em vez de propagar.

---

## Fase 3 — Ferramentas do Agente / Tools

### Task 3.1 — `src/tools/weather.py` (Open-Meteo)
**Objetivo:** Tool de clima com geocoding e formatação para voz.
**Arquivos criados:**
- `home-assistent-brain/src/tools/__init__.py`
- `home-assistent-brain/src/tools/weather.py`
**Conteúdo decisivo:**
- Schemas Pydantic: `WeatherInput(location: str)`, `WeatherOutput(summary: str)`.
- `@tool` `get_weather(location: str) -> str`:
  1. GET Open-Meteo geocoding (`/v1/search?name={location}`) → lat/lon.
  2. GET Open-Meteo forecast (`/v1/forecast`) → temperaturas máxima/mínima e
     código de tempo atual.
  3. Comprime em frase curta falável ("Máxima de 28, mínima de 19, pancadas à
     tarde").
  4. Fallback textual em falha ("Não consegui obter o clima agora") — sem
     propagar exceção.
- `httpx.AsyncClient` com timeout explícito.
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- Timeout explícito; nenhum `Any` (saídas tipadas).
- Testes com `respx`: geocoding + forecast mockados; caso de cidade não
  encontrada retorna a frase de fallback.

### Task 3.2 — `src/tools/search.py` (Tavily)
**Objetivo:** Tool de busca web defensiva, resultado digerido para voz.
**Arquivos criados:**
- `home-assistent-brain/src/tools/search.py`
**Conteúdo decisivo:**
- Schemas: `SearchInput(query: str, max_results: int = 3)`, `SearchOutput(answer: str)`.
- `@tool` `web_search(query: str) -> str`:
  1. `TavilyClient(api_key=tavily_api_key)` com timeout via `httpx` subjacente
     ou wrap síncrono seguro.
  2. Extrai `answer` (Tavily) ou compõe a partir dos `results`.
  3. Comprime em poucas frases (v1: usa `answer` cru, truncado).
  4. Fallback textual ("Não encontrei nada sobre isso") — sem propagar exceção.
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- `TAVILY_API_KEY` lida via `Settings` (nunca hardcoded).
- Timeout explícito; falha de rede retorna fallback, não exceção.
- Teste com `respx` mockando a API do Tavily (sucesso + erro).

### Task 3.3 — `src/tools/home.py` (controle de dispositivos)
**Objetivo:** Tool de automação com validação estrita de `entity_id` (baseline
de segurança item 3).
**Arquivos criados:**
- `home-assistent-brain/src/tools/home.py`
**Conteúdo decisivo:**
- Schemas: `HomeInput(action: Literal["on","off","toggle"], entity_id: str)`.
  `entity_id` validado por regex `^(switch|light|media_player)\..+`.
- `@tool` `control_device(action, entity_id) -> str`:
  1. Delega a `HomeAssistantClient.toggle/call_service`.
  2. Retorna confirmação falável ("Liguei a tomada da sala") baseada em
     `action` + um mapeamento `entity_id → apelido` (configurável em
     `Settings` ou um dict simples em memória no v1).
  3. Falha de HA → frase defensiva ("Não consegui acionar o dispositivo").
- **Não** implementa TTS aqui (TTS é nó do grafo — ADR-0002).
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- `entity_id` fora do padrão é rejeitado pela validação Pydantic (não chega ao HA).
- Teste com `respx` mockando `ha_client`: on/off/toggle sucesso + falha de
  rede; teste de schema rejeita `entity_id` inválido.

### Task 3.4 — `src/tools/__init__.py` (registro do catálogo de tools)
**Objetivo:** Ponto único exportando a lista de tools para o grafo.
**Arquivos modificados:**
- `home-assistent-brain/src/tools/__init__.py`
**Conteúdo decisivo:**
- `ALL_TOOLS: list[BaseTool] = [get_weather, web_search, control_device]`.
- Re-exports públicos.
**Critérios de aceitação:**
- `make typecheck` passa; `ALL_TOOLS` tipada como `list[BaseTool]`.
- `from src.tools import ALL_TOOLS` funciona.

---

## Fase 4 — O Cérebro / LangGraph

### Task 4.1 — `src/graph/state.py` (AgentState)
**Objetivo:** TypedDict com o estado do grafo (decisão Q6).
**Arquivos criados:**
- `home-assistent-brain/src/graph/__init__.py`
- `home-assistent-brain/src/graph/state.py`
**Conteúdo decisivo:**
```python
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    spoken: bool
    error: str | None
```
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- `Annotated[list[BaseMessage], add_messages]` corretamente tipado.

### Task 4.2 — System Prompt otimizado para voz
**Objetivo:** Prompt de sistema **sem Markdown**, em linguagem natural falável.
**Arquivos criados:**
- `home-assistent-brain/src/graph/prompt.py`
**Conteúdo decisivo:**
- Constante `SYSTEM_PROMPT: str` instruindo o motor a: responder em frases
  curtas e naturais para voz, sem Markdown/símbolos, sem listas numeradas,
  confirmar ações de automação em uma frase, e nunca inventar dados de clima
  ou busca (sempre usar a tool).
- Texto plano (sem `#`, `*`, ```).
**Critérios de aceitação:**
- `make typecheck` passa.
- Inspeção manual: prompt não contém caracteres de Markdown.
- (Opcional) teste asserta que `SYSTEM_PROMPT` não contém `#` nem `**`.

### Task 4.3 — `src/graph/nodes.py` (chatbot, tool_node, speak_node)
**Objetivo:** Nós do grafo, incluindo o nó terminal de TTS (ADR-0002).
**Arquivos criados:**
- `home-assistent-brain/src/graph/nodes.py`
**Conteúdo decisivo:**
- `async def chatbot_node(state) -> AgentState`: bind do `get_llm()` com
  `ALL_TOOLS`, injeta `SYSTEM_PROMPT`, invoca o modelo, devolve
  `{"messages": [ai_message]}`.
- `tool_node = ToolNode(ALL_TOOLS)` (ou wrapper async tipado).
- `async def speak_node(state) -> AgentState`: lê o conteúdo da última
  `AIMessage`, chama `ha_client.speak(text)`, seta `spoken` conforme
  `{"ok": True}`, e `error` em caso de falha. Sempre devolve uma resposta
  textual (a própria `AIMessage`) para a API.
- Injeção de dependências: `ha_client` recebido por parâmetro/factory (não
  global) para testabilidade.
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- Nós tipados como `Callable[[AgentState], AgentState]` (ou equivalente async).
- Teste unitário de `speak_node`: mock de `ha_client.speak` (ok e falha)
  valida `spoken`/`error`.

### Task 4.4 — `src/graph/workflow.py` (montagem e compilação)
**Objetivo:** Montar o `StateGraph[AgentState]` e compilar.
**Arquivos criados:**
- `home-assistent-brain/src/graph/workflow.py`
**Conteúdo decisivo:**
- `build_graph(ha_client) -> CompiledGraph`:
  - `add_node("chatbot", chatbot_node)`
  - `add_node("tools", tool_node)`
  - `add_node("speak", speak_node)` (nó terminal — ADR-0002)
  - `set_entry_point("chatbot")`
  - `add_conditional_edges("chatbot", route_tools, {"tools": "tools", "end": "speak"})`
  - `add_edge("tools", "chatbot")`
  - `add_edge("speak", END)`
- `route_tools(state)`: inspeciona tool calls da última `AIMessage`.
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- `build_graph(...)` retorna um grafo compilável sem erro em runtime.
- Teste de integração leve: grafo compilado com mocks de LLM e `ha_client`
  percorre `chatbot → speak → END` em um turno sem tool calls.

### Task 4.5 — `src/graph/__init__.py` (export público)
**Arquivos modificados:**
- `home-assistent-brain/src/graph/__init__.py`
**Conteúdo:** re-exports `build_graph`, `AgentState`, `SYSTEM_PROMPT`.
**Critérios de aceitação:** `make typecheck` passa; `from src.graph import build_graph` funciona.

---

## Fase 5 — Interfaces de Entrada e Saída

### Task 5.1 — `api.py` (FastAPI — Cérebro)
**Objetivo:** Servidor FastAPI expondo `POST /chat` (contrato Q5).
**Arquivos criados:**
- `home-assistent-brain/api.py`
**Conteúdo decisivo:**
- `app = FastAPI(title="home-assistent-brain")`.
- Dependência `verify_api_key(x_api_key: str = Header(...))` valida contra
  `Settings.brain_api_key` (baseline item 4).
- `POST /chat` recebe `ChatRequest{text: str}`, retorna
  `ChatResponse{reply: str, spoken: bool}`.
- Constrói `ha_client` (lifespan), `build_graph(ha_client)`, invoca o grafo
  com `{"messages": [HumanMessage(text)], "spoken": False, "error": None}`,
  extrai `reply` da última `AIMessage` + `spoken` do estado.
- `GET /health` sem auth.
- Execução: `uvicorn api:app` (alvo `make run-brain`).
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam em `api.py`.
- `TestClient` (httpx): `POST /chat` sem `X-API-Key` → 401/422; com chave
  válida + grafo mockado → 200 com `reply`/`spoken`.
- `GET /health` → 200.
- Cliente `httpx` com timeout explícito apontando para `BRAIN_URL` (nova
  variável em `Settings`, default `http://localhost:8123`).

### Task 5.2 — `satelite.py` (Faster-Whisper STT — Satélite)
**Objetivo:** Captação de voz local, push-to-talk, STT 100% offline (decisão Q7).
**Arquivos criados:**
- `home-assistent-brain/satelite.py`
**Conteúdo decisivo:**
- Carrega `WhisperModel(Settings.whisper_model, device="cpu", compute_type="int8")`.
- Loop push-to-talk: aguarda gatilho (tecla/flag CLI) → grava áudio com
  `sounddevice` → transcreve com `language="pt"` → `POST /chat` ao Cérebro
  com header `X-API-Key` → loga `reply` e `spoken`.
- Ponto de extensão documentado para trocar push-to-talk por VAD sem
  reescrever o pipeline (função `capture_audio()` isolada).
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam em `satelite.py`.
- Sem dependência de rede para o STT (Whisper roda offline).
- `BRAIN_URL` e `BRAIN_API_KEY` lidos via `Settings`.
- Teste unitário (sem hardware): `transcribe()` com um fixture de áudio curto
  retorna string; `send_to_brain()` com `respx` valida o `POST /chat` + header.

### Task 5.3 — Adicionar `BRAIN_URL` a `Settings` e `.env.example`
**Objetivo:** Fechar a variável de configuração introduzida na Task 5.2.
**Arquivos modificados:**
- `home-assistent-brain/src/config.py` (`brain_url: AnyHttpUrl = "http://localhost:8000"`)
- `home-assistent-brain/.env.example` (adiciona `BRAIN_URL`)
**Critérios de aceitação:**
- `make typecheck` e `make lint` passam.
- Paridade `Settings` ↔ `.env.example` mantida.

---

## Critérios globais de conclusão do roadmap

- `make check` (lint + typecheck + test) verde em todo o repositório.
- `pre-commit run --all-files` verde.
- `api.py` sobe via `make run-brain` e `GET /health` responde.
- `satelite.py` inicializa o Whisper sem erro (sem hardware, ao menos importa
  e instancia o modelo).
- Nenhum secret commitado; `gitleaks` limpo.
- Árvore de diretórios final coincide com o `AGENT.md` (com `tests/` e
  `docs/` adicionados como camadas paralelas, sem alterar a árvore interna
  de `src/`).

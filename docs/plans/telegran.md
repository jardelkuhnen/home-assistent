# Plano: Integração bidirecional Telegram ↔ Cérebro (LangGraph)

## Contexto

Hoje o ecossistema `home-assistent-brain` tem dois canais: o **Satélite** (voz, push-to-talk → STT → `/chat` → Alexa) e chamadas diretas à API. O PRD `docs/PRD_telegran.md` pede um **terceiro canal**, Telegram, bidirecional e por texto, **isolado** das notificações de voz: mensagens com `source="telegram"` não podem acionar a Alexa/Home Assistant.

O contrato atual do `/chat` é `{"text":...}` → `{"reply","spoken","error"}`, e o grafo **sempre** termina no nó `speak` (Alexa). Não há noção de canal de origem. Este PR adiciona `source` ao fluxo, um roteamento que pula `speak` para Telegram, um novo serviço `telegram_bot.py` (long polling) e o conjunto de env/config/testes — **sem quebrar o Satélite** (regressão zero, fiel ao DoD).

As 21 decisões foram fechadas em sessão de grilling. Este plano as reflete sem reabrir escolhas.

## Decisões-chave (resumo)

- **Contrato**: request mantém `text` (obrigatório) + `metadata` opcional (`Metadata(BaseModel)` com `source: str | None`, `session_id: str | None`). `source` é case-insensitive. Response ganha `source` + `metadata.tools_used`, mantém `reply`/`spoken`/`error`.
- **Default de source**: ausente/vazio/inválido → comportamento `satellite` (fala na Alexa). Só `"telegram"` normalizado pula Alexa. Satélite inalterado.
- **Roteamento**: em `route_tools`, sem tool calls + `source=="telegram"` → `END`; senão → `speak`. Nó `speak` mantém responsabilidade única.
- **session_id**: aceito e levado no `AgentState`, **não consumido** (checkpoint futuro).
- **Prompt**: `SYSTEM_PROMPT_TELEGRAM` dedicado (markdown/listas permitidos); `chatbot_node` escolhe conforme `source`.
- **Bot**: `python-telegram-bot` (long polling + handlers); `httpx.AsyncClient` no hop bot→cérebro reusando `brain_url`/`brain_api_key`/`brain_timeout_s` (90s) + header `X-API-Key`.
- **Segurança**: whitelist `ALLOWED_USERS` (env `"11111,22222"`); não-autorizado → ignorado + log `WARNING` com `user.id`.
- **UX**: reply em texto puro (sem `parse_mode`); `typing` reenviado a cada ~4s em task concorrente; fallback genérico ao usuário ("Algo deu errado, tente de novo."); handler com try/except amplo (bot nunca cai).
- **Tools**: Telegram usa as mesmas `ALL_TOOLS`.

## Arquivos a modificar/criar

### 1. `src/config.py` — novas settings
Adicionar ao `Settings` (que continua único, `extra="forbid"`):
- `telegram_bot_token: SecretStr` — obrigatório (sem default).
- `allowed_users: list[int]` — obrigatório; usar `field_validator` para aceitar env como string separada por vírgula (`"11111,22222"` → `[11111,22222]`). Não usar `case_sensitive` em pydantic (necessário validador custom).

### 2. `src/graph/state.py` — `AgentState`
- Adicionar `source: str | None` (default lógico `None`).
- Documentar que `source=None` ⇒ comportamento de voz.

### 3. `src/graph/prompt.py` — novo prompt
- Adicionar `SYSTEM_PROMPT_TELEGRAM` (permite markdown, listas, respostas mais longas; ainda proíbe inventar dados de clima/busca — usar tools).

### 4. `src/graph/nodes.py` — seleção de prompt + roteamento
- `chatbot_node`: ler `state.get("source")`; se `"telegram"` → `SYSTEM_PROMPT_TELEGRAM`, senão `SYSTEM_PROMPT`.
- `route_tools`: quando não há tool calls, ler `source`; se `source and source.lower()=="telegram"` → `"end"`, senão → `"speak"`. Atualizar o dict de edges em `workflow.py` de `{"tools":..., "end":"speak"}` para incluir o caso telegram→end (a chave `"end"` continua apontando para `speak`, e uma nova chave `"telegram_end"` aponta para `END`). Decidir nomes de chave de rota de acordo com o padrão do `add_conditional_edges` existente (`src/graph/workflow.py:26`).

### 5. `src/graph/workflow.py` — arestas
- Ajustar `add_conditional_edges` para suportar 3 destinos: `tools`, `speak` (satélite/alexa), `END` (telegram). `route_tools` retorna `"tools"` | `"speak"` | `"telegram_end"`.
- `speak` continua com `graph.add_edge("speak", END)`.

### 6. `api.py` — contrato + coleta de tools_used
- `ChatRequest`: adicionar `metadata: Metadata | None = None` (modelo `Metadata(BaseModel)` definido aqui ou em `src/config.py` — preferir em `api.py` junto das outras models, como `ChatRequest`/`ChatResponse`).
- `ChatResponse`: adicionar `source: str` e `metadata: ResponseMetadata` (`{"tools_used": list[str]}`).
- Em `chat()`: derivar `source = (request.metadata.source.lower() if request.metadata and request.metadata.source else "satellite")`; injetar `source` e `session_id` em `initial_state`.
- Coletar `tools_used` reaproveitando a lógica de `events_from_node_update` (ramo `node_name=="tools"`, `src/api.py:66`) — extrair nomes das tools já é feito lá; acumular em um `set` e devolver em `metadata.tools_used`.
- Determinar `source` do response a partir do estado resultante (ou do request normalizado).
- Manter regressão: sem `metadata` ⇒ `source="satellite"` ⇒ `speak` roda ⇒ `spoken=True` (como hoje).

### 7. `telegram_bot.py` (NOVO, raiz) — espelho do `satelite.py`
Estrutura:
- `logging.basicConfig(level=INFO)` + `logger = logging.getLogger("telegram_bot")`.
- Cliente HTTP ao cérebro: função `send_to_brain(text, chat_id, settings, client)` reusando padrão do `satelite.py:106` — `POST /chat` com `{"text":..., "metadata":{"source":"telegram","session_id":f"telegram_{chat_id}"}}`, header `X-API-Key`, timeout `brain_timeout_s`.
- Handler de texto (PTB `MessageHandler(filters.TEXT, ...)`):
  - Validar `effective_user.id` em `settings.allowed_users`; senão → `logger.warning("unauthorized user_id=%s", uid)` e retornar sem responder.
  - Iniciar task concorrente de `typing` (loop `await context.bot.send_chat_action(chat_id, "typing")` a cada 4s até um `asyncio.Event` setado no fim).
  - `await send_to_brain(...)`; extrair `reply = result.get("reply")`.
  - Cancelar task de typing.
  - Se `reply` vazio → mensagem fallback. Enviar reply como texto puro (`parse_mode=None`).
  - Try/except amplo: `httpx.HTTPError`/`httpx.TimeoutException`/`Exception` → `logger.error(..., exc_info=True)` + mensagem genérica. Nunca propagar.
- `async def main()`: construir `Application.builder().token(...).build()`, registrar handler, `application.run_polling()`.
- Bloco `if __name__ == "__main__": main()`.

### 8. `Makefile` — alvo `run-telegram`
- Adicionar `run-telegram: python telegram_bot.py` (alinhado a `run-satellite`).

### 9. `pyproject.toml` — dependência
- Adicionar `python-telegram-bot` a `dependencies`. (Manter `httpx` listado, já está.)

### 10. `.env.example` — novas vars
- Adicionar `TELEGRAM_BOT_TOKEN=` e `ALLOWED_USERS=` (com comentário de formato "11111,22222").

### 11. `README.md` — seção Telegram
- Adicionar seção: pré-requisitos (criar bot via BotFather, obter token), configurar `TELEGRAM_BOT_TOKEN`/`ALLOWED_USERS`, `make run-telegram`, fluxo bidirecional, isolamento da Alexa.

### 12. Testes (decisão 15)
- `tests/test_api.py`:
  - Request com `metadata.source="telegram"` (mock `astream` sem nó `speak`) → `spoken=False`, `source="telegram"`, `metadata.tools_used` populado.
  - Regressão: request sem `metadata` → `source="satellite"`, `spoken=True` (comportamento atual).
  - `source` case-insensitive (`"Telegram"` → telegram).
- `tests/test_graph.py`:
  - `route_tools` com `source="telegram"` e sem tool calls → `"telegram_end"` (ou a chave escolhida).
  - `route_tools` com `source=None`/`"satellite"` e sem tool calls → `"speak"`.
  - Atualizar helper `_state_with_reply`/literais de estado para incluir `source` quando necessário.
  - `chatbot_node` seleciona prompt telegram (verificar SystemMessage injetada via monkeypatch do LLM) — teste opcional, priorizar roteamento.
- `tests/test_telegram_bot.py` (NOVO):
  - Whitelist: autorizado chama `send_to_brain`; não-autorizado loga WARNING e não chama (mock PTB `Application`/`Update` construídos à mão, mock `send_to_brain`).
  - Montagem do payload `{"text","metadata":{"source":"telegram","session_id":"telegram_<chat_id>"}}` via `respx` no endpoint `/chat`.
  - Fallback: `send_to_brain` levanta `httpx.ReadTimeout` → handler envia mensagem genérica, não propaga.
  - Reenvio de `typing`: `send_chat_action` chamado ≥1x; verificar cancelamento.
- `tests/conftest.py`: adicionar `TELEGRAM_BOT_TOKEN`/`ALLOWED_USERS` ao `_ENV` (para o bot instanciar `Settings` nos testes). Campos de HA/whisper continuam obrigatórios (já estão no `_ENV`).

## Padrões a reusar (não reimplementar)
- `satelite.py:106` `send_to_brain` — padrão do hop HTTP ao cérebro (header, timeout, `raise_for_status`).
- `api.py:58` `events_from_node_update` — extração de nomes de tools para `tools_used`.
- `src/graph/nodes.py:22` `content_to_text` — já usado para extrair reply limpo.
- `src/services/ha_client.py` `HomeAssistantClient` — sem mudança; `speak_node` continua chamando-o só no caminho de voz.

## Verificação (end-to-end)

1. **Testes**: `make check` (lint + typecheck + test). Verificar que `test_satelite.py` segue verde (regressão zero).
2. **Typecheck**: `mypy --strict src` — atenção a `AgentState` (TypedDict) e novas models pydantic.
3. **Cérebro manual**:
   - `make run-brain`.
   - `curl -X POST localhost:8000/chat -H "X-API-Key:..." -d '{"text":"clima em cascavel","metadata":{"source":"telegram"}}'` → `spoken:false`, `source:"telegram"`, `metadata.tools_used:["get_weather"]`.
   - Mesmo curl **sem** metadata → `spoken:true` (satélite/alexa, como hoje).
4. **Bot manual**:
   - Configurar `.env` (`TELEGRAM_BOT_TOKEN`, `ALLOWED_USERS` com seu user id).
   - `make run-telegram`.
   - Mandar "qual a previsão do tempo?" pelo Telegram → receber resposta de texto, "digitando..." visível durante o turno, sem aviso na Alexa.
   - Mandar de um usuário não autorizado → ignorado; checar log `WARNING` no console do bot.
   - Derrubar o cérebro (`Ctrl-C`) e mandar mensagem → bot responde "Algo deu errado..." e **continua rodando**; subir o cérebro de novo e confirmar retomada.

## Fora deste PR
- Checkpointer de memória por `session_id` (decisão 2).
- Dockerização do bot (decisão 7).
- `/start`, `/help`, tratamento de mídia (decisão 16).
- HTML/MarkdownV2 no envio (decisão 11).

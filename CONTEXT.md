# home-assistent-brain

Orquestrador multi-agente de automação residencial e IA na arquitetura
Cérebro-Satélite: um Cérebro (FastAPI + LangGraph) que decide e aciona ferramentas, e
um Satélite (borda local) que captura voz offline.

## Language

**Cérebro**:
O núcleo de orquestração — API FastAPI que roda o grafo LangGraph e expõe as `@tool` ao LLM.
_Avoid_: servidor, backend, app.

**Satélite**:
O processo de borda local responsável por capturar áudio e converter em texto (STT) via `faster-whisper`, 100% offline.
_Avoid_: cliente, listener, microservice.

**Motor cognitivo**:
O LLM que raciocina e decide qual ferramenta acionar. É um componente trocável, isolado atrás de uma factory — nunca acoplado diretamente ao grafo.
_Avoid_: modelo, engine, brain (ambíguo com Cérebro).

**Tool**:
Uma capacidade exposta ao motor cognitivo via `@tool` (busca, clima, controle de dispositivos). Tem schema Pydantic e design defensivo.
_Avoid_: função, skill, ação.

**Integração Física**:
A camada de comunicação REST com o Home Assistant (dispositivos IoT) e síntese de voz via Alexa Media Player.
_Avoid_: integração (genérico), conexão.

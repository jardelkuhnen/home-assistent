Especificação Técnica: Integração Bidirecional Telegram ↔ LangGraph1. 

# Visão Geral e Objetivos
Adicionar suporte ao Telegram como canal bidirecional no ecossistema home-assistent. O sistema deve permitir que um usuário autorizado envie mensagens de texto pelo chat do Telegram, processe a requisição através do endpoint /chat (LangGraph) e receba a resposta gerada diretamente no próprio chat do Telegram, mantendo o isolamento em relação às notificações de voz (Home Assistant / Alexa).2. Arquitetura e Fluxo de Dados

[ Usuário (Telegram) ]
         │  (Mensagem de Texto)
         ▼
[ telegram_bot.py ] ──(POST /chat)──► [ Cérebro (LangGraph) ]
         ▲                                   │
         │                                   ├─► Executa Tools (Clima, Web, etc.)
         │                                   │
         │                                   ▼
         │                            [ Roteador de Saída ]
         │                              │              │
         │                   (source="telegram")  (source="alexa" / "satellite")
         │                              │              │
         │                              ▼              ▼
         └──────── (JSON Response) ─────┘     [ Home Assistant / Alexa ]


## Contratos de API (Endpoint /chat)
O endpoint /chat deve ser ajustado para receber metadados de contexto do cliente, permitindo identificar o canal de origem e gerenciar sessões independentes.3.1. Request Payload (JSON)CampoTipoObrigatórioDescriçãoExemplomessagestringSimTexto do comando ou pergunta do usuário"Qual a previsão do tempo?"sourcestringSimIdentificador do canal de origem (telegram, satellite, alexa)"telegram"session_idstringNãoIdentificador único da sessão/conversa no canal"telegram_123456789"Exemplo de Request (POST /chat)
```
{
  "metadata": {
    "source": "telegram",
  },
  "message": "Como está o clima hoje?",
  "session_id": "telegram_123456789"
}

```

3.2. Response Payload 
Resposta textual final gerada pelo LangGraphsourcestringCanal processado na requisiçãometadataobjectOpcional. Informações extras de execução (tools acionadas, tempo, etc.)Exemplo de Response (200 OK):
```
{
  "response": "Hoje o dia em Cascavel está ensolarado, com máxima de 26°C.",
  "source": "telegram",
  "metadata": {
    "tools_used": ["weather_tool"]
  }
}
```

4. Modificações no Cérebro (LangGraph)
Leitura do Contexto de Origem: O estado do grafo (GraphState) deve incluir o atributo source vindo da requisição HTTP. Nó de Roteamento Final (Output Router)
Se source == "telegram": O grafo deve finalizar a execução retornando apenas a resposta textual no payload HTTP. 
Não acionar a tool/serviço de notificação de áudio do Home Assistant/Alexa. Se source == "satellite" ou "alexa" O comportamento atual é mantido, disparando a notificação no Home Assistant ao término do processamento.

Isolamento de Sessão: O session_id (telegram_<chat_id>) deve ser utilizado nos checkpointers de memória (caso ativos) para evitar cruzamento de contexto entre o Telegram e o satélite de voz.

# Especificação do Serviço (telegram_bot.py)
O bot operará como um serviço desacoplado na mesma rede (ou no mesmo host) da API do cérebro.
### 5.1. Requisitos Não-Funcionais
Modo de Operação: Long Polling (evita necessidade de IP público ou webhooks com HTTPS externo). Cliente HTTP: Assíncrono (httpx.AsyncClient) com timeout configurado para 60 segundos (para acomodar o tempo de raciocínio do LangGraph e chamada de tools).
Segurança (Whitelist): O serviço deve validar explicitamente o ID do usuário (effective_user.id). Mensagens de IDs não listados na variável de ambiente ALLOWED_USERS devem ser descartadas ou respondidas com mensagem de acesso negado.

5.2. Fluxo de Execução do Handler de TextoIntercepta mensagem de texto recebida. Valida se o usuário remetente está autorizado (ALLOWED_USERS).Envia o status interativo de "Digitando..." (typing) para o chat do Telegram.Monta o payload contendo:message: Texto do usuário.source: "telegram".session_id: "telegram_<chat_id>".Executa request POST para BRAIN_API_URL.
Trata exceções (timeout, falha de rede, erro 500 do cérebro) com mensagens de fallback legíveis no chat.Devolve o texto em response para o usuário no chat correspondente.6. 
Critérios de Aceitação (DoD)
[ ] O endpoint /chat aceita requisições contendo source="telegram" sem quebrar a interoperabilidade com satelity.py.
[ ] Mensagens originadas com source="telegram" não geram aviso sonoro na Alexa ou no Home Assistant.
[ ] O bot responde com sucesso a perguntas que exigem execução de tools (ex.: tempo ou busca na web).
[ ] Usuários com ID do Telegram fora da lista de autorizados recebem recusa ou são ignorados pelo serviço.
[ ] Quedas ou timeouts da API /chat são tratados graciosamente pelo bot sem encerrar o processo.
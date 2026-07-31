0. Pré-requisitos (uma única vez)

Crie o bot e descubra seu user id:

1. No Telegram, fale com @BotFather (https://t.me/BotFather) → /newbot → escolha nome e username → copie o token (123456:ABC...).
2. Descubra seu user_id: mande qualquer mensagem ao bot recém-criado e rode o bot (passo 4) — ele loga WARNING unauthorized user_id=NNNNN. Ou use o @userinfobot (https://t.me/userinfobot).

1. Configure o .env

cp .env.example .env   # se ainda não existir

Preencha/verifique no .env (além das chaves já existentes: LLM_PROVIDER, GEMINI_API_KEY/etc., HA_URL, HA_TOKEN, ALEXA_MEDIA_ENTITY, TAVILY_API_KEY, BRAIN_API_KEY):

TELEGRAM_BOT_TOKEN=123456:ABC...
ALLOWED_USERS=NNNNN        # seu user_id (ou vários: 11111,22222)

2. Instale a nova dependência

pip install -e ".[dev]"

3. Garantia automática (lint + tipos + testes)

make check
# ruff ✓, mypy --strict ✓, 54 testes ✓

4. Suba o Cérebro (terminal 1)

make run-brain

Valide:

curl http://localhost:8000/health
# {"status":"ok"}

5. Teste o contrato do /chat por canal (isolamento)

Telegram — deve pular a Alexa (spoken:false, source:"telegram", tool usada):

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $BRAIN_API_KEY" \
  -d '{"text":"clima em cascavel","metadata":{"source":"telegram"}}'
# Esperado: {"reply":"...","spoken":false,"source":"telegram","metadata":{"tools_used":["get_weather"]},"error":null}

Sem metadata (regressão) — deve acionar a Alexa (spoken:true, como antes):

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $BRAIN_API_KEY" \
  -d '{"text":"clima em cascavel"}'
# Esperado: spoken:true, source:"satellite"

▎ Confirme que nenhum dos dois acionou a Alexa no caso do source:"telegram" — esse é o ponto central do isolamento.

6. Suba o bot (terminal 2)

make run-telegram
# Log: "Telegram bot iniciado | allowed_users=[NNNNN]"

7. Teste bidirecional pelo app do Telegram

Com o usuário autorizado:

1. Mande: qual a previsão do tempo?
  - ✅ Recebe resposta de texto.
  - ✅ "digitando..." visível durante o turno.
  - ✅ Sem aviso na Alexa em casa.
2. Mande algo que use outra tool: ligue a luz da sala → confirmação em texto (sem fala).
3. Acione a Alexa pelo Satélite (voz) em paralelo e confirme que a voz continua funcionando — os canais não interferem.

8. Teste de segurança (whitelist)

De um outro usuário (ou um user_id fora de ALLOWED_USERS):

- Mande qualquer mensagem.
- ✅ Sem resposta no Telegram.
- ✅ No console do bot (terminal 2): WARNING unauthorized user_id=...

9. Teste de resiliência (bot nunca cai)

1. Com o bot rodando, derrube o Cérebro: Ctrl-C no terminal 1.
2. Mande uma mensagem pelo Telegram.
  - ✅ Recebe: "Algo deu errado, tente de novo."
  - ✅ Bot continua rodando (sem traceback/crash).
  - ✅ Log no terminal 2: ERROR brain request failed: ....
3. Suba o Cérebro de novo (make run-brain) e mande outra mensagem → ✅ retomada normal.

10. O que confirma sucesso (DoD)

┌───────────────────────────────┬───────────────────────────────────────────────────────────────┐
│           Critério            │                        Como verificar                         │
├───────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Regressão zero (Satélite/voz) │ test_satelite.py verde + Alexa continua falando pelo Satélite │
├───────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Telegram pula Alexa           │ curl com source:"telegram" → spoken:false                     │
├───────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Isolamento                    │ usuário não autorizado ignorado + log WARNING                 │
├───────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Resiliência                   │ cérebro derrubado → bot responde fallback e sobrevive         │
├───────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Tools no Telegram             │ metadata.tools_used populado no curl                          │
├───────────────────────────────┼───────────────────────────────────────────────────────────────┤
│ Qualidade                     │ make check 100% verde                                         │
└───────────────────────────────┴───────────────────────────────────────────────────────────────┘

Resumo dos comandos

# T1
make run-brain

# T2 (depois do user_id descoberto)
make run-telegram

# Verificações rápidas
curl http://localhost:8000/health
curl -X POST localhost:8000/chat -H "X-API-Key:$BRAIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"clima em cascavel","metadata":{"source":"telegram"}}'

Se quiser, posso rodar agora os passos puramente locais (2, 3 e os curls de contrato com um mock) para validar a parte que não depende do seu token/Telegram real.
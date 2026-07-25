# Síntese de voz (TTS) como nó final do grafo, não como tool do LLM

A saída de voz via Alexa Media Player é modelada como um **nó terminal** do
StateGraph: toda resposta textual produzida pelo motor cognitivo é roteada a esse nó,
que chama `ha_client.speak()` (serviço `notify.alexa_media` do Home Assistant).

Rejeitamos a alternativa de expor `falar_com_usuario` como uma `@tool` que o LLM
decide invocar. Num assistente de voz, o LLM "esquecer" de chamar a tool de fala é
uma falha de UX grave e silenciosa — o usuário fica sem resposta. Tratando a Alexa
como canal de saída obrigatório, garantimos que toda resposta seja falada; a
confirmação de ações ("Liguei a tomada") emerge do texto natural do motor e segue o
mesmo canal. `src/tools/home.py` fica responsável apenas por controle/estado de
dispositivos, sem sobrepor responsabilidades com a fala.

# LLM motor abstraído atrás de factory própria

O `AGENT.md` fixa "Google Gemini 1.5" como motor, mas também afirma que o motor é
"um repositório que poderá ser trocado por qualquer outro LLM". Em vez de instanciar
`ChatGoogleGenerativeAI` diretamente no grafo, isolamos a construção do modelo atrás de
uma factory `get_llm()` (com protocol tipado) em `src/config.py`, com Gemini
(`langchain-google-genai`) e OpenAI-compatível (`langchain-openai`) como backends
selecionáveis por configuração. Decidimos pela abstração desde o início — e não só
quando houver uma segunda necessidade — porque a cláusula de troca já é um requisito
explícito do contrato, e o custo de refatorar o binding do grafo depois é maior que o
de manter uma indireção agora.

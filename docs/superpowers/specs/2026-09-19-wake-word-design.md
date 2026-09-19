# Wake word no Satélite — design

Issue: jardelkuhnen/home-assistent#5

## Objetivo

Trocar o push-to-talk (segurar Shift, one-shot) por ativação hands-free com a
wake word **"hey jarvis"**: o Satélite fica escutando, ao ouvir a frase grava
o comando até detectar silêncio, transcreve (STT offline) e envia ao Cérebro
via `POST /chat`. Depois volta a escutar.

## Decisões (fechadas no brainstorming)

| Tema | Decisão |
|---|---|
| Fim da fala | VAD por silêncio, com teto de segurança. Sem fallback de Shift |
| Hardware alvo | Raspberry Pi / mini-PC dedicado, headless, sempre ligado |
| Feedback ao detectar | Só log no stderr (comportamento atual). Sem beep, sem aviso pela Alexa |
| Motor de wake word | openWakeWord, modelo pré-treinado `hey_jarvis`, backend ONNX |
| VAD | `openwakeword.vad.VAD` (mesma dependência; o modelo `silero_vad.onnx` é baixado junto com os de wake word) |
| `pynput` | Removido: depende de teclado/X, inexistente num Pi headless |

Descartados: Porcupine (exige AccessKey e ativação online, contra o princípio
offline) e detecção da frase pelo próprio Whisper (CPU contínua, latência e
falsos positivos).

## Design

### `capture_audio()` continua sendo a única costura

Contrato externo inalterado: `capture_audio() -> bytes` (PCM mono int16,
16 kHz) devolve **uma fala**, ou `b""`. Quem muda é o interior:

```text
abre InputStream (16 kHz, mono, int16, blocos de 80 ms = 1280 amostras)
├─ ESCUTANDO  bloco → wake_score(bloco); score ≥ wake_word_threshold → GRAVANDO
└─ GRAVANDO   acumula blocos (só os posteriores à detecção); is_speech(bloco)
              ├─ fala → zera contador de silêncio
              ├─ end_silence_s de silêncio após ter ouvido fala → fim
              ├─ nenhuma fala em no_speech_timeout_s após o wake → aborta (b"")
              └─ max_record_s → fim forçado
fecha o stream → retorna os bytes
```

- **Um stream por fala.** Abre em ESCUTANDO e fecha ao retornar. Durante
  transcrição e chamada ao Cérebro o microfone fica fechado, o que reduz a
  chance de recapturar a resposta da Alexa.
- **O áudio do "hey jarvis" não entra no buffer**, então o Whisper não
  transcreve a frase de ativação.
- **Falso positivo sem voz** cai no `no_speech_timeout_s` e devolve `b""`;
  `run_once` já responde "Nada transcrito." e o loop volta a escutar.

### Máquina de estados isolada e testável

A lógica de estados vive em uma função pura:

```python
def _listen(blocks: Iterable[np.ndarray],
            wake_score: Callable[[np.ndarray], float],
            is_speech: Callable[[np.ndarray], bool],
            settings: Settings) -> bytes: ...
```

`capture_audio` só faz a cola: abre o `InputStream`, monta os detectores reais
(openWakeWord `Model(inference_framework="onnx")` e `VAD`) e chama `_listen`.
Os detectores são injetados, então trocar o VAD por outro é mudar uma função.

### Loop e ciclo de vida

- `main()` passa a rodar `run_forever()`, um `while True: await run_once()`.
- **Ctrl+C precisa encerrar limpo.** `capture_audio` roda em `asyncio.to_thread`
  e fica bloqueado esperando o wake word; sem cuidado, o `asyncio.run` espera
  essa thread para sempre ao cancelar. Solução: um `threading.Event` de parada
  (`_stop`). O gerador de blocos do stream consulta o evento a cada 0,2 s
  (`queue.get(timeout=0.2)`) e termina quando ele é setado; `run_forever` seta
  o evento em um `finally`, ou seja, ao ser cancelado.
- `run_once` não transcreve áudio vazio: se `capture_audio` devolver `b""`
  (falso positivo sem voz), imprime "Nada capturado." e volta a escutar. Sem
  isso o Whisper receberia um array vazio.
- O tratamento de timeout/erro HTTP do Cérebro já está dentro de `run_once`,
  então o loop sobrevive a um Cérebro fora do ar.
- `transcribe()` hoje recarrega o `WhisperModel` a cada chamada; num loop isso
  custaria segundos por comando. O modelo passa a ser carregado uma vez
  (`functools.lru_cache`, mesmo idioma de `get_settings`).
- Os modelos de wake word e VAD são criados uma vez por processo, não por fala.

### Configuração

Novas settings, todas com default (não quebram `.env` existente, dado
`extra="forbid"`):

| Setting | Default |
|---|---|
| `wake_word_model` | `hey_jarvis` |
| `wake_word_threshold` | `0.5` |
| `end_silence_s` | `1.0` |
| `no_speech_timeout_s` | `5.0` |
| `max_record_s` | `15.0` (substitui a constante `_MAX_RECORD_SECONDS = 30`) |

O limiar do VAD é constante interna (valor fixo, sem configuração). Frames do
VAD devem dividir o bloco de 1280 amostras (ex.: `frame_size=640`).

`pyproject.toml`: entra `openwakeword`, sai `pynput`. `.env.example` e README
atualizados (o README também deixa de mencionar push-to-talk).

### Provisionamento

Nada vem no pacote: nem os modelos de wake word, nem os de features
(melspectrogram/embedding), nem o `silero_vad.onnx` do VAD. Um passo
explícito, uma vez por dispositivo e com rede, documentado no README:

```bash
python -c "import openwakeword; openwakeword.utils.download_models(['hey_jarvis'])"
```

`download_models` baixa sempre os modelos de features e o VAD, além dos
modelos pedidos (`.tflite` e `.onnx`). Se algum arquivo faltar, `Model(...)` ou
`VAD()` falha na abertura; o erro propaga (fail fast), sem download implícito.

### Erros e casos de borda

| Caso | Comportamento |
|---|---|
| Falso positivo sem voz | Aborta em `no_speech_timeout_s`, `b""`, volta a escutar |
| Fala longa | Corta em `max_record_s` e segue o pipeline |
| Ruído antes do wake word | Ignorado, não entra no buffer |
| Microfone indisponível (`PortAudioError`) | Falha rápida com mensagem clara; restart fica com o supervisor (systemd). Sem retry no código |
| Cérebro fora do ar / timeout | Tratado em `run_once`; o loop continua |
| Eco da resposta da Alexa | Stream fechado durante processamento. **Risco residual aceito:** o áudio da Alexa tocar depois de o mic reabrir só vira problema se a resposta contiver "hey jarvis". Sem cooldown agora |

### Testes

- `_listen` com blocos sintéticos e detectores falsos: wake+fala+silêncio
  devolve só o áudio pós-wake; wake sem voz devolve `b""` em
  `no_speech_timeout_s`; fala contínua para em `max_record_s`; ruído anterior
  ao wake não vaza para o buffer.
- Settings: defaults novos presentes; `.env` antigo continua válido.
- Removidos: `_is_shift` e seu teste. Mantidos: `run_once`, `send_to_brain`,
  `transcribe` (o stub do teste de `transcribe` se ajusta ao cache do modelo).
- Sem teste automatizado: o `while True` de `main()` (uma linha) e a detecção
  real com modelo e microfone. Esta última é um **teste de fumaça manual no
  Pi**, e o resultado entra no relatório de entrega.

## Verificado na documentação oficial

- openWakeWord espera PCM 16-bit a 16 kHz, em blocos múltiplos de 80 ms;
  `predict()` devolve score 0–1 por bloco; ONNX Runtime é suportado em todas as
  plataformas (TFLite não no Windows); os modelos são baixados por
  `openwakeword.utils.download_models()`.
- `openwakeword.vad.VAD`: `predict(x, frame_size=480)` devolve a probabilidade
  média de fala, a entrada tem de ser múltiplo de `frame_size` e `reset_states()`
  existe. O modelo ONNX do VAD **não** vem incluído: é baixado por
  `download_models()`.
- O VAD do faster-whisper não tem API pública documentada por bloco e usa um
  modelo versionado (`silero_vad_v6.onnx`); por isso não foi escolhido.

## Confirmado no código instalado (openwakeword 0.6.0, Python 3.12)

1. `Model.__init__` aceita `inference_framework` (`"tflite"` default, ou
   `"onnx"`). Nomes pré-treinados são resolvidos por
   `get_pretrained_model_paths(inference_framework)`, então
   `Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")` procura
   o `.onnx`.
2. `VAD.predict` faz `x/32767` internamente: recebe **int16 bruto**, exatamente o
   que o `InputStream` entrega, sem conversão. O comprimento deve ser múltiplo
   de `frame_size`; `frame_size=640` divide o bloco de 1280 amostras.
3. `Model.predict` devolve um dict de scores indexado pelo nome passado em
   `wakeword_models` (aqui, a chave `"hey_jarvis"`).

**Por que ONNX e não o default tflite:** a documentação do próprio
openWakeWord diz que tflite é mais eficiente em x86/ARM64, mas o
`tflite-runtime` não é garantido para Python 3.12 (não verificado no Pi) e o
`pip install openwakeword` neste Mac trouxe só o `onnxruntime`. ONNX é o
denominador comum (Mac de desenvolvimento e Pi) e o Makefile
fixa o 3.12. Se o consumo de CPU no Pi for problema, reavaliar tflite; o teste
de fumaça mede isso.

## Teste de fumaça (Mac x86, Python 3.12, voz sintética do `say`)

`Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")` e `VAD()`
instanciam e rodam com blocos int16 de 1280 amostras:

- `predict()` devolve `{'hey_jarvis': float}`; `VAD.predict(..., frame_size=640)`
  devolve `float32`.
- Fluxo silêncio + "hey jarvis" + silêncio + comando + silêncio: o score do
  wake word foi de 0.08 para 0.82 e 0.99 no fim da frase (limiar 0.5 dispara
  no bloco de 1.76 s) e ficou em 0.000 durante o comando. O VAD deu ≥0.97 em
  fala e ≤0.05 em silêncio, com separação limpa em torno de 0.5.
- Custo: ~4.4 ms por bloco de 80 ms (~5% de um core) neste Mac. **O Pi não foi
  medido**; é o item principal do teste de fumaça no dispositivo.

Limites: voz sintética, sem microfone real nem ruído de sala. Limiares finais
se ajustam no dispositivo.

**Consequência para o design:** o `Model` e o `VAD` guardam estado interno
(buffer de features do wake word, LSTM do VAD) e vivem o processo inteiro. No
início de cada `capture_audio`, chamar `model.reset()` e `vad.reset_states()`
(ambos existem e rodaram no teste), para que áudio de uma fala anterior não
contamine a próxima. Não é preciso resetar de novo ao entrar em GRAVANDO: o VAD
só é chamado nesse estado, então o estado dele continua zerado até lá.

## Licença

Código do openWakeWord: Apache 2.0. Modelos pré-treinados (incluindo
`hey_jarvis`): CC BY-NC-SA 4.0, ou seja, **uso não comercial**. Adequado para
uso doméstico.

## Fora de escopo

Beep/feedback sonoro, cooldown pós-resposta, fallback de push-to-talk, wake
word customizada, otimização de latência do Whisper `small` no Pi.

# Embedding Contract

[![ci](https://github.com/alexandryasukaitisss-cmd/embedding-contract/actions/workflows/ci.yml/badge.svg)](https://github.com/alexandryasukaitisss-cmd/embedding-contract/actions/workflows/ci.yml)

Embedding Contract не даёт смешать в памяти векторы от разных моделей. Если изменились модель, размерность, то, как готовятся запросы и документы, или локальный файл модели, старые векторы считаются устаревшими и их нужно пересчитать. Пакет также дозаполняет отсутствующие векторы батчами и не трогает валидные.

По умолчанию всё локально: Gemma `embeddinggemma-2:270m` через Ollama. Опциональный маршрут Gemini использует `gemini-embedding-2`; он отправляет текст в Google и никогда не выбирается сам.

Это логичное продолжение **[MemPalace Retrieval](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval)** и **[Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger)**. Retrieval решает, что можно отдавать; этот слой решает, какие векторы могут лежать в одном индексе и как безопасно их дозаполнить.

Версия 0.1.0. Python 3.11+. MIT. Без runtime-зависимостей.

English version: [README.md](README.md).

## Зачем

Большинство ошибок памяти — не ошибки ранжирования, а ошибки идентичности:

- вектор от `model-a` сравнивают с вектором от `model-b`;
- одна модель используется с другим префиксом или другой размерностью;
- старый вектор переживает переиндексацию и молча выигрывает;
- облачный fallback возвращает правдоподобный вектор из другого пространства.

Пакет делает эти состояния явными и отклоняет их.

## Две модели, два смысла

| Маршрут | Модель | Можно скачать? | Примечание |
| --- | --- | --- | --- |
| Локальный, по умолчанию | **`embeddinggemma-2:270m`** | **Да**, через Ollama | 768 измерений. Это Google **Gemma**, не Gemini. Модель работает на вашей машине, текст никуда не уходит. |
| Облачный, опционально | **`gemini-embedding-2`** | **Нет**, только API | Актуальная Gemini embedding-модель. 768 / 1536 / 3072 измерений. Отправляет текст в Google. |
| Облачный, legacy | `gemini-embedding-001` | Нет, только API | Использует `taskType`; пространство **несовместимо** с `gemini-embedding-2`. |

`text-embedding-004` использовать нельзя: модель отключена 14 января 2026. Официальная документация: [Gemini embeddings](https://ai.google.dev/gemini-api/docs/embeddings).

**Если нужна именно загружаемая Google-модель — скачивайте Gemma:**

```sh
ollama pull embeddinggemma-2:270m
```

Gemini нельзя скачать: это API. Пакет поддерживает оба маршрута, но это разные провайдеры с разной идентичностью и разной приватностью.

## Установка

```sh
git clone https://github.com/alexandryasukaitisss-cmd/embedding-contract.git
cd embedding-contract
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

## Проверка

Локальная Gemma через Ollama:

```sh
ollama pull embeddinggemma-2:270m
python -m embedding_contract.cli health \
  --provider ollama \
  --model embeddinggemma-2:270m \
  --dimensions 768 \
  --base-url http://127.0.0.1:11434
```

Если Ollama запущен изолированным сервисом на другом loopback-порту, укажите его явно. Скачивается та же модель.

Облачный Gemini:

```sh
read -s GEMINI_API_KEY
export GEMINI_API_KEY
python -m embedding_contract.cli health \
  --provider gemini \
  --model gemini-embedding-2 \
  --dimensions 768
```

Облачный адаптер включается только явно. Автоматического перехода между провайдерами нет.

## Embed

```sh
python -m embedding_contract.cli embed \
  --provider ollama \
  --model embeddinggemma-2:270m \
  --dimensions 768 \
  --purpose query \
  --text "где мы решили хранить ledger?"
```

В выводе — вектор и точные метаданные идентичности, которые нужно сохранить вместе с ним.

## Backfill

Файлы `examples/` показывают поля конструктора; CLI использует явные флаги и автоматически применяет Gemma retrieval-префиксы для `embeddinggemma-2:270m`.

```sh
python -m embedding_contract.cli backfill \
  --provider ollama \
  --model embeddinggemma-2:270m \
  --dimensions 768 \
  --input records.jsonl \
  --output records.embedded.jsonl
```

Запись JSONL может выглядеть так:

```json
{"id":"fact-1","text":"Окно отгрузки: четверг утром.","embedding":null,"embedding_metadata":null}
```

В Python:

```python
from embedding_contract import OllamaProvider, backfill

provider = OllamaProvider(
    model="embeddinggemma-2:270m",
    dimensions=768,
    base_url="http://127.0.0.1:11434",
    prompt_contract={
        "query_prefix": "task: search result | query: ",
        "document_prefix": "title: none | text: ",
    },
)

summary = backfill(
    records,
    provider,
    get_text=lambda record, index: record["body"],
    get_embedding=lambda record, index: record.get("vector"),
    get_metadata=lambda record, index: record.get("vector_metadata"),
    set_embedding=lambda record, vector, index: record.__setitem__("vector", vector),
    set_metadata=lambda record, metadata, index: record.__setitem__("vector_metadata", metadata),
)
```

Валидный существующий вектор сохраняется только при точном совпадении метаданных с текущей идентичностью. Отсутствующие, невалидные и устаревшие векторы заменяются ограниченными батчами. База данных этому пакету не принадлежит.

## Идентичность

Идентичность связывает:

- провайдера и модель;
- размерность;
- prompt contract (префиксы query/document или пару task types);
- опциональный digest артефакта провайдера (для локального Ollama).

Digest — SHA-256 по каноническому JSON с сортированными ключами. `EmbeddingIdentity.metadata()` возвращает точные поля для сохранения с вектором. `identity_from_metadata()` проверяет digest до использования.

Изменение любого поля меняет идентичность. Это намеренно: новая идентичность означает переиндексацию, а не смешивание.

## Чего пакет не делает

- Это не векторная база.
- Он не ищет и не ранжирует; для этого есть `mempalace-retrieval`.
- Он не ведёт историю фактов; для этого есть `memory-ledger`.
- Он не поднимает HTTP-сервер.
- Он не переключается на другого провайдера, endpoint или облако.
- Он не хранит и не печатает API-ключи.

## Безопасность и приватность

- Локальный Ollama по умолчанию только loopback: `http://127.0.0.1:<port>`, без credentials, path и redirects.
- Gemini использует HTTPS для любого non-loopback origin. Loopback HTTP допускается только для локальных тестов и отправляет ключ только на `127.0.0.1`. Ключ читается из окружения, не из CLI-флага, и не попадает в ошибки или `repr`.
- Gemini отправляет текст в Google. Используйте его только когда такой egress допустим. Локальная Gemma никуда текст не отправляет. Оба адаптера игнорируют HTTP-прокси из окружения и не следуют редиректам. Gemini-адаптер покрыт fake-wire тестами; перед рабочим backfill запустите `health` со своим ключом.
- Ответы провайдера ограничены по размеру, проверяются по форме, размерности, конечности, нулю и L2-нормализуются.
- Тихая обрезка запрещена; несовпадение размерности отклоняется.

См. [SECURITY.md](SECURITY.md) и [docs/identity.md](docs/identity.md).

## Тесты

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Тесты используют fake HTTP-серверы и не требуют модели, сети, API-ключа или личной базы. Они проверяют identity digests, размерности и нормализацию, backfill stale/invalid, идентичность провайдера, префиксы/task types, redaction ключа и отсутствие fallback.

## Семейство проектов

- [MemPalace Retrieval](https://github.com/alexandryasukaitisss-cmd/mempalace-retrieval) — поиск с проверкой доступа и отказ при слабых совпадениях.
- [Memory Ledger](https://github.com/alexandryasukaitisss-cmd/memory-ledger) — append-only история фактов и supersession.
- **Embedding Contract** — идентичность векторов и безопасный backfill (этот репозиторий).

## Лицензия

MIT. См. [LICENSE](LICENSE).

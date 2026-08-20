# AI Engineering Agent

Agente hotelero autonomo construido con LangGraph. El modelo decide cuando utilizar una
herramienta y el grafo vuelve al modelo despues de cada resultado. Los errores esperados de
las herramientas tambien vuelven al modelo para que pueda corregir la llamada o pedir una
aclaracion.

## Requisitos

- Python 3.12
- `uv`
- Una API compatible con tool calling de OpenAI

## Instalacion

```bash
uv sync --all-groups
cp .env.example .env
```

Configura al menos `OPENAI_API_KEY` en `.env`. Para proveedores compatibles puedes definir
tambien `OPENAI_BASE_URL` y `OPENAI_MODEL`.

## Uso

Una sola consulta:

```bash
uv run hotel-agent --thread-id demo "Busca hoteles en Metropolis"
```

Sesion interactiva:

```bash
uv run hotel-agent --thread-id demo
```

El mismo `thread_id` recupera la conversacion desde la base SQLite indicada por
`AGENT_DB_PATH`, incluso despues de reiniciar el proceso. Threads diferentes permanecen
aislados. El agente limita cada ejecucion a 10 pasos de razonamiento.

## Arquitectura

- Las funciones `@tool` tienen contratos descriptivos y reciben el cliente mediante
  `ToolRuntime`; esa dependencia no se expone al modelo.
- `ToolNode` ejecuta herramientas y `tools_condition` decide entre ejecutar herramientas o
  finalizar, sin rutas `if/else` por intencion.
- `AsyncSqliteSaver` persiste el estado acumulado por `thread_id`.
- Injector construye el modelo, el cliente y la fabrica del agente.
- Typer ofrece el CLI y `asyncio` ejecuta el flujo completo.
- Los errores de dominio se devuelven como `ToolMessage`; los errores inesperados se
  conservan y se manejan en el limite de la aplicacion.

## Calidad

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

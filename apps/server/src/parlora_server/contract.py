"""The OpenAPI contract exactly as committed to `contracts/openapi.json` (design D8).

The committed file is the integration surface: the web client's types and any
integrator's client are generated from it, so the export and the drift test
share this one rendering.

release-please bumps `info.version` by re-serialising the file with
JavaScript's JSON, which lists integer-like keys (response status codes) first
in ascending order and writes integral numbers without `.0`. The document is
rendered in that form already, so the release pull request changes only the
version and the drift test still passes on it.
"""

import json
from typing import Any

from parlora_server.config import Settings
from parlora_server.main import create_app

_MAX_ARRAY_INDEX = 2**32 - 2


def render_openapi() -> str:
    # Routes do not depend on configuration, so defaults render the same document.
    app = create_app(Settings(_env_file=None))
    return json.dumps(_js_stable(app.openapi()), indent=2, ensure_ascii=False) + "\n"


def _js_stable(value: Any) -> Any:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_js_stable(item) for item in value]
    if isinstance(value, dict):
        index_keys = sorted((key for key in value if _is_array_index(key)), key=int)
        ordered = [*index_keys, *(key for key in value if not _is_array_index(key))]
        return {key: _js_stable(value[key]) for key in ordered}
    return value


def _is_array_index(key: object) -> bool:
    return (
        isinstance(key, str)
        and key.isdigit()
        and str(int(key)) == key
        and int(key) <= _MAX_ARRAY_INDEX
    )

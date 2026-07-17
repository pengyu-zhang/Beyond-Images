"""HTTP helpers: shared session with retries, timeouts, and a polite user agent."""

from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = (
    "BeyondImages/1.0 (research pipeline; "
    "+https://github.com/pengyu-zhang/Beyond-Images)"
)


def make_session(
    retries: int = 3,
    backoff: float = 1.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers["User-Agent"] = user_agent
    return session

"""Small HTTP helpers shared by adapters."""

from __future__ import annotations

import urllib.request
from urllib.request import ProxyHandler


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: a provider must not move text or a key to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)
# Provider traffic must not inherit HTTP(S)_PROXY from the environment.
DIRECT_NO_REDIRECT_OPENER = urllib.request.build_opener(ProxyHandler({}), _NoRedirect)

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from fastapi import HTTPException, status

from .config import get_settings
from .security import create_oauth_state, decode_oauth_state


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


def build_google_login_url() -> str:
    settings = get_settings()
    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google login is not configured.",
        )

    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": create_oauth_state(),
        "access_type": "offline",
        "prompt": "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def exchange_google_code(code: str) -> dict:
    settings = get_settings()
    payload = urlencode(
        {
            "code": code,
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "redirect_uri": settings.google_redirect_uri,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    request = Request(
        GOOGLE_TOKEN_URL,
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_google_identity(id_token: str) -> dict:
    settings = get_settings()
    request_url = f"{GOOGLE_TOKENINFO_URL}?{urlencode({'id_token': id_token})}"

    with urlopen(request_url, timeout=15) as response:
        identity = json.loads(response.read().decode("utf-8"))

    if identity.get("aud") != settings.google_client_id:
        raise ValueError("Google token audience does not match this application.")

    if identity.get("email_verified") not in {True, "true", "True"}:
        raise ValueError("Google did not verify this email address.")

    if not identity.get("email") or not identity.get("sub"):
        raise ValueError("Google identity response is missing required fields.")

    return identity


def validate_google_oauth_state(state: str) -> None:
    decode_oauth_state(state)

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.core.google_oauth import verify_google_identity


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_google_verified_email_is_accepted():
    payload = {
        "aud": "google-client-id",
        "email": "person@gmail.com",
        "email_verified": "true",
        "sub": "google-user-sub",
    }

    with (
        patch(
            "app.core.google_oauth.get_settings",
            return_value=SimpleNamespace(google_client_id="google-client-id"),
        ),
        patch("app.core.google_oauth.urlopen", return_value=FakeResponse(payload)),
    ):
        identity = verify_google_identity("id-token")

    assert identity["email"] == "person@gmail.com"


def test_google_unverified_email_is_rejected():
    payload = {
        "aud": "google-client-id",
        "email": "person@gmail.com",
        "email_verified": "false",
        "sub": "google-user-sub",
    }

    with (
        patch(
            "app.core.google_oauth.get_settings",
            return_value=SimpleNamespace(google_client_id="google-client-id"),
        ),
        patch("app.core.google_oauth.urlopen", return_value=FakeResponse(payload)),
    ):
        with pytest.raises(ValueError, match="verify"):
            verify_google_identity("id-token")

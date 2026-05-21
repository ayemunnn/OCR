from app.core.config import Settings


def test_cors_origins_default_to_localhost_values():
    settings = Settings(_env_file=None)

    assert "http://localhost:5173" in settings.backend_cors_origins
    assert "http://127.0.0.1:5173" in settings.backend_cors_origins


def test_cors_origins_accept_single_url_string():
    settings = Settings(
        _env_file=None,
        backend_cors_origins="https://example.azurestaticapps.net",
    )

    assert settings.backend_cors_origins == ["https://example.azurestaticapps.net"]


def test_cors_origins_accept_comma_separated_string():
    settings = Settings(
        _env_file=None,
        backend_cors_origins=(
            "https://example.azurestaticapps.net,http://localhost:5173"
        ),
    )

    assert settings.backend_cors_origins == [
        "https://example.azurestaticapps.net",
        "http://localhost:5173",
    ]


def test_cors_origins_accept_json_list_string():
    settings = Settings(
        _env_file=None,
        backend_cors_origins=(
            '["https://example.azurestaticapps.net","http://localhost:5173"]'
        ),
    )

    assert settings.backend_cors_origins == [
        "https://example.azurestaticapps.net",
        "http://localhost:5173",
    ]

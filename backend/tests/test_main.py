from app.main import health_check, root


def test_root_route():
    assert root() == {"message": "PaperSleuth API is running"}


def test_health_route():
    assert health_check() == {"status": "healthy"}

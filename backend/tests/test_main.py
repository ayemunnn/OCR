from app.main import health_check, root
from app.api.documents import MAX_UPLOAD_SIZE_BYTES


def test_root_route():
    assert root() == {"message": "PaperSleuth API is running"}


def test_health_route():
    assert health_check() == {"status": "healthy"}


def test_document_upload_limit_is_one_mb():
    assert MAX_UPLOAD_SIZE_BYTES == 1 * 1024 * 1024

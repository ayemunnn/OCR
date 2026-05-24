import pytest

from app.core.email_validation import validate_legitimate_email


def test_valid_email_is_accepted():
    assert validate_legitimate_email("User@Gmail.com") == "user@gmail.com"


def test_invalid_email_is_rejected():
    with pytest.raises(ValueError, match="valid email"):
        validate_legitimate_email("not-an-email")


def test_placeholder_email_domain_is_rejected():
    with pytest.raises(ValueError, match="placeholder"):
        validate_legitimate_email("person@example.com")


def test_disposable_email_domain_is_rejected():
    with pytest.raises(ValueError, match="Disposable"):
        validate_legitimate_email("person@mailinator.com")

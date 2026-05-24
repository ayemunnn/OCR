from email_validator import EmailNotValidError, validate_email


PLACEHOLDER_EMAIL_DOMAINS = {
    "example.com",
    "test.com",
    "fake.com",
    "invalid.com",
    "localhost",
}

DISPOSABLE_EMAIL_DOMAINS = {
    "10minutemail.com",
    "guerrillamail.com",
    "mailinator.com",
    "sharklasers.com",
    "temp-mail.org",
    "tempmail.com",
    "throwawaymail.com",
    "yopmail.com",
}


def validate_legitimate_email(email: str) -> str:
    """Validate format and block common placeholder/disposable email domains."""
    try:
        result = validate_email(email, check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError("Enter a valid email address.") from exc

    normalized_email = result.normalized.lower()
    domain = normalized_email.rsplit("@", 1)[-1]

    if domain in PLACEHOLDER_EMAIL_DOMAINS:
        raise ValueError("Please use a real email address, not a placeholder domain.")

    if domain in DISPOSABLE_EMAIL_DOMAINS:
        raise ValueError("Disposable email addresses are not allowed.")

    return normalized_email

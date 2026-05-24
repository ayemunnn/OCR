import secrets
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.email_validation import validate_legitimate_email
from ..core.google_oauth import (
    build_google_login_url,
    exchange_google_code,
    validate_google_oauth_state,
    verify_google_identity,
)
from ..core.security import create_access_token, decode_access_token
from ..core.security import hash_password, verify_password
from ..db.session import get_db
from ..models.user import User
from ..schemas.user import Token, UserCreate, UserLogin, UserResponse

router = APIRouter(prefix="/auth", tags=["auth"])
bearer_scheme = HTTPBearer()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email).first()


def get_user_by_google_sub(db: Session, google_sub: str) -> User | None:
    return db.query(User).filter(User.google_sub == google_sub).first()


def build_frontend_redirect(params: dict[str, str], use_fragment: bool = False) -> RedirectResponse:
    settings = get_settings()
    separator = "#" if use_fragment else ("&" if "?" in settings.frontend_url else "?")
    return RedirectResponse(f"{settings.frontend_url}{separator}{urlencode(params)}")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(credentials.credentials)
    except ValueError as exc:
        raise credentials_exception from exc

    email = payload.get("sub")
    if not email:
        raise credentials_exception

    user = get_user_by_email(db, email)
    if not user or not user.is_active:
        raise credentials_exception

    return user


@router.post("/signup", response_model=UserResponse)
def signup(user_create: UserCreate, db: Session = Depends(get_db)):
    try:
        email = validate_legitimate_email(str(user_create.email))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    existing_user = get_user_by_email(db, email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists.",
        )

    user = User(
        email=email,
        hashed_password=hash_password(user_create.password),
        full_name=user_create.full_name,
        auth_provider="email",
        email_verified=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.post("/login", response_model=Token)
def login(user_login: UserLogin, db: Session = Depends(get_db)):
    try:
        email = validate_legitimate_email(str(user_login.email))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    user = get_user_by_email(db, email)
    if not user or not verify_password(user_login.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token({"sub": user.email})
    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/google/login")
def google_login():
    # Starts Google OAuth without exposing the client secret to the frontend.
    return RedirectResponse(build_google_login_url())


@router.get("/google/callback")
def google_callback(
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    if error:
        return build_frontend_redirect({"auth_error": f"Google login failed: {error}"})

    if not code or not state:
        return build_frontend_redirect({"auth_error": "Google login was cancelled."})

    try:
        validate_google_oauth_state(state)
        token_response = exchange_google_code(code)
        identity = verify_google_identity(token_response["id_token"])
        email = validate_legitimate_email(identity["email"])
    except (HTTPError, URLError, KeyError, ValueError) as exc:
        return build_frontend_redirect({"auth_error": str(exc)})

    google_sub = identity["sub"]
    user = get_user_by_google_sub(db, google_sub) or get_user_by_email(db, email)

    if user:
        if not user.google_sub:
            user.google_sub = google_sub
        user.email_verified = True
        db.commit()
        db.refresh(user)
    else:
        user = User(
            email=email,
            # Google users authenticate with OAuth, but the existing schema requires
            # a password hash. This random hash is never shown or used for OAuth.
            hashed_password=hash_password(secrets.token_urlsafe(32)),
            full_name=identity.get("name"),
            auth_provider="google",
            google_sub=google_sub,
            email_verified=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    access_token = create_access_token({"sub": user.email})
    return build_frontend_redirect(
        {
            "access_token": access_token,
            "token_type": "bearer",
        },
        use_fragment=True,
    )


@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: User = Depends(get_current_user)):
    return current_user

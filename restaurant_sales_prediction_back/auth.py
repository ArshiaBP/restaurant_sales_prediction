"""
auth.py
=======
Simple JWT authentication — username, password, optional restaurant name.

Endpoints:
  POST /auth/signup   — create account {username, password, restaurant_name?}
  POST /auth/login    — {username, password} → returns JWT token
  GET  /auth/me       — current user profile (requires token)

After login, pass the token in every protected request:
  Authorization: Bearer <your_token>
"""
import os
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
import bcrypt
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import User, get_db

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Change SECRET_KEY before deploying to production.
# Generate a strong one with: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY           = os.environ["SECRET_KEY"]
ALGORITHM            = "HS256"
TOKEN_EXPIRE_MINUTES = 60 * 24   # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router        = APIRouter(prefix="/auth", tags=["Auth"])


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username        : str
    password        : str
    restaurant_name : Optional[str] = None   # optional

class UserResponse(BaseModel):
    id              : int
    username        : str
    restaurant_name : Optional[str]
    created_at      : datetime

    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token : str
    token_type   : str = "bearer"
    expires_in   : int   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Password helpers
# ─────────────────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ─────────────────────────────────────────────────────────────────────────────
# JWT helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_access_token(username: str) -> str:
    expire  = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[str]:
    """Returns username from token, or None if invalid / expired."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dependency — resolves current user from Bearer token
# ─────────────────────────────────────────────────────────────────────────────

def get_current_user(
    token: str     = Depends(oauth2_scheme),
    db   : Session = Depends(get_db),
) -> User:
    username = decode_token(token)
    if not username:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Invalid or expired token",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "User not found",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    return user


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    """
    Create a new account and immediately return a JWT token.
    Fields: username (required), password (required), restaurant_name (optional).
    """
    if len(body.username.strip()) < 3:
        raise HTTPException(400, "Username must be at least 3 characters")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(400, f"Username '{body.username}' is already taken")

    user = User(
        username        = body.username.strip(),
        restaurant_name = body.restaurant_name,
        hashed_pw       = hash_password(body.password),
    )
    db.add(user)
    db.commit()

    token = create_access_token(user.username)
    return TokenResponse(
        access_token = token,
        expires_in   = TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db  : Session = Depends(get_db),
):
    """
    Login with username + password → returns a JWT token valid for 24 hours.

    Use the token in subsequent requests:
      Authorization: Bearer <access_token>
    """
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "Incorrect username or password",
            headers     = {"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(user.username)
    return TokenResponse(
        access_token = token,
        expires_in   = TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get("/me", response_model=UserResponse)
def me(current_user: User = Depends(get_current_user)):
    """Returns the profile of the currently authenticated user."""
    return current_user
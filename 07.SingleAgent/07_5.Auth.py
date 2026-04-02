import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

JWT_SECRET       = os.getenv("JWT_SECRET", os.getenv("BACKEND_API_KEY", ""))
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
AUTH_USERNAME    = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD_HASH = os.getenv("AUTH_PASSWORD_HASH", "")

_bearer = HTTPBearer()


def create_token(username: str) -> str:
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")


def authenticate(username: str, password: str) -> bool:
    if not AUTH_PASSWORD_HASH:
        raise HTTPException(status_code=500, detail="서버에 AUTH_PASSWORD_HASH가 설정되지 않았습니다.")
    return username == AUTH_USERNAME and bcrypt.checkpw(password.encode(), AUTH_PASSWORD_HASH.encode())

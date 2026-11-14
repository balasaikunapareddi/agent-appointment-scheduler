"""Field-level encryption for patient records (Fernet, AES-128 + HMAC).

The encryption key comes from the environment (``SCHEDULER_ENCRYPTION_KEY``).
It is never committed — see ``.env.example`` for how to generate one. When
no key is set, an ephemeral key is generated for the process and a warning
is logged: records encrypted with it do not survive a restart. That keeps
the demo runnable without setup while making the production rule obvious.
"""

from __future__ import annotations

import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENV_VAR = "SCHEDULER_ENCRYPTION_KEY"
_ephemeral: Fernet | None = None


def get_fernet() -> Fernet:
    """Return the Fernet instance for this process."""
    global _ephemeral
    raw = os.environ.get(_ENV_VAR, "").strip()
    if raw:
        return Fernet(raw.encode())
    if _ephemeral is None:
        _ephemeral = Fernet(Fernet.generate_key())
        logger.warning(
            "%s is not set — using an ephemeral encryption key. "
            "Patient records will not survive a restart.",
            _ENV_VAR,
        )
    return _ephemeral


def encrypt_field(value: str) -> str:
    """Encrypt one field value; returns a URL-safe token string."""
    return get_fernet().encrypt((value or "").encode()).decode()


def decrypt_field(token: str) -> str:
    """Decrypt one field value; raises ValueError on a bad token."""
    try:
        return get_fernet().decrypt(token.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Could not decrypt field value.") from exc

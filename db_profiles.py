"""
Encrypted database connection profile storage with bulk import.

Profiles are organized by App → Env → DB and stored in an AES-encrypted file
(Fernet, backed by PBKDF2-derived key from the application password).

Bulk loading:
    Place a ``db_profiles_seed.json`` file next to this module with all your
    connection details.  On first startup (or via the /api/profiles/import
    endpoint) the seed file is encrypted into ``db_profiles.enc`` and the
    plaintext seed is deleted automatically.

Seed file format (array of profile objects):
    [
      {
        "app_name": "IDP
        "env_name": "Production",
        "db_name": "p1cy6d48",
        "database_type": "oracle",
        "host": "zlpy21815.vci.att.com",
        "port": 1524,
        "database": "p1cy6d48",
        "user": "SHCAT1",
        "password": "...",
        "use_sid": true
      },
      ...
    ]
"""

import json
import os
import uuid
import base64
import logging
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

logger = logging.getLogger(__name__)

# Paths next to this module
_MODULE_DIR = Path(__file__).parent
_DEFAULT_PROFILES_PATH = _MODULE_DIR / "db_profiles.enc"
_DEFAULT_SEED_PATH = _MODULE_DIR / "db_profiles_seed.json"

# Fixed salt — unique to this application.
_KDF_SALT = b"AI_DBA_Assistant_ProfileSalt_v1"


def _derive_key(password: str) -> bytes:
    """Derive a Fernet-compatible key from *password* via PBKDF2-HMAC-SHA256."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_KDF_SALT,
        iterations=480_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))


class ProfileStore:
    """Thread-safe, Fernet-encrypted profile storage with bulk import."""

    _SAFE_FIELDS = {
        "id", "app_name", "env_name", "db_name",
        "database_type", "host", "port", "database", "user",
        "use_sid", "pdb_name",
        "use_ssh_tunnel", "ssh_jump_host", "ssh_jump_user",
        "ssh_jump_port", "ssh_remote_host", "ssh_remote_port",
        "ssh_local_port",
    }

    def __init__(self, master_password: str, path: Optional[str] = None,
                 seed_path: Optional[str] = None):
        self._fernet = Fernet(_derive_key(master_password))
        self._path = Path(path) if path else _DEFAULT_PROFILES_PATH
        self._seed_path = Path(seed_path) if seed_path else _DEFAULT_SEED_PATH
        self._lock = Lock()
        # Auto-import seed on first init if seed file exists
        self._auto_import_seed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict[str, Any]]:
        """Decrypt and return the profiles list, or [] if file missing/empty."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return []
        try:
            cipher_text = self._path.read_bytes()
            plain = self._fernet.decrypt(cipher_text)
            return json.loads(plain)
        except (InvalidToken, json.JSONDecodeError) as exc:
            logger.error("Failed to decrypt profiles file: %s", exc)
            return []

    def _save(self, profiles: List[Dict[str, Any]]) -> None:
        """Encrypt and persist the profiles list."""
        plain = json.dumps(profiles, indent=2).encode("utf-8")
        cipher_text = self._fernet.encrypt(plain)
        self._path.write_bytes(cipher_text)

    @staticmethod
    def _sanitize(profile: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of *profile* with the password stripped."""
        return {k: v for k, v in profile.items() if k != "password"}

    def _auto_import_seed(self) -> None:
        """If a seed JSON file exists, import it and delete the plaintext."""
        if not self._seed_path.exists():
            return
        try:
            imported = self.import_from_file(self._seed_path)
            # Delete the plaintext seed after successful import
            self._seed_path.unlink()
            logger.info(
                "Auto-imported %d profiles from %s and deleted plaintext seed",
                imported, self._seed_path.name,
            )
        except Exception as exc:
            logger.error("Failed to auto-import seed file: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_profiles(self) -> List[Dict[str, Any]]:
        """Return all profiles *without* passwords (safe for UI)."""
        with self._lock:
            return [self._sanitize(p) for p in self._load()]

    def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Return a single profile *with* the password, or None."""
        with self._lock:
            for p in self._load():
                if p.get("id") == profile_id:
                    return dict(p)
        return None

    def save_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Add or update a profile.  Returns the saved profile (no password)."""
        with self._lock:
            profiles = self._load()

            if not profile.get("id"):
                profile["id"] = str(uuid.uuid4())

            for field in ("app_name", "env_name", "db_name", "database_type",
                          "host", "database", "user", "password"):
                if not profile.get(field):
                    raise ValueError(f"Missing required profile field: {field}")

            # Upsert by id
            profiles = [p for p in profiles if p["id"] != profile["id"]]
            profiles.append(profile)
            self._save(profiles)
            return self._sanitize(profile)

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a profile by id. Returns True if found & deleted."""
        with self._lock:
            profiles = self._load()
            before = len(profiles)
            profiles = [p for p in profiles if p["id"] != profile_id]
            if len(profiles) == before:
                return False
            self._save(profiles)
            return True

    def import_from_file(self, file_path: Path) -> int:
        """Import profiles from a plaintext JSON file (array of objects).

        Merges by (app_name, env_name, db_name) — existing matches are
        overwritten with the new data.  Returns the count of imported profiles.
        """
        raw = file_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Seed file must contain a JSON array of profile objects")

        with self._lock:
            profiles = self._load()
            # Build lookup by natural key
            existing = {}
            for p in profiles:
                key = (p.get("app_name", ""), p.get("env_name", ""), p.get("db_name", ""))
                existing[key] = p

            count = 0
            for entry in data:
                for field in ("app_name", "env_name", "db_name", "database_type",
                              "host", "database", "user", "password"):
                    if not entry.get(field):
                        raise ValueError(
                            f"Seed entry missing required field '{field}': {entry}"
                        )
                key = (entry["app_name"], entry["env_name"], entry["db_name"])
                if key in existing:
                    # Preserve the id from the existing profile
                    entry["id"] = existing[key]["id"]
                if not entry.get("id"):
                    entry["id"] = str(uuid.uuid4())
                existing[key] = entry
                count += 1

            self._save(list(existing.values()))
            return count

    def import_from_list(self, entries: List[Dict[str, Any]]) -> int:
        """Import profiles from an in-memory list (e.g. from an API call).

        Same merge logic as import_from_file.  Returns count imported.
        """
        with self._lock:
            profiles = self._load()
            existing = {}
            for p in profiles:
                key = (p.get("app_name", ""), p.get("env_name", ""), p.get("db_name", ""))
                existing[key] = p

            count = 0
            for entry in entries:
                for field in ("app_name", "env_name", "db_name", "database_type",
                              "host", "database", "user", "password"):
                    if not entry.get(field):
                        raise ValueError(
                            f"Entry missing required field '{field}': {entry}"
                        )
                key = (entry["app_name"], entry["env_name"], entry["db_name"])
                if key in existing:
                    entry["id"] = existing[key]["id"]
                if not entry.get("id"):
                    entry["id"] = str(uuid.uuid4())
                existing[key] = entry
                count += 1

            self._save(list(existing.values()))
            return count

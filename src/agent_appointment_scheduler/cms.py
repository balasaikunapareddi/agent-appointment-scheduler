"""CMS NPI Registry client (https://npiregistry.cms.hhs.gov/api/).

Free, no key. Rules learned from the live API and encoded here:
- ``state`` alone is rejected (error 07: "Field state requires additional
  search criteria") — always combine it with city, taxonomy, or a name.
- CMS v2.1 has NO ``taxonomy_code`` search field: sending it is silently
  ignored and the registry returns an unfiltered list (verified live
  2026-11-14 — a "cardiology" search returned nurses, social workers and
  a chiropractor). Specialty search uses ``taxonomy_description`` instead
  (e.g. "Cardiovascular Disease"). Taxonomy CODES are still kept for the
  local seed fallback, which filters on them.
- Responses may carry an ``Errors`` array instead of ``results``.
- Retries with backoff, a hard timeout, and a 24h local cache. When CMS
  is unreachable or returns an error, fall back to local seed providers
  so the demo never dies — seed results are marked ``seed: true``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from importlib import resources

import httpx

from . import store

logger = logging.getLogger(__name__)

BASE_URL = "https://npiregistry.cms.hhs.gov/api/"
VERSION = "2.1"
TIMEOUT_SECONDS = 10.0
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (0.5, 1.0, 2.0)

# Specialty name -> taxonomy code (NUCC). The display name is what the user
# says; the code is what the local seed fallback filters on.
SPECIALTIES: dict[str, str] = {
    "cardiology": "207RC0000X",
    "cardiovascular disease": "207RC0000X",
    "cardiologist": "207RC0000X",
    "family medicine": "207Q00000X",
    "family practice": "207Q00000X",
    "internal medicine": "207R00000X",
    "internist": "207R00000X",
    "obstetrics & gynecology": "207V00000X",
    "ob/gyn": "207V00000X",
    "obstetrics": "207V00000X",
    "gynecology": "207V00000X",
}

# Taxonomy code -> the description CMS v2.1 actually searches on
# (``taxonomy_description``). Every value here was verified live against
# the registry 2026-11-14: each returns providers whose taxonomy matches.
TAXONOMY_DESCRIPTIONS: dict[str, str] = {
    "207RC0000X": "Cardiovascular Disease",
    "207Q00000X": "Family Medicine",
    "207R00000X": "Internal Medicine",
    "207V00000X": "Obstetrics & Gynecology",
}

DEFAULT_CITY = "Minneapolis"
DEFAULT_STATE = "MN"


def taxonomy_code_for(specialty: str) -> str | None:
    """Resolve a spoken specialty to its taxonomy code, or None.

    Accepts a name ("cardiology") or a code passed through as the
    specialty ("207RC0000X") — the agent tried the code form live and
    got a 400, so both spellings resolve here.
    """
    if not specialty:
        return None
    cleaned = specialty.strip()
    if cleaned.upper() in SPECIALTIES.values():
        return cleaned.upper()
    return SPECIALTIES.get(cleaned.lower())


def taxonomy_description_for(taxonomy_code: str | None) -> str | None:
    """Resolve a taxonomy code to the description CMS searches on.

    CMS v2.1 has no ``taxonomy_code`` search field, so the code itself
    is never sent to the registry — this description is, via
    ``taxonomy_description``. Returns None for an unknown code; callers
    then search without a specialty filter (city/state still apply).
    """
    if not taxonomy_code:
        return None
    return TAXONOMY_DESCRIPTIONS.get(taxonomy_code.strip().upper())


def _load_seed_providers() -> list[dict]:
    with (resources.files("agent_appointment_scheduler") / "data" / "seed_providers.json").open() as f:
        return json.load(f)["providers"]


def _cache_key(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _normalize(record: dict) -> dict:
    """Shrink a raw CMS record to the fields the agent needs."""
    basic = record.get("basic", {}) or {}
    taxonomies = record.get("taxonomies", []) or []
    primary = next((t for t in taxonomies if t.get("primary")), taxonomies[0] if taxonomies else {})
    addresses = record.get("addresses", []) or []
    location = next(
        (a for a in addresses if a.get("address_purpose") == "LOCATION"),
        addresses[0] if addresses else {},
    )
    name = " ".join(
        part for part in [basic.get("first_name", ""), basic.get("last_name", "")]
        if part
    ).strip()
    if record.get("enumeration_type") == "NPI-2":
        name = basic.get("organization_name", "") or name
    return {
        "npi": record.get("number", ""),
        "name": name.title() if name else "Unknown",
        "credential": basic.get("credential", ""),
        "specialty": primary.get("desc", ""),
        "taxonomy_code": primary.get("code", ""),
        "organization": "",
        "address": location.get("address_1", ""),
        "city": location.get("city", ""),
        "state": location.get("state", ""),
        "zip": location.get("postal_code", ""),
        "phone": location.get("telephone_number", ""),
        "seed": False,
    }


class CMSClient:
    """Search the NPI Registry with retries, caching and seed fallback."""

    def __init__(
        self,
        db_path: str = "scheduler.db",
        transport: httpx.BaseTransport | None = None,
        enabled: bool = True,
    ) -> None:
        self._db_path = db_path
        self._transport = transport
        self._enabled = enabled

    def _cached(self, key: str) -> list[dict] | None:
        conn = store.connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT payload, fetched_at FROM providers_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row and time.time() - row["fetched_at"] < store.PROVIDER_CACHE_TTL_SECONDS:
                return json.loads(row["payload"])
            return None
        finally:
            conn.close()

    def _store_cache(self, key: str, providers: list[dict]) -> None:
        conn = store.connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO providers_cache "
                "(cache_key, payload, fetched_at) VALUES (?, ?, ?)",
                (key, json.dumps(providers), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def _seed_fallback(
        self, taxonomy_code: str | None, city: str | None, state: str | None, limit: int
    ) -> list[dict]:
        providers = _load_seed_providers()
        out = []
        for provider in providers:
            if taxonomy_code and provider["taxonomy_code"] != taxonomy_code:
                continue
            if city and provider["city"].lower() != city.lower():
                continue
            if state and provider["state"].lower() != state.lower():
                continue
            # Seed entries store first/last name; the API shape needs a
            # single display name, same as _normalize builds for CMS rows.
            entry = dict(provider)
            entry["name"] = " ".join(
                part for part in [provider.get("first_name", ""), provider.get("last_name", "")]
                if part
            ).strip()
            out.append(entry)
            if len(out) >= limit:
                break
        return out

    def search(
        self,
        taxonomy_code: str | None = None,
        city: str | None = None,
        state: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        limit: int = 10,
    ) -> dict:
        """Search providers. Returns ``{"providers": [...], "source": ...}``.

        ``source`` is ``cms`` for a live result, ``cache`` for a cached one
        and ``seed`` when CMS was unreachable or errored. State alone is
        never sent to CMS (it would be rejected); callers must combine it
        with city, taxonomy or a name.
        """
        params: dict[str, str | int] = {"version": VERSION, "limit": limit}
        # Appointments are booked with a person, not an organization:
        # ask CMS for individual providers (NPI-1) only. Without this the
        # registry returns organizational listings (NPI-2) too, and the
        # live demo booked "1 Steptransitionsservices" — an org, not a
        # cardiologist. Results are also filtered below in case a record
        # is explicitly marked NPI-2 (e.g. from an older cache).
        params["enumeration_type"] = "NPI-1"
        # CMS v2.1 silently ignores a taxonomy_code parameter, so search
        # by taxonomy_description instead (see module docstring). The code
        # is still kept for the seed fallback, which filters on it.
        taxonomy_description = taxonomy_description_for(taxonomy_code)
        if taxonomy_description:
            params["taxonomy_description"] = taxonomy_description
        if city:
            params["city"] = city
        if state:
            params["state"] = state
        if first_name:
            params["first_name"] = first_name
        if last_name:
            params["last_name"] = last_name
        if state and not (city or taxonomy_code or first_name or last_name):
            # CMS rejects state alone (error 07). Fall back to the seed list
            # for the state instead of sending a request CMS will refuse.
            logger.warning("CMS search with state alone is rejected; using seed fallback.")
            return {
                "providers": self._seed_fallback(taxonomy_code, city, state, limit),
                "source": "seed",
                "note": "State alone is not accepted by the CMS registry; seed providers shown.",
            }

        key = _cache_key(params)
        cached = self._cached(key)
        if cached is not None:
            return {"providers": cached, "source": "cache"}

        if not self._enabled:
            return {
                "providers": self._seed_fallback(taxonomy_code, city, state, limit),
                "source": "seed",
                "note": "CMS client disabled; seed providers shown.",
            }

        last_error: str | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with httpx.Client(
                    transport=self._transport, timeout=TIMEOUT_SECONDS
                ) as client:
                    response = client.get(BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
            except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
                # InvalidURL is NOT an HTTPError subclass (it derives from
                # plain Exception) — it surfaces when the sandbox proxy
                # config can't be parsed. Any transport failure falls back
                # to the seed list, the same as CMS being unreachable.
                last_error = str(exc)
                logger.warning("CMS request failed (attempt %d): %s", attempt + 1, exc)
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                continue
            errors = data.get("Errors") or []
            if errors:
                last_error = "; ".join(
                    f"{e.get('number')}: {e.get('description')}" for e in errors
                )
                logger.warning("CMS returned errors: %s", last_error)
                break
            providers = [
                _normalize(r)
                for r in data.get("results", [])
                if r.get("enumeration_type", "NPI-1") == "NPI-1"
            ]
            self._store_cache(key, providers)
            return {"providers": providers, "source": "cms"}

        return {
            "providers": self._seed_fallback(taxonomy_code, city, state, limit),
            "source": "seed",
            "note": f"CMS unavailable ({last_error}); seed providers shown.",
        }

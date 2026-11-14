"""CMS client tests with a mocked HTTP transport: errors, caching, seed fallback."""

import httpx

from agent_appointment_scheduler import cms


def _cms_payload():
    return {
        "result_count": 1,
        "results": [
            {
                "number": "1215596358",
                "basic": {"first_name": "Muhannad", "last_name": "Abbasi", "credential": "MD"},
                "taxonomies": [{"code": "207RC0000X", "desc": "Cardiovascular Disease"}],
                "addresses": [
                    {
                        "address_1": "100 Main St",
                        "city": "Minneapolis",
                        "state": "MN",
                        "postal_code": "55401",
                        "telephone_number": "612-555-0100",
                    }
                ],
            }
        ],
    }


def test_search_maps_taxonomy_and_normalizes(db_path):
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_cms_payload())

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    result = client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    assert result["source"] == "cms"
    providers = result["providers"]
    assert len(providers) == 1
    provider = providers[0]
    assert provider["npi"] == "1215596358"
    assert provider["name"] == "Muhannad Abbasi"
    assert provider["specialty"] == "Cardiovascular Disease"
    assert provider["city"] == "Minneapolis"
    # The taxonomy code, not the word, went to CMS.
    assert seen["params"]["taxonomy_code"] == "207RC0000X"


def test_taxonomy_code_for():
    assert cms.taxonomy_code_for("cardiology") == "207RC0000X"
    assert cms.taxonomy_code_for("Cardiologist") == "207RC0000X"
    assert cms.taxonomy_code_for("family medicine") == "207Q00000X"
    assert cms.taxonomy_code_for("not a specialty") is None
    # A code passed as the specialty resolves too (the agent tried this
    # live and got a 400 before the fix).
    assert cms.taxonomy_code_for("207RC0000X") == "207RC0000X"
    assert cms.taxonomy_code_for("207q00000x") == "207Q00000X"


def test_search_requests_individuals_only(db_path):
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        return httpx.Response(200, json=_cms_payload())

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    assert seen["params"]["enumeration_type"] == "NPI-1"


def test_organizations_dropped_from_results(db_path):
    payload = _cms_payload()
    payload["results"].append(
        {
            "number": "1831998038",
            "enumeration_type": "NPI-2",
            "basic": {"organization_name": "1 Steptransitionsservices"},
            "taxonomies": [{"code": "207RC0000X", "desc": "Cardiovascular Disease"}],
            "addresses": [
                {
                    "address_1": "1 Main St",
                    "city": "Minneapolis",
                    "state": "MN",
                    "postal_code": "55401",
                    "telephone_number": "612-555-0100",
                }
            ],
        }
    )
    payload["results"][0]["enumeration_type"] = "NPI-1"

    def handler(request):
        return httpx.Response(200, json=payload)

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    result = client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    assert result["source"] == "cms"
    names = [p["name"] for p in result["providers"]]
    assert names == ["Muhannad Abbasi"], "an organization must never be bookable"


def test_state_alone_never_hits_cms(db_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_cms_payload())

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    result = client.search(state="MN")
    assert calls == [], "state-only search must not call CMS"
    assert result["source"] == "seed"
    assert result["providers"] and result["providers"][0]["seed"] is True


def test_cms_errors_array_returns_seed(db_path):
    def handler(request):
        return httpx.Response(
            200,
            json={"Errors": [{"number": "07", "description": "Field state requires additional search criteria"}]},
        )

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    result = client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    assert result["source"] == "seed"
    assert result["providers"] and result["providers"][0]["seed"] is True


def test_results_cached_for_24h(db_path):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=_cms_payload())

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    assert len(calls) == 1, "second identical search must come from cache"


def test_unreachable_cms_falls_back_to_seed(db_path):
    def handler(request):
        raise httpx.ConnectError("unreachable", request=request)

    client = cms.CMSClient(db_path=db_path, transport=httpx.MockTransport(handler))
    result = client.search(taxonomy_code="207RC0000X", city="Minneapolis", state="MN")
    assert result["source"] == "seed"
    assert result["providers"] and all(p["seed"] for p in result["providers"])

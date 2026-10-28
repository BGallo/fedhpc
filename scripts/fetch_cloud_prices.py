"""Fetch on-demand Linux prices for the cloud equivalents used by the
generated instance classes (src/fedhpc/instance_gen.py) and save them to
data/cloud_prices.json.

Sources (both public, no credentials):
  AWS    — the EC2 on-demand price map that backs the AWS pricing pages,
           us-east-1 / Linux / shared tenancy:
           https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/
           ec2-ondemand-without-sec-sel/US%20East%20(N.%20Virginia)/Linux/index.json
           (gzip JSON; carries vCPU and memory next to the price).
  Azure  — the Azure Retail Prices API, eastus / Consumption, Linux meter
           (product names without "Windows", sku names without "Spot" /
           "Low Priority"): https://prices.azure.com/api/retail/prices
           The API carries no shape, so vCPU / memory come from AZURE_SHAPES
           below (Microsoft's published size tables).

The raw responses are kept next to the summary under data/cloud_prices_raw/
so the numbers can be re-audited without the network.

    uv run python scripts/fetch_cloud_prices.py
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedhpc.instance_gen import CLOUD_SKUS  # noqa: E402

OUT = ROOT / "data" / "cloud_prices.json"
RAW = ROOT / "data" / "cloud_prices_raw"

AWS_REGION = "US East (N. Virginia)"
AWS_URL = ("https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/"
           "ec2-ondemand-without-sec-sel/" + urllib.parse.quote(AWS_REGION) + "/Linux/index.json")
AZURE_REGION = "eastus"
AZURE_URL = "https://prices.azure.com/api/retail/prices"

# vCPU, memory (GiB) — learn.microsoft.com/azure/virtual-machines/sizes
AZURE_SHAPES = {
    "Standard_D32s_v5": (32, 128),
    "Standard_D64s_v5": (64, 256),
    "Standard_D96s_v5": (96, 384),
    "Standard_D128s_v6": (128, 512),
    "Standard_E64s_v5": (64, 512),
}


def _get(url: str, retries: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
            break
        except urllib.error.HTTPError as e:  # the Azure API rate-limits with 429
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 10 * 2 ** attempt))
    return gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body


def fetch_aws(skus: list[str]) -> dict[str, dict]:
    data = json.loads(_get(AWS_URL))
    rows = {v["Instance Type"]: v for v in data["regions"][AWS_REGION].values()}
    # the full map is ~0.7 MB; keep the manifest and the rows used
    (RAW / "aws_us-east-1_linux_ondemand.json").write_text(json.dumps(
        dict(manifest=data["manifest"], rows={s: rows[s] for s in skus}), indent=1))
    out = {}
    for sku in skus:
        v = rows[sku]
        out[sku] = dict(
            provider="aws", sku=sku, region="us-east-1",
            vcpu=int(v["vCPU"]), mem_gib=float(v["Memory"].split()[0]),
            usd_per_hour=float(v["price"]), rate_code=v["rateCode"],
        )
    return out


def fetch_azure(skus: list[str]) -> dict[str, dict]:
    names = " or ".join(f"armSkuName eq '{s}'" for s in skus)
    flt = (f"serviceName eq 'Virtual Machines' and armRegionName eq '{AZURE_REGION}' "
           f"and priceType eq 'Consumption' and ({names})")
    url = AZURE_URL + "?" + urllib.parse.urlencode({"$filter": flt})
    items = []
    while url:
        page = json.loads(_get(url))
        items += page["Items"]
        url = page.get("NextPageLink")
    (RAW / "azure_eastus_consumption.json").write_text(json.dumps(items, indent=1))
    out = {}
    for sku in skus:
        linux = [i for i in items
                 if i["armSkuName"] == sku
                 and "Windows" not in i["productName"]
                 and not any(w in i["skuName"] for w in ("Spot", "Low Priority"))
                 and i["unitOfMeasure"] == "1 Hour"]
        if len(linux) != 1:
            raise RuntimeError(f"{sku}: expected one Linux on-demand meter, got {len(linux)}")
        i = linux[0]
        vcpu, mem = AZURE_SHAPES[sku]
        out[sku] = dict(
            provider="azure", sku=sku, region=AZURE_REGION, vcpu=vcpu, mem_gib=float(mem),
            usd_per_hour=float(i["retailPrice"]), meter_id=i["meterId"],
            effective_start=i["effectiveStartDate"],
        )
    return out


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    by_provider: dict[str, list[str]] = {}
    for provider, sku in CLOUD_SKUS:
        by_provider.setdefault(provider, []).append(sku)
    prices = {**fetch_aws(by_provider["aws"]), **fetch_azure(by_provider["azure"])}
    doc = dict(
        _info=dict(
            description="On-demand Linux list prices (USD/hour) of the cloud types "
                        "used by fedhpc.instance_gen; written by scripts/fetch_cloud_prices.py.",
            fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            aws_source=AWS_URL, azure_source=AZURE_URL + f" (eastus, Consumption, Linux)",
        ),
        prices=dict(sorted(prices.items())),
    )
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    for p in doc["prices"].values():
        print(f"{p['provider']:5s} {p['sku']:20s} {p['vcpu']:4d} vCPU {p['mem_gib']:6.0f} GiB "
              f"${p['usd_per_hour']:.4f}/h")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

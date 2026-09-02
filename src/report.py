"""Auto-generated audit artifacts for every pipeline run."""
import json
import os
from datetime import datetime, timezone


def write_report(data: dict, out_dir: str = "reports") -> str:
    """Persist a machine-readable JSON report + human-readable Markdown
    receipt of a full pipeline run. Returns the JSON report path."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(out_dir, f"report_{stamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    md_path = os.path.join(out_dir, f"report_{stamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_markdown(data))

    return json_path


def _render_markdown(data: dict) -> str:
    lines = [
        "# Pipeline Verification Report",
        "",
        f"Generated: {data.get('generated_at', 'n/a')}",
        "",
        "## Network",
        f"- RPC: `{data.get('network', {}).get('rpc_url')}`",
        f"- Chain ID: `{data.get('network', {}).get('chain_id')}`",
        f"- Contract: `{data.get('network', {}).get('contract_address')}`",
        "",
        "## Stage 1 - Face Encoding",
        f"- Input image: `{data.get('stage1', {}).get('image')}`",
        f"- Bounding box: `{data.get('stage1', {}).get('bbox')}`",
        f"- Embedding dim: `{data.get('stage1', {}).get('embedding_dim')}`",
        f"- Biometric hash: `{data.get('stage1', {}).get('face_hash')}`",
        "",
        "## Stage 2 - Dynamic Web Discovery",
        f"- Hosted image: {data.get('stage2', {}).get('image_host_url')}",
        f"- Entity title: {data.get('stage2', {}).get('title')}",
        f"- Platform: {data.get('stage2', {}).get('platform')}",
        f"- Post URL: {data.get('stage2', {}).get('post_url')}",
        "",
        "## Stage 3 - On-Chain Anchoring",
        f"- Fingerprint: `{data.get('stage3', {}).get('fingerprint')}`",
        f"- Already anchored: `{data.get('stage3', {}).get('already_anchored', False)}`",
        f"- Tx hash: `{data.get('stage3', {}).get('tx_hash')}`",
        f"- Block: `{data.get('stage3', {}).get('block')}`",
        f"- Gas used: `{data.get('stage3', {}).get('gas_used')}`",
        "",
        "## Stage 4 - Verification & Tamper Drill",
        f"- Verification: **{data.get('stage4', {}).get('verification')}**",
        f"- On-chain fingerprint: `{data.get('stage4', {}).get('on_chain_fingerprint')}`",
        f"- Tamper detected: `{data.get('stage4', {}).get('tamper_detected')}`",
        f"- Tampered URL: {data.get('stage4', {}).get('tampered_url')}",
        "",
    ]
    return "\n".join(lines)

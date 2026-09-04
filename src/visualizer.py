"""ASCII-art + Rich visual helpers for the pipeline output."""

import os

from rich.align import Align
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


def render_pipeline_header() -> Panel:
    """Top-level pipeline header with ASCII branding."""
    banner = Text(
        "  ╔══════════════════════════════════════════════════╗\n"
        "  ║   FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION  ║\n"
        "  ╚══════════════════════════════════════════════════╝",
        style="bold cyan",
    )
    return Panel(banner, border_style="cyan", padding=(1, 2))


def render_stage_progress(stage: int) -> Text:
    """Show a 4-stage progress indicator."""
    stages = ["Face Detection", "Web Search", "Blockchain", "Verification"]
    parts = []
    for i, name in enumerate(stages, 1):
        if i < stage:
            parts.append(f"[green]✔ {name}[/green]")
        elif i == stage:
            parts.append(f"[bold reverse cyan] → {name} [/bold reverse cyan]")
        else:
            parts.append(f"[dim]  {name}[/dim]")
    return Text(" → ".join(parts))


def render_face_panel(crop_path: str, title: str = "Detected Face") -> Panel:
    """Show face crop info as a panel."""
    if os.path.exists(crop_path):
        size = os.path.getsize(crop_path)
        body = f"[cyan]{crop_path}[/cyan]\n[dim]size: {size:,} bytes[/dim]"
    else:
        body = f"[dim]{crop_path}[/dim]"
    return Panel(body, title=f"[bold]{title}[/bold]", border_style="cyan", padding=(1, 2))


def render_hash_panel(face_hash: str, embedding_dim: int) -> Panel:
    """Show the biometric hash prominently."""
    body = (
        f"[bold cyan]Biometric Hash (SHA-256)[/bold cyan]\n"
        f"[white]{face_hash}[/white]\n"
        f"[dim]from {embedding_dim}-d SFace embedding vector[/dim]"
    )
    return Panel(body, title="[bold]Stage 1 Output[/bold]", border_style="green", padding=(1, 2))


def render_comparison_panel(crop_path: str, match_data: dict) -> Panel:
    """Show side-by-side: face crop info vs discovered post."""
    left = Text(f"[Detected Face]\n[cyan]{crop_path}[/cyan]")
    right = Text(
        f"[Discovered Post]\n"
        f"[bold]{match_data.get('title', 'N/A')}[/bold]\n"
        f"[green]{match_data.get('platform', 'N/A')}[/green]\n"
        f"[dim]{match_data.get('link', '')[:50]}[/dim]"
    )
    cols = Columns([left, right], expand=True, equal=True)
    return Panel(cols, title="[bold]Stage 2: Face → Web Match[/bold]", border_style="green", padding=(1, 2))


def render_blockchain_panel(tx_hash: str, block: int, gas: int) -> Panel:
    """Show the anchoring transaction details."""
    body = (
        f"[bold]Transaction Hash:[/bold] [cyan]{tx_hash}[/cyan]\n"
        f"[bold]Block Number:[/bold]     {block}\n"
        f"[bold]Gas Used:[/bold]         {gas:,}\n"
        f"[bold]Status:[/bold]           [green]✔ Confirmed[/green]"
    )
    return Panel(body, title="[bold]Stage 3: Blockchain Anchor[/bold]", border_style="magenta", padding=(1, 2))


def render_verification_panel(valid: bool, on_chain_hash: str, local_hash: str) -> Panel:
    """Show on-chain verification result with hash comparison."""
    if valid:
        status = "[bold green]✔ VERIFIED[/bold green]"
        border = "green"
    else:
        status = "[bold red]✖ MISMATCH[/bold red]"
        border = "red"

    body = (
        f"{status}\n\n"
        f"[dim]On-Chain Fingerprint:[/dim]\n[cyan]{on_chain_hash}[/cyan]\n\n"
        f"[dim]Local Recomputed:[/dim]\n[cyan]{local_hash}[/cyan]"
    )
    return Panel(body, title="[bold]Stage 4: Verification[/bold]", border_style=border, padding=(1, 2))


def show_face_detection(input_image_path: str, bbox: tuple, auto_close_ms: int = 2000):
    """Show an OpenCV window with the detected face bounding box."""
    try:
        import cv2

        img = cv2.imread(input_image_path)
        if img is None:
            return
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Face Detection", img)
        cv2.waitKey(auto_close_ms)
        cv2.destroyAllWindows()
    except Exception as e:
        console.print(f"[dim]Could not display OpenCV window: {e}[/dim]")

# ---------------------------------------------------------------- identity
def _person_name_from_lens(lens_result):
    try:
        kg = getattr(lens_result, 'knowledge_graph', None)
        if kg is not None and getattr(kg, 'title', None):
            t = kg.title.strip()
            if t and t.lower() not in ('n/a', 'unknown', 'knowledge graph', ''):
                return t
    except Exception:
        pass
    return ''

def _truncate(s, n):
    s = str(s) if s is not None else ''
    return s if len(s) <= n else s[: n - 1] + chr(0x2026)

def _truncate_url(url, n=70):
    if not url: return '(no URL)'
    if len(url) <= n: return url
    h = n // 2 - 1; t = n // 2 + 1
    return url[:h] + chr(0x2026) + url[-t:]

def render_identity_block(lens_result, face_hash=None):
    sel = getattr(lens_result, 'selected', None)
    title = getattr(sel, 'title', 'Unknown') if sel else 'Unknown'
    platform = getattr(sel, 'platform', 'web') if sel else 'web'
    link = getattr(sel, 'link', '') if sel else ''
    reason = getattr(sel, 'reason', '') if sel else ''
    person = _person_name_from_lens(lens_result)
    cache_state = 'HIT' if getattr(lens_result, 'cache_hit', False) else 'MISS'
    serpapi_total = getattr(lens_result, 'serpapi_total_time_s', None) or 0.0
    visual_count = getattr(lens_result, 'visual_match_count', 0)
    upload_ms = getattr(lens_result, 'upload_ms', 0.0) or 0.0
    lens_ms = getattr(lens_result, 'lens_ms', 0.0) or 0.0
    platform_color = {
        'youtube.com': 'red', 'instagram.com': 'magenta', 'x.com': 'white',
        'twitter.com': 'white', 'linkedin.com': 'blue', 'facebook.com': 'blue',
        'reddit.com': 'red', 'official': 'green', 'news': 'yellow',
    }.get(platform, 'cyan')
    identity_line = '[bold magenta]' + person + '[/bold magenta]' if person else '[bold yellow]' + title + '[/bold yellow]'
    body = Text()
    body.append(chr(0x250C) + chr(0x2500) + ' ', style='bold green')
    body.append('IDENTITY', style='bold white on green')
    body.append(' ' + chr(0x2500) * 42 + chr(0x2510) + chr(10), style='bold green')
    body.append('  Person / Entity   : ', style='bold'); body.append(identity_line); body.append(chr(10))
    body.append('  Matched title     : ', style='bold'); body.append(_truncate(title, 70)); body.append(chr(10))
    body.append('  Platform          : ', style='bold'); body.append('[' + platform_color + ']' + platform + '[/' + platform_color + ']'); body.append(chr(10))
    body.append('  Source URL        : ', style='bold'); body.append('[link=' + link + ']' + _truncate_url(link, 80) + '[/link]'); body.append(chr(10))
    if reason: body.append('  Selection reason  : ', style='bold'); body.append('[dim]' + reason + '[/dim]'); body.append(chr(10))
    body.append(chr(0x2514) + chr(0x2500) * 53 + chr(0x2518) + chr(10), style='bold green')
    stats = Table.grid(padding=(0, 2))
    stats.add_column(style='dim'); stats.add_column()
    stats.add_row('Lens candidates', '[bold]' + str(visual_count) + '[/bold]')
    stats.add_row('SerpApi total', '{:.2f}s'.format(serpapi_total))
    stats.add_row('Upload + Lens', '{:.0f}ms + {:.0f}ms'.format(upload_ms, lens_ms))
    stats.add_row('Cache', '[green]HIT[/]' if cache_state == 'HIT' else '[yellow]MISS[/]')
    if face_hash: stats.add_row('Face hash', '[dim]' + _truncate(face_hash, 22) + '[/dim]')
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(); grid.add_column()
    grid.add_row(Panel(body, border_style='green', padding=(0, 1), expand=False), Panel(stats, title='[bold]Stage 2 stats[/bold]', border_style='cyan', padding=(0, 1)))
    return Panel(grid, title='[bold white on green] ' + chr(0x2605) + '  IDENTITY MATCH  ' + chr(0x2605) + ' [/bold white on green]', border_style='bright_green', padding=(0, 1))

def render_final_summary(lens_result, face_hash, on_chain_fingerprint, tx_hash, block, gas, verification_passed, tamper_detected, tamper_local_hash, tamper_on_chain_hash, total_seconds):
    sel = getattr(lens_result, 'selected', None)
    title = getattr(sel, 'title', 'Unknown') if sel else 'Unknown'
    platform = getattr(sel, 'platform', 'web') if sel else 'web'
    link = getattr(sel, 'link', '') if sel else ''
    person = _person_name_from_lens(lens_result)
    if not verification_passed: border = 'red'; status_color = 'red'
    elif tamper_detected: border = 'yellow'; status_color = 'yellow'
    else: border = 'bright_green'; status_color = 'green'
    def col(title_, body_text, color):
        t = Text()
        t.append(chr(0x2554) + chr(0x2550) + chr(0x2550) + ' ', style='bold ' + color)
        t.append(title_, style='bold ' + color)
        t.append(chr(0x255A) + chr(0x2550) * 32 + chr(0x255D) + chr(10), style='bold ' + color)
        for ln in body_text.split(chr(10)):
            t.append(ln + chr(10))
        return t
    identity_body = '  Person / Entity : [bold magenta]' + (person or title) + '[/bold magenta]' + chr(10) + '  Source title    : ' + _truncate(title, 50) + chr(10) + '  Platform        : ' + platform + chr(10) + '  Link            : [link=' + link + ']' + _truncate_url(link, 60) + '[/link]' + chr(10)
    mid_body = '  Face hash       : [cyan]' + _truncate(face_hash, 22) + '[/cyan]' + chr(10) + '  On-chain finger : [cyan]' + _truncate(on_chain_fingerprint, 22) + '[/cyan]' + chr(10)
    if tx_hash: mid_body += '  Tx hash         : [cyan]' + _truncate(tx_hash, 22) + '[/cyan]' + chr(10)
    if block is not None: mid_body += '  Block / Gas     : ' + str(block) + ' / ' + '{:,}'.format(gas) + chr(10)
    status_lbl = ('[bold green]' + chr(0x2714) + ' CONFIRMED[/bold green]') if verification_passed else ('[bold red]' + chr(0x2716) + ' FAILED[/bold red]')
    mid_body += '  Verification    : ' + status_lbl + chr(10)
    if tamper_detected:
        right_body = '  Altered URL  : [dim]' + _truncate(link + '_tampered', 50) + '[/dim]' + chr(10) + '  Local hash   : [red]' + _truncate(tamper_local_hash or '', 22) + '[/red]' + chr(10) + '  On-chain     : [green]' + _truncate(tamper_on_chain_hash or '', 22) + '[/green]' + chr(10) + '  ' + chr(0x21D2) + ' MISMATCH detected (expected)'
    else:
        right_body = '  [dim]Skipped (verification failed)[/dim]'
    col_left  = col('IDENTITY  (Google Lens)',           identity_body, status_color)
    col_mid   = col('ANCHOR  (Anvil chain 31337)',       mid_body,       'magenta')
    col_right = col('TAMPER-EVIDENCE  (security drill)', right_body,     status_color)
    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1); grid.add_column(ratio=1); grid.add_column(ratio=1)
    grid.add_row(col_left, col_mid, col_right)
    footer = '[dim]Total wall time:[/dim] [bold]' + '{:.2f}s'.format(total_seconds) + '[/bold]   [dim]' + chr(0x00B7) + '[/dim]   [dim]Audit report:[/dim] [bold]reports/report_*.json[/bold]'
    return Panel(grid, title='[bold white on bright_green] ' + chr(0x2605) + '  PIPELINE RESULT  ' + chr(0x2605) + ' [/bold white on bright_green]', subtitle=footer, border_style=border, padding=(1, 2))

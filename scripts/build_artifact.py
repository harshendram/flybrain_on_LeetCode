"""Turn the Vite build (web/dist) into a single-page claude.ai Artifact bundle.

The artifact host wraps the page in its own <html>/<head>/<body>, so we emit a fragment: <title> first, the Google
Fonts link, inlined CSS and JS, then the body markup. Data files are published next to the page under data/.
"""

import argparse
import base64
import re
import shutil
from pathlib import Path

from leetfly import paths

DIST = paths.ROOT / "web" / "dist"


def main(out_dir: Path) -> None:
    html = (DIST / "index.html").read_text(encoding="utf-8")
    title = re.search(r"<title>.*?</title>", html, re.S).group(0)
    fonts = re.search(r'<link\s+rel="stylesheet"\s+href="https://fonts\.googleapis\.com[^>]*>', html, re.S).group(0)
    css_href = re.search(r'<link rel="stylesheet"[^>]*href="\./(assets/[^"]+\.css)"[^>]*>', html).group(1)
    js_src = re.search(r'<script type="module"[^>]*src="\./(assets/[^"]+\.js)"[^>]*></script>', html).group(1)
    body = re.search(r"<body>(.*)</body>", html, re.S).group(1)
    body = re.sub(r'\s*<script type="module"[^>]*></script>', "", body).strip()

    css = (DIST / css_href).read_text(encoding="utf-8")
    js = (DIST / js_src).read_text(encoding="utf-8").replace("</script", "<\\/script")
    # artifacts don't serve .bin: ship binaries as base64 text and tell the loader (web/src/data.ts)
    flag = "<script>window.__LEETFLY_B64__ = true;</script>"
    page = f'{title}\n{fonts}\n<style>\n{css}\n</style>\n{body}\n{flag}\n<script type="module">\n{js}\n</script>\n'

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(page, encoding="utf-8")
    data_out = out_dir / "data"
    if data_out.exists():
        shutil.rmtree(data_out)
    data_out.mkdir()
    for f in (DIST / "data").iterdir():
        if f.suffix == ".bin":
            (data_out / f"{f.name}.b64.txt").write_text(base64.b64encode(f.read_bytes()).decode("ascii"))
        else:
            shutil.copy(f, data_out / f.name)
    print(f"{out_dir / 'index.html'}: {len(page) / 1e6:.2f} MB; data: "
          + ", ".join(f"{p.name} {p.stat().st_size / 1e6:.2f} MB" for p in sorted((out_dir / "data").iterdir())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    main(ap.parse_args().out_dir)

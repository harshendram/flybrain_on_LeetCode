"""Turn one page of the site into a single-page claude.ai Artifact bundle.

The artifact host wraps the page in its own <html>/<head>/<body>, so we emit a fragment: <title> first, the Google
Fonts link, inlined CSS and JS, then the body markup. Inlined module code cannot import sibling chunks, so each
page gets its own single-page Vite build (LEETFLY_PAGES) with no shared chunk. Data files are published next to
the page; binaries go as base64 text because artifacts don't serve raw .bin files.

  python scripts/build_artifact.py out/brain --page index --link https://claude.ai/artifact/<fly>
  python scripts/build_artifact.py out/fly --page fly --link https://claude.ai/artifact/<brain>
"""

import argparse
import base64
import os
import re
import shutil
import subprocess
from pathlib import Path

from leetfly import paths

WEB = paths.ROOT / "web"
DIST = WEB / "dist-artifact"
# what each page loads at runtime (relative to the site root); None = the whole folder
PAGES = {
    "index": {"data": None},
    "fly": {"data": ["model.json", "model.bin", "learning.json", "learning.bin", "phase2.json"], "fly": None},
}
OTHER = {"index": "./fly.html", "fly": "./index.html"}


def vite_build(page: str) -> None:
    env = {**os.environ, "LEETFLY_PAGES": f"{page}.html"}
    subprocess.run("npx vite build --outDir dist-artifact --emptyOutDir", cwd=WEB, env=env, shell=True, check=True)


def main(out_dir: Path, page: str, link: str | None) -> None:
    vite_build(page)
    html = (DIST / f"{page}.html").read_text(encoding="utf-8")
    title = re.search(r"<title>.*?</title>", html, re.S).group(0)
    fonts = re.search(r'<link\s+rel="stylesheet"\s+href="https://fonts\.googleapis\.com[^>]*>', html, re.S).group(0)
    css = "\n".join((DIST / h).read_text(encoding="utf-8")
                    for h in re.findall(r'<link rel="stylesheet"[^>]*href="\./(assets/[^"]+\.css)"[^>]*>', html))
    js_src = re.findall(r'<script type="module"[^>]*src="\./(assets/[^"]+\.js)"[^>]*></script>', html)
    assert len(js_src) == 1, f"expected one module script, got {js_src}"
    body_attrs, body = re.search(r"<body([^>]*)>(.*)</body>", html, re.S).groups()
    body = re.sub(r'\s*<script type="module"[^>]*></script>', "", body).strip()
    if link:
        body = body.replace(f'href="{OTHER[page]}"', f'href="{link}" target="_blank" rel="noopener"')
    body_class = re.search(r'class="([^"]+)"', body_attrs)
    if body_class:  # the host owns <body>; the class on <html> matches the same descendant selectors
        body = f'<script>document.documentElement.classList.add("{body_class.group(1)}");</script>\n{body}'

    js = (DIST / js_src[0]).read_text(encoding="utf-8").replace("</script", "<\\/script")
    flag = "<script>window.__LEETFLY_B64__ = true;</script>"  # tells web/src/data.ts to fetch .b64.txt
    fragment = f'{title}\n{fonts}\n<style>\n{css}\n</style>\n{body}\n{flag}\n<script type="module">\n{js}\n</script>\n'

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(fragment, encoding="utf-8")
    for folder, names in PAGES[page].items():
        dest = out_dir / folder
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir()
        for f in sorted((DIST / folder).iterdir()):
            if names is not None and f.name not in names:
                continue
            if f.suffix in (".bin", ".glb"):
                (dest / f"{f.name}.b64.txt").write_text(base64.b64encode(f.read_bytes()).decode("ascii"))
            else:
                shutil.copy(f, dest / f.name)
    files = sorted(p for p in out_dir.rglob("*") if p.is_file() and p.name != "index.html")
    print(f"{out_dir / 'index.html'}: {len(fragment) / 1e6:.2f} MB; "
          + ", ".join(f"{p.relative_to(out_dir).as_posix()} {p.stat().st_size / 1e6:.2f} MB" for p in files))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--page", choices=PAGES, default="index")
    ap.add_argument("--link", help="artifact URL of the other page (replaces the relative cross-page link)")
    a = ap.parse_args()
    main(a.out_dir, a.page, a.link)

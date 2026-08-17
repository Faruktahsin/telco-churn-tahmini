"""Adım 7 — Yazıyı Medium'a kopyalanabilir HTML'e çevir (görseller gömülü)."""
import os as _os, pathlib as _p
KOK = _p.Path(__file__).resolve().parent.parent
_os.chdir(KOK)
for _d in ("outputs", "figures"):
    (KOK / _d).mkdir(exist_ok=True)

import base64
import re
from pathlib import Path

import markdown

md = Path("makale/medium_yazisi.md").read_text(encoding="utf-8")

govde = markdown.markdown(md, extensions=["fenced_code", "tables", "attr_list"])


def gom(m):
    yol, alt = m.group(2), m.group(1)
    p = Path(yol)
    if not p.exists():
        p = Path("figures") / Path(yol).name
    if not p.exists():
        return m.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return (f'<figure><img src="data:image/png;base64,{b64}" alt="{alt}">'
            f'<figcaption>{alt} &nbsp;·&nbsp; <code>{yol}</code></figcaption></figure>')


govde = re.sub(r'<img alt="([^"]*)" src="([^"]+)"\s*/?>', gom, govde)

html = f"""<!DOCTYPE html>
<html lang="tr"><head><meta charset="utf-8">
<title>Churn Tahmininde En Zor Kısım Model Değil</title>
<style>
  body {{ background:#f9f9f7; margin:0; padding:48px 16px 96px;
         font-family: Georgia, "Times New Roman", serif; color:#242424; }}
  main {{ max-width:700px; margin:0 auto; font-size:20px; line-height:1.62; }}
  h1 {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif; font-size:40px;
        line-height:1.18; letter-spacing:-0.6px; margin:0 0 8px; }}
  h3 {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif; font-size:22px;
        font-weight:400; color:#6b6b6b; line-height:1.35; margin:0 0 36px; }}
  h2 {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif; font-size:28px;
        letter-spacing:-0.3px; margin:52px 0 14px; }}
  h3:not(:first-of-type) {{ font-family: system-ui,-apple-system,"Segoe UI",sans-serif;
        font-size:21px; font-weight:700; color:#242424; margin:36px 0 10px; }}
  p {{ margin:0 0 26px; }}
  hr {{ border:0; text-align:center; margin:44px 0; }}
  hr:after {{ content:"· · ·"; letter-spacing:8px; color:#a0a0a0; font-size:22px; }}
  figure {{ margin:38px 0; }}
  img {{ width:100%; border-radius:4px; display:block; }}
  figcaption {{ font-family: system-ui,sans-serif; font-size:14px; color:#8a8a8a;
        text-align:center; margin-top:10px; }}
  figcaption code {{ background:#eeeded; padding:1px 6px; border-radius:3px;
        font-size:12.5px; }}
  pre {{ background:#f2f0ef; border-radius:4px; padding:18px 20px; overflow-x:auto;
        font-size:15px; line-height:1.5; margin:0 0 26px; }}
  code {{ font-family: "SF Mono", Menlo, Consolas, monospace; }}
  p code, li code {{ background:#f2f0ef; padding:2px 6px; border-radius:3px;
        font-size:16px; }}
  ul, ol {{ margin:0 0 26px; padding-left:26px; }}
  li {{ margin-bottom:10px; }}
  strong {{ font-weight:700; }}
  table {{ border-collapse:collapse; width:100%; font-family:system-ui,sans-serif;
        font-size:15px; margin:0 0 26px; }}
  th,td {{ border-bottom:1px solid #e6e6e6; padding:9px 10px; text-align:left; }}
  th {{ color:#6b6b6b; font-weight:700; }}
  em {{ color:#6b6b6b; }}
  .uyari {{ background:#fff8e6; border-left:3px solid #eda100; padding:16px 20px;
        font-family:system-ui,sans-serif; font-size:16px; line-height:1.5;
        border-radius:0 4px 4px 0; margin:0 0 40px; }}
</style></head><body><main>
<div class="uyari"><strong>Bu dosya Medium'a kopyalamak için.</strong>
Aşağıdaki her şeyi seçip (Ctrl/Cmd + A) kopyalayın, Medium editörüne yapıştırın.
Başlıklar, kalın yazılar ve listeler korunur. Görseller yapışmazsa, her görselin
altındaki dosya adına bakıp o noktaya elle yükleyin. Kod bloklarını Medium'da
yeniden oluşturmak için satır başında <code>```</code> yazıp Enter'a basın.
<em>Yayımlamadan önce bu kutuyu silin.</em></div>
{govde}
</main></body></html>"""

Path("makale/medium_kopyala.html").write_text(html, encoding="utf-8")
print("medium_kopyala.html üretildi:",
      round(len(html.encode()) / 1024 / 1024, 2), "MB")

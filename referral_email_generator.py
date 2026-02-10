"""
紹介メール自動生成ツール
========================
companies.csv から企業情報を読み込み、各社Webサイトから理念・ビジョンを抽出し、
Gemini API を用いてパーソナライズされた紹介メールを生成する。
"""

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

import google.generativeai as genai

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------
OUR_MISSION = """
【自社ミッション】
ビジネスモデル: 居室サブリース × 訪問介護・看護
ターゲット: 低所得者・生活保護受給者層
ミッション: 効率的な仕組みを構築し、社会的に不安定な方々へ安定した住まいとケアを届ける。
私たちは「住まい」と「ケア」を一体で提供することで、制度の狭間に落ちてしまう方々を
一人でも多く支えたいと考えています。
""".strip()

VISION_KEYWORDS = [
    "理念", "ビジョン", "ミッション", "代表挨拶", "代表メッセージ",
    "会社概要", "私たちについて", "about", "philosophy", "vision",
    "mission", "greeting", "message", "company",
]

OUTPUT_DIR = Path("output")
REQUEST_TIMEOUT = 15  # seconds
SCRAPE_DELAY = 2  # seconds between requests (有料枠向け高速設定)
SCRAPE_TEXT_LIMIT = 10000  # 1ページあたりの取得文字数上限


# ---------------------------------------------------------------------------
# Excel → CSV プリプロセス
# ---------------------------------------------------------------------------
NAME_HINTS = ["company_name", "社名", "企業名", "会社名", "name", "企業"]
URL_HINTS = ["url", "hp", "ホームページ", "website", "サイト", "リンク", "link"]


def _guess_column(df: pd.DataFrame, hints: list[str]) -> str | None:
    """列名またはセル内容のパターンから該当列を推測して返す。"""
    col_lower = {c: str(c).strip().lower() for c in df.columns}

    # 1) 列名がヒントに一致するか
    for col, name in col_lower.items():
        if any(h in name for h in hints):
            return col

    # 2) セルの中身から推測（URL列はhttp を含むかで判定）
    if hints is URL_HINTS:
        for col in df.columns:
            sample = df[col].dropna().astype(str).head(10)
            if sample.str.contains(r"https?://", case=False).mean() > 0.5:
                return col
    return None


def preprocess_excel(xlsx_path: Path, csv_path: Path) -> None:
    """
    Excel ファイルを読み込み、企業名・URL列を自動判定して
    標準形式の companies.csv に変換する。
    """
    print(f"Excelファイルを検出: {xlsx_path}")
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    print(f"  -> 読み込み行数: {len(df)}, 列: {list(df.columns)}")

    name_col = _guess_column(df, NAME_HINTS)
    url_col = _guess_column(df, URL_HINTS)

    if not name_col or not url_col:
        print(
            f"エラー: 企業名またはURL列を特定できませんでした。\n"
            f"  企業名列候補: {name_col}, URL列候補: {url_col}\n"
            f"  検出した列: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  -> 企業名列: '{name_col}', URL列: '{url_col}'")
    out = df[[name_col, url_col]].rename(
        columns={name_col: "company_name", url_col: "url"}
    )
    out = out.dropna(subset=["company_name", "url"])
    out.to_csv(csv_path, index=False)
    print(f"  -> {csv_path} に {len(out)} 件書き出しました。")


# ---------------------------------------------------------------------------
# スクレイピング
# ---------------------------------------------------------------------------
def _build_session() -> requests.Session:
    """共通の requests Session を構築する。"""
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en;q=0.9",
    })
    return session


def _extract_text(soup: BeautifulSoup) -> str:
    """HTML から本文テキストを抽出する。"""
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    # 連続空行を圧縮
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _find_subpages(soup: BeautifulSoup, base_url: str) -> list[str]:
    """理念・ビジョン系のサブページ URL を探す。"""
    found: list[str] = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        link_text = a_tag.get_text(strip=True).lower()
        href_lower = href.lower()
        if any(kw in link_text or kw in href_lower for kw in VISION_KEYWORDS):
            full_url = urljoin(base_url, href)
            # 同一ドメインのみ
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                found.append(full_url)
    return list(dict.fromkeys(found))  # 重複排除・順序維持


def scrape_company(url: str, session: requests.Session) -> str:
    """
    企業サイトをスクレイピングし、理念・ビジョン関連テキストを返す。
    トップページ + サブページ（最大3件）を取得する。
    """
    collected_texts: list[str] = []

    try:
        resp = session.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        return f"[スクレイピングエラー] {url}: {e}"

    # トップページのテキスト
    top_text = _extract_text(soup)
    collected_texts.append(f"=== トップページ ({url}) ===\n{top_text[:SCRAPE_TEXT_LIMIT]}")

    # サブページの探索
    subpages = _find_subpages(soup, url)[:3]
    for sub_url in subpages:
        time.sleep(SCRAPE_DELAY)
        try:
            resp = session.get(sub_url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            sub_soup = BeautifulSoup(resp.text, "html.parser")
            sub_text = _extract_text(sub_soup)
            collected_texts.append(
                f"=== サブページ ({sub_url}) ===\n{sub_text[:SCRAPE_TEXT_LIMIT]}"
            )
        except requests.RequestException:
            continue  # サブページの失敗は無視

    return "\n\n".join(collected_texts)


# ---------------------------------------------------------------------------
# Gemini によるメール生成
# ---------------------------------------------------------------------------
def _configure_gemini() -> genai.GenerativeModel:
    """Gemini API を設定し、モデルを返す。"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("エラー: 環境変数 GEMINI_API_KEY が設定されていません。", file=sys.stderr)
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("models/gemini-2.0-flash")


PROMPT_TEMPLATE = """\
あなたは、企業のビジョンを深く理解し、文脈を読み解く「戦略的ライティング・エンジニア」です。

以下の情報をもとに、紹介会社 {company_name} へ送る **パーソナライズされた紹介メール** を作成してください。

---
{our_mission}
---

【相手企業のWebサイトから取得した情報】
{scraped_text}

---

## 指示
1. 相手企業の「想い（理念・ビジョン）」を読み取り、自社ミッションとの **共通点** を特定してください。
2. 「なぜ今、御社と組む必要があるのか」という **推論** を含めてください。
3. 以下の構成でメールを生成してください（Markdown形式）:

```
## 件名
（件名をここに）

## 本文

（挨拶）

（相手のビジョンへの共感 — 具体的に引用・言及すること）

（自社の紹介と接点 — 居室サブリース×訪問介護の仕組みを簡潔に説明し、
  相手のビジョンとどう結びつくかを論理的に述べる）

（面談の提案 — 具体的な次のステップを提示する）

（結び）
```

4. トーンは **エモーショナルかつ論理的** に。丁寧なビジネス日本語で書いてください。
5. メール本文は 400〜600 文字程度に収めてください。
"""


MAX_RETRIES = 3
RETRY_BASE_WAIT = 2  # seconds (指数バックオフの基底: 2s, 4s, 8s)


def generate_email(
    model: genai.GenerativeModel,
    company_name: str,
    scraped_text: str,
) -> str:
    """Gemini API を呼び出してメール本文を生成する（指数バックオフ付きリトライ）。"""
    prompt = PROMPT_TEMPLATE.format(
        company_name=company_name,
        our_mission=OUR_MISSION,
        scraped_text=scraped_text[:SCRAPE_TEXT_LIMIT],
    )
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_WAIT ** attempt  # 2s -> 4s -> 8s
                print(f"  -> API エラー (試行 {attempt}/{MAX_RETRIES}): {e}")
                print(f"     {wait}秒後にリトライします...")
                time.sleep(wait)
    return f"[メール生成エラー] {company_name}: {last_error}"


# ---------------------------------------------------------------------------
# メイン処理
# ---------------------------------------------------------------------------
def main() -> None:
    xlsx_path = Path("companies.xlsx")
    csv_path = Path("companies.csv")

    # Excel が存在すれば自動変換
    if xlsx_path.exists():
        preprocess_excel(xlsx_path, csv_path)

    if not csv_path.exists():
        print(f"エラー: {csv_path} が見つかりません。", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    required_cols = {"company_name", "url"}
    if not required_cols.issubset(df.columns):
        print(
            f"エラー: CSV に必要な列がありません。必要: {required_cols}",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = _configure_gemini()
    session = _build_session()

    print(f"対象企業数: {len(df)}")
    print("=" * 60)

    for idx, row in df.iterrows():
        company_name = str(row["company_name"]).strip()
        url = str(row["url"]).strip()
        print(f"\n[{idx + 1}/{len(df)}] {company_name} ({url})")

        # 1. スクレイピング
        print("  -> Webサイトを取得中...")
        scraped_text = scrape_company(url, session)

        # 2. メール生成
        print("  -> メールを生成中...")
        email_md = generate_email(model, company_name, scraped_text)

        # 3. ファイル出力
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", company_name)
        out_path = OUTPUT_DIR / f"{safe_name}.md"
        out_path.write_text(email_md, encoding="utf-8")
        print(f"  -> 保存完了: {out_path}")

        # レート制限対策
        time.sleep(SCRAPE_DELAY)

    print("\n" + "=" * 60)
    print(f"全件完了。出力先: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

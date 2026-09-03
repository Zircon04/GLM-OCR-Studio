#!/usr/bin/env python3
"""SQLite FTS5 index for completed local OCR results."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "library.sqlite3"


def connection() -> sqlite3.Connection:
    db = sqlite3.connect(DB)
    db.execute("pragma journal_mode=WAL")
    db.execute("create table if not exists documents(id integer primary key, path text unique, title text, source text, pages integer, created real, updated real, status text)")
    db.execute("create virtual table if not exists document_text using fts5(title, body, path UNINDEXED)")
    return db


def find_markdown(root: Path) -> Path | None:
    files = [p for p in root.rglob("*.md") if ".job" not in p.parts]
    enhanced = [p for p in files if p.name.endswith("_enhanced.md")]
    return (enhanced or files)[0] if files else None


def index_result(root: Path) -> None:
    state_path = root / "job-state.json"
    if not state_path.exists(): return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    markdown = find_markdown(root)
    if not markdown: return
    text = markdown.read_text(encoding="utf-8", errors="replace")
    plain = re.sub(r"<[^>]+>|!\[[^]]*\]\([^)]+\)|[`#$*_]", " ", text)
    db = connection()
    cursor = db.execute(
        "insert into documents(path,title,source,pages,created,updated,status) values(?,?,?,?,?,?,?) on conflict(path) do update set title=excluded.title,source=excluded.source,pages=excluded.pages,updated=excluded.updated,status=excluded.status returning id",
        (str(root), state.get("result_name", root.name), state.get("source", ""), state.get("total_pages", 1), state.get("started_at", time.time()), time.time(), state.get("status", "complete")),
    )
    rowid = cursor.fetchone()[0]
    db.execute("delete from document_text where rowid=?", (rowid,))
    db.execute("insert into document_text(rowid,title,body,path) values(?,?,?,?)", (rowid, state.get("result_name", root.name), plain, str(root)))
    db.commit(); db.close()


def search(query: str) -> list[dict]:
    db = connection()
    statement = "select d.title,d.path,d.source,d.pages,d.updated,snippet(document_text,1,'<b>','</b>','…',18) from document_text join documents d on d.id=document_text.rowid where document_text match ? order by rank limit 50"
    try:
        rows = db.execute(statement, (query,)).fetchall()
    except sqlite3.OperationalError:
        rows = db.execute(statement, ('"' + query.replace('"', '""') + '"',)).fetchall()
    if not rows:
        needle = f"%{query}%"
        rows = db.execute(
            "select d.title,d.path,d.source,d.pages,d.updated,substr(document_text.body,max(1,instr(document_text.body,?)-40),180) from document_text join documents d on d.id=document_text.rowid where document_text.body like ? or d.title like ? order by d.updated desc limit 50",
            (query, needle, needle),
        ).fetchall()
    db.close()
    return [{"title": r[0], "path": r[1], "source": r[2], "pages": r[3], "updated": r[4], "snippet": r[5]} for r in rows]


def list_documents() -> list[dict]:
    db = connection()
    rows = db.execute("select title,path,source,pages,updated,status from documents order by updated desc limit 200").fetchall()
    db.close()
    return [{"title": r[0], "path": r[1], "source": r[2], "pages": r[3], "updated": r[4], "status": r[5]} for r in rows]


def remove_document(path: str) -> None:
    db = connection()
    row = db.execute("select id from documents where path=?", (str(Path(path).expanduser().resolve()),)).fetchone()
    if row:
        db.execute("delete from document_text where rowid=?", (row[0],))
        db.execute("delete from documents where id=?", (row[0],))
    db.commit(); db.close()


def prune_missing() -> int:
    db = connection()
    rows = db.execute("select id,path from documents").fetchall()
    missing = [(rowid, path) for rowid, path in rows if not Path(path).exists()]
    for rowid, _ in missing:
        db.execute("delete from document_text where rowid=?", (rowid,))
        db.execute("delete from documents where id=?", (rowid,))
    db.commit(); db.close()
    return len(missing)


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index"); a.add_argument("path")
    s = sub.add_parser("search"); s.add_argument("query")
    sub.add_parser("list")
    r = sub.add_parser("remove"); r.add_argument("path")
    sub.add_parser("prune")
    args = p.parse_args()
    if args.cmd == "index": index_result(Path(args.path).resolve()); return 0
    if args.cmd == "remove": remove_document(args.path); return 0
    if args.cmd == "prune": print(json.dumps({"removed": prune_missing()})); return 0
    print(json.dumps(search(args.query) if args.cmd == "search" else list_documents(), ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())

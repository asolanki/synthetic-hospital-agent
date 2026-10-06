"""Ranked search over the public CMS ICD-10-CM 2025 code book (tabular list, alphabetic index,
Neoplasm Table and Table of Drugs and Chemicals).

Only the public-domain CMS release is indexed, never the benchmark's own diagnosis table, so the
tool gives an agent what a human coder has: titles, inclusion terms and index entries. It cannot
reveal a patient's reference codes. The 2025 release matches the simulator's terminology table.

    uv run python -m training.icd10_codebook fetch     # downloads + verifies the CMS release

Documents: each code's tabular title, each of its inclusion terms, each alphabetic-index path
("Disease, lung, interstitial" -> J84.9), and each Neoplasm Table / drug-table cell phrased as a
clinician would ("esophagus carcinoma" -> C15.9, "Lisinopril adverse effect" -> T46.4X5). A query
is scored with BM25 against every document; a code's score is its best document. Results carry the
official tabular title and the matched text. Common anatomical adjectives are mapped to their
nouns (esophageal -> esophagus) on both sides; this is generic vocabulary, not label-derived.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

# training/environment/synthetic_hospital_dx/icd10.py -> repo root is parents[3].
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "runs" / "icd10cm"
CMS_URL = "https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index.zip"
CMS_SHA256 = "ecad76ac02b0ee74fda1f526f205f067795204fbb34bb4051c74410b1722fd18"
ZIP_NAME = "tables-index-2025.zip"
INDEX_NAME = "search-index-2025-v3.json"

STOPWORDS = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "into", "is", "nos", "not",
             "of", "on", "or", "the", "to", "with", "without", "due", "elsewhere", "classified"}
K1, B = 1.2, 0.75
# Anatomical adjective -> the noun the code book uses.
ADJECTIVES = {"esophageal": "esophagus", "gastric": "stomach", "hepatic": "liver", "renal": "kidney",
              "pulmonary": "lung", "cardiac": "heart", "uterine": "uterus", "vulvar": "vulva",
              "vaginal": "vagina", "pancreatic": "pancreas", "colonic": "colon", "rectal": "rectum",
              "ovarian": "ovary", "testicular": "testis", "prostatic": "prostate", "cerebral": "brain",
              "pleural": "pleura", "bronchial": "bronchus", "tracheal": "trachea", "laryngeal": "larynx",
              "splenic": "spleen", "cutaneous": "skin", "ocular": "eye", "vesical": "bladder",
              "duodenal": "duodenum", "ileal": "ileum", "biliary": "bile"}
# Column headings of the CMS Neoplasm Table and Table of Drugs and Chemicals, as the phrases a
# clinician writes. Each phrase becomes its own short document, so BM25 length normalization does
# not favour a column for having fewer synonyms.
NEOPLASM_COLUMNS = {"2": ("malignant neoplasm", "cancer", "carcinoma", "adenocarcinoma", "malignant tumor"),
                    "3": ("secondary malignant neoplasm", "metastasis", "metastatic cancer"),
                    "4": ("carcinoma in situ",), "5": ("benign neoplasm", "benign tumor"),
                    "6": ("neoplasm of uncertain behavior",), "7": ("neoplasm of unspecified behavior",)}
DRUG_COLUMNS = {"2": ("poisoning", "accidental overdose"), "3": ("intentional self-harm poisoning",),
                "4": ("assault poisoning",), "5": ("poisoning undetermined intent",),
                "6": ("adverse effect", "adverse reaction", "side effect"), "7": ("underdosing",)}


def codebook_dir() -> Path:
    return Path(os.environ.get("SHRL_ICD10_DIR", DEFAULT_DIR))


def tokens(text: str) -> list[str]:
    out = []
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        if raw in STOPWORDS:
            continue
        if raw == "colorectal":
            out.extend(("colon", "rectum"))
            continue
        raw = ADJECTIVES.get(raw, raw)
        if len(raw) > 4 and raw.endswith("ies"):
            raw = raw[:-3] + "y"
        elif len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        out.append(raw)
    return out


def _text(element) -> str:
    return " ".join("".join(element.itertext()).split()) if element is not None else ""


def build_documents(zip_path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """(code -> official title, [(document text, code)]) from the CMS tabular list and tables."""
    with zipfile.ZipFile(zip_path) as archive:
        tabular = ET.fromstring(archive.read("icd10cm_tabular_2025.xml"))
        index = ET.fromstring(archive.read("icd10cm_index_2025.xml"))
        neoplasm = ET.fromstring(archive.read("icd10cm_neoplasm_2025.xml"))
        drug = ET.fromstring(archive.read("icd10cm_drug_2025.xml"))
    titles: dict[str, str] = {}
    documents: list[tuple[str, str]] = []
    for diag in tabular.iter("diag"):
        code, title = _text(diag.find("name")), _text(diag.find("desc"))
        if not code or not title:
            continue
        titles[code] = title
        documents.append((title, code))
        for block in diag.findall("inclusionTerm"):
            documents.extend((_text(note), code) for note in block.findall("note") if _text(note))

    def walk(term, path: list[str]) -> None:
        here = path + [_text(term.find("title"))]
        code = _text(term.find("code")).rstrip("-").rstrip(".")
        if code:
            documents.append((", ".join(p for p in here if p), code))
        for child in term.findall("term"):
            walk(child, here)

    for main in index.iter("mainTerm"):
        walk(main, [])

    def table(term, path: list[str], columns: dict[str, tuple[str, ...]]) -> None:
        here = path + [_text(term.find("title"))]
        for cell in term.findall("cell"):
            code = _text(cell).rstrip("-").rstrip(".")
            if code and cell.get("col") in columns and re.match(r"^[A-Z]\d", code):
                site = " ".join(p for p in here if p)
                documents.extend((f"{site} {phrase}", code) for phrase in columns[cell.get("col")])
        for child in term.findall("term"):
            table(child, here, columns)

    for main in neoplasm.iter("mainTerm"):
        # The root row ("Neoplasm, neoplastic") only means "unspecified site"; its children are sites.
        if _text(main.find("title")).lower().startswith("neoplasm"):
            for child in main.findall("term"):
                table(child, [], NEOPLASM_COLUMNS)
        else:
            table(main, [], NEOPLASM_COLUMNS)
    for main in drug.iter("mainTerm"):
        table(main, [], DRUG_COLUMNS)
    return titles, documents


@dataclass
class Hit:
    code: str
    title: str
    matched: str
    score: float


class CodeBook:
    def __init__(self, titles: dict[str, str], documents: list[tuple[str, str]]):
        self.titles = titles
        self.documents = documents
        self.lengths = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, (text, _) in enumerate(documents):
            toks = tokens(text)
            self.lengths.append(len(toks))
            counts: dict[str, int] = defaultdict(int)
            for t in toks:
                counts[t] += 1
            for t, c in counts.items():
                self.postings[t].append((i, c))
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        n = len(documents)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def search(self, query: str, limit: int = 8) -> list[Hit]:
        scores: dict[int, float] = defaultdict(float)
        for t in set(tokens(query)):
            for i, count in self.postings.get(t, ()):
                norm = count + K1 * (1 - B + B * self.lengths[i] / self.avg_length)
                scores[i] += self.idf[t] * count * (K1 + 1) / norm
        best: dict[str, int] = {}
        for i, score in scores.items():
            code = self.documents[i][1]
            # A code's score is its best document; ties prefer the shorter document.
            if code not in best or (score, -self.lengths[i]) > (scores[best[code]], -self.lengths[best[code]]):
                best[code] = i
        ranked = sorted(best.items(), key=lambda kv: (-scores[kv[1]], self.lengths[kv[1]], -len(kv[0]), kv[0]))
        return [Hit(code, self.titles.get(code, self.documents[i][0]), self.documents[i][0], round(scores[i], 3))
                for code, i in ranked[:limit]]


@lru_cache(maxsize=1)
def load() -> CodeBook:
    directory = codebook_dir()
    cache = directory / INDEX_NAME
    if cache.exists():
        payload = json.loads(cache.read_text())
        return CodeBook(payload["titles"], [tuple(d) for d in payload["documents"]])
    zip_path = directory / ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} missing; run `python -m training.icd10_codebook fetch`")
    titles, documents = build_documents(zip_path)
    cache.write_text(json.dumps({"source": CMS_URL, "sha256": CMS_SHA256,
                                 "titles": titles, "documents": documents}))
    return CodeBook(titles, documents)


def render(query: str, hits: list[Hit]) -> str:
    if not hits:
        return f'No ICD-10-CM 2025 entries match "{query}". Try other words or a broader term.'
    lines = [f'ICD-10-CM 2025 code book results for "{query}":']
    for n, h in enumerate(hits, 1):
        extra = f"  [matched: {h.matched}]" if h.matched != h.title else ""
        category = f" (category {h.code.split('.')[0]})" if "." in h.code else ""
        lines.append(f"{n}. {h.code} {h.title}{category}{extra}")
    return "\n".join(lines)

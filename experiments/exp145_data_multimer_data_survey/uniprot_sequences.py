# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch protein sequences for UniProt accessions, with a UniParc fallback.

Two callers need this: ``curate_val_reference.py`` (~42 k accessions) and
``curate_subunit_sequences.py`` (~2.0 M), which is the second use that
``experiments/AGENTS.md`` sets as the bar for factoring a helper out.

The fallback is the whole point. AFDB is built on a pinned UniProt release, and
about 62% of the contacts-v1 validation accessions are TrEMBL entries UniProt has
since deleted. Those return HTTP 200 with an **empty body** from
``/uniprotkb/<acc>.fasta``, so a status-code check does not notice, and they are
simply absent from a batch response. UniParc keeps every sequence it ever saw,
so it recovers them.

Two parsing traps, both of which fail silently and look like missing data:

- ``/uniprotkb/accessions`` paginates independently of its 100-accession request
  limit. Reading only the first page recovers about 38%.
- One UniParc record lists every accession sharing that sequence, separated by
  ``"; "`` rather than commas. Splitting on commas alone keeps only the first.
"""

import http.client
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

UNIPROTKB_ACCESSIONS = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPARC_SEARCH = "https://rest.uniprot.org/uniparc/search"

# UniProt caps the accessions endpoint at 100 accessions per request; pagination
# is separate, so ask for a page large enough to hold a whole batch in one go.
KB_BATCH = 100
KB_PAGE = 500

# UniParc batches are OR-joined into the query string, so they stay well inside
# the URL length limit: 50 ten-character accessions is roughly 700 characters.
UNIPARC_BATCH = 50

RETRY_CODES = (429, 500, 502, 503, 504)
_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


def get_page(url: str, timeout: int = 180, attempts: int = 6) -> tuple[str, str | None]:
    """GET one REST page, returning (body, next-page URL or None).

    Retries transient HTTP codes and dropped connections with exponential backoff.
    Anything else, and any
    exhausted retry, propagates: a silently dropped page is indistinguishable
    from a genuinely absent accession downstream.
    """
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read().decode()
                link = response.headers.get("Link", "")
            match = _NEXT_LINK.search(link)
            return body, match.group(1) if match else None
        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            http.client.HTTPException,
            OSError,
        ) as exc:
            # RemoteDisconnected is an http.client.HTTPException and a
            # ConnectionResetError, and urllib does not wrap it in URLError, so a
            # narrower except clause lets it kill a multi-hour run outright.
            # TimeoutError and ConnectionResetError are both OSError subclasses.
            code = getattr(exc, "code", None)
            if (code is not None and code not in RETRY_CODES) or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def parse_fasta(text: str, out: dict[str, str]) -> None:
    """Accumulate ``>sp|ACC|NAME`` FASTA records into ``out``, keyed by accession."""
    accession, chunks = None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if accession:
                out[accession] = "".join(chunks)
            accession, chunks = line.split("|")[1], []
            continue
        chunks.append(line.strip())
    if accession:
        out[accession] = "".join(chunks)


def _fetch_kb_batch(batch: list[str]) -> dict[str, str]:
    query = urllib.parse.urlencode(
        {"accessions": ",".join(batch), "format": "fasta", "size": str(KB_PAGE)}
    )
    url: str | None = f"{UNIPROTKB_ACCESSIONS}?{query}"
    out: dict[str, str] = {}
    while url:
        text, url = get_page(url)
        parse_fasta(text, out)
    return out


def _fetch_uniparc_batch(batch: list[str]) -> dict[str, str]:
    query = urllib.parse.urlencode({
        "query": " OR ".join(batch),
        "format": "tsv",
        "fields": "upi,accession,sequence",
        "size": str(max(len(batch) * 2, 50)),
    })
    url: str | None = f"{UNIPARC_SEARCH}?{query}"
    wanted = set(batch)
    out: dict[str, str] = {}
    while url:
        text, url = get_page(url)
        for line in text.splitlines()[1:]:
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            listed, sequence = fields[1], fields[2]
            # "A0A016IB74.1; A0A017N198.1" — semicolon-separated and versioned.
            for entry in re.split(r"[;,]", listed):
                accession = entry.strip().split(".")[0]
                if accession in wanted:
                    out[accession] = sequence
    return out


def _run_batches(
    accessions: list[str],
    size: int,
    worker: Callable[[list[str]], dict[str, str]],
    threads: int,
    label: str,
    progress: int,
) -> dict[str, str]:
    batches = [accessions[i:i + size] for i in range(0, len(accessions), size)]
    out: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for result in pool.map(worker, batches):
            out.update(result)
            done += 1
            if done % progress == 0 or done == len(batches):
                seen = min(done * size, len(accessions))
                print(f"  [{label}] {seen:,}/{len(accessions):,} ({len(out):,} sequences)",
                      flush=True)
    return out


def fetch_sequences(
    accessions: Iterable[str], threads: int = 8, progress: int = 25
) -> tuple[dict[str, str], dict[str, int]]:
    """Resolve accessions to sequences, UniProtKB first then UniParc.

    Returns (sequences, counts). ``counts`` reports how many came from each
    source and how many resolved nowhere, so a caller can surface the shortfall
    instead of discovering it as a short FASTA.
    """
    accessions = list(accessions)
    sequences = _run_batches(
        accessions, KB_BATCH, _fetch_kb_batch, threads, "uniprotkb", progress
    )
    from_kb = len(sequences)

    absent = [a for a in accessions if a not in sequences]
    if absent:
        print(f"  {len(absent):,} absent from UniProtKB; recovering from UniParc", flush=True)
        sequences.update(
            _run_batches(absent, UNIPARC_BATCH, _fetch_uniparc_batch, threads, "uniparc", progress)
        )

    counts = {
        "requested": len(accessions),
        "resolved": len(sequences),
        "from_uniprotkb": from_kb,
        "from_uniparc": len(sequences) - from_kb,
        "unresolved": len(accessions) - len(sequences),
    }
    return sequences, counts


def write_fasta(path, sequences: dict[str, str], order: Iterable[str] | None = None) -> int:
    """Write sequences as wrapped FASTA. Returns the number of records written."""
    written = 0
    with open(path, "w") as handle:
        for accession in (order if order is not None else sorted(sequences)):
            sequence = sequences.get(accession)
            if not sequence:
                continue
            handle.write(f">{accession}\n")
            for i in range(0, len(sequence), 60):
                handle.write(sequence[i:i + 60] + "\n")
            written += 1
    return written

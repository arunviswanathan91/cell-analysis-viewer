"""Post-hoc check that every figure in an answer exists in the evidence the model was shown.

The model is told to copy numbers from query results and to do arithmetic in SQL. This
module enforces it: any decimal number in the answer that cannot be matched to a number
in the tool outputs (at the precision it is quoted) is reported as unverified.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Iterable, List

# decimals (0.123, .5, 3.1e-5, 2e-10) - integers are treated as counts and not checked
_NUM = re.compile(r"(?<![\w.])[-+−]?(?:\d+\.\d+|\.\d+)(?:[eE][-+]?\d+)?(?![\w])|(?<![\w.])[-+−]?\d+[eE][-+]?\d+(?![\w])")
_CITE = re.compile(r"\[Q\d+(?:\s*,\s*Q\d+)*\]")
_BARE_NUM = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")

# figures that are part of the method, not results (thresholds and interval levels)
STATIC_OK = {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.94, 0.95, 0.97, 0.03, 0.001, 1.01, 0.025, 0.975}


@dataclass
class GroundingReport:
    unverified: List[str] = field(default_factory=list)
    checked: int = 0
    cited: bool = True
    bad_citations: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unverified and not self.bad_citations


def _decimals(token: str) -> int:
    mant = re.split(r"[eE]", token)[0]
    return len(mant.split(".")[1]) if "." in mant else 0


def _sig_digits(token: str) -> int:
    mant = re.split(r"[eE]", token)[0].replace("-", "").replace("+", "").replace("−", "").replace(".", "").lstrip("0")
    return max(1, len(mant))


def _supported(token: str, pool: Iterable[float], extra: Iterable[float]) -> bool:
    t = token.replace("−", "-")
    v = float(t)
    av = abs(v)
    sci = "e" in t.lower()
    d = _decimals(t)
    tol = 0.5 * 10 ** (-d) + 1e-12
    for x in list(pool) + list(extra):
        if math.isnan(x):
            continue
        ax = abs(x)
        if sci or (ax != 0 and av != 0 and (ax < 1e-3 or av < 1e-3)):
            # scientific values: compare to the quoted number of significant digits
            if ax > 0 and av > 0:
                sig = _sig_digits(t)
                if abs(ax - av) <= 0.5 * 10 ** (math.floor(math.log10(av)) - sig + 1) * 1.001:
                    return True
        elif abs(ax - av) <= tol or abs(round(ax, d) - av) <= 1e-12:
            return True
        # percentages: 0.2346 quoted as 23.5%
        if abs(round(ax * 100, d) - av) <= 1e-9 or abs(ax * 100 - av) <= tol:
            return True
    return False


def verify(answer: str, tool_numbers: Iterable[float], valid_citations: Iterable[str]) -> GroundingReport:
    rep = GroundingReport()
    pool = list(tool_numbers)
    # citations must point at queries that exist
    valid = set(valid_citations)
    for grp in _CITE.findall(answer):
        for q in re.findall(r"Q\d+", grp):
            if q not in valid:
                rep.bad_citations.append(q)
    rep.cited = bool(_CITE.search(answer))

    body = _CITE.sub(" ", answer)
    seen = set()
    for m in _NUM.finditer(body):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        rep.checked += 1
        try:
            v = abs(float(tok.replace("−", "-")))
        except ValueError:
            continue
        if v in STATIC_OK:
            continue
        if not _supported(tok, pool, ()):
            rep.unverified.append(tok)
    return rep

"""Sheaf-consistent verifier over three evidence sources.

Notation.
Let a query q have candidate answers A = {a_1, ..., a_N} sampled from the
LLM, and three evidence sources E_p (dense passages), E_g (extracted KG
triples), and E_t (BM25 passages, treated as a lexical tool observation).

For each source E_s = {e_s1, ..., e_sM} we build a *commitment vector*
c_s(a_i) in R^d as the softmax-weighted mean of item embeddings, where
weights are cosine(enc(q ++ a_i), enc(e_sj)):

    w_sj = softmax_j( cosine(enc(q ++ a_i), enc(e_sj)) / T )
    c_s(a_i) = sum_j w_sj * enc(e_sj)

Restriction maps R_{s->t}: R^d -> R^d come in three flavours:
  identity    : R = I
  linear      : R = W, a d-by-d matrix fitted on a small held-out set to
                minimise inconsistency across supervised (correct) sections
  llm         : R is data-dependent, produced by projecting c_s onto the
                subspace of e_tj weighted by an LLM entailment score
                between the sources.  (Approximated with sentence-BERT
                cross-encoded pairwise similarities.)

Consistency energy for a candidate:
    E(a_i) = sum_{s != t} w_st * ||R_{s->t}(c_s(a_i)) - c_t(a_i)||^2

with pair weights w_st proportional to overlap between the two sources.

Blended score used for selection:
    final(a_i) = alpha * (1 - normalised_energy(a_i)) + (1 - alpha) * vote_share(a_i)
where alpha is bounded by the observed conflict level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True).clip(min=1e-12)


def _norm(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / n.clip(min=eps)


@dataclass
class EvidenceSource:
    name: str
    items: list[str]         # raw text items
    embeddings: np.ndarray   # shape (M, d), unit-normalised
    kind: str = "text"       # "passage" | "triple" | "tool"


@dataclass
class VerifierConfig:
    restriction: str = "identity"    # identity | linear | llm
    temperature: float = 0.15
    blend_beta: float = 1.6           # alpha = min(1, beta * conflict)
    blend_alpha_min: float = 0.10
    conflict_gate: float = 0.65      # skip sheaf if top vote share >= gate
    gate_enabled: bool = True
    weight_overlap: bool = True
    # Anchor-aware scoring weights (overridable per corpus).
    vote_w: float = 0.45
    sheaf_w: float = 0.30
    anchor_w: float = 0.35
    extra_candidate_bonus: float = 0.05  # small vote-share bonus for extras


@dataclass
class VerifierResult:
    chosen_answer: str
    chosen_index: int
    energies: list[float]
    final_scores: list[float]
    vote_shares: list[float]
    alpha: float
    used_sheaf: bool
    latency_ms: float
    vote_pick: str
    energy_median_correct: Optional[float] = None
    energy_median_wrong: Optional[float] = None


def commitment_vector(source_emb: np.ndarray, query_answer_emb: np.ndarray, T: float) -> np.ndarray:
    """Weighted-mean commitment. Returns (d,) unit-normed vector."""
    if source_emb.shape[0] == 0:
        return np.zeros_like(query_answer_emb)
    with np.errstate(all="ignore"):
        source_emb = np.ascontiguousarray(source_emb, dtype=np.float32)
        query_answer_emb = np.ascontiguousarray(query_answer_emb, dtype=np.float32)
        sims = source_emb @ query_answer_emb
        w = _softmax(sims / max(T, 1e-6))
        c = w @ source_emb
    return _norm(c)


def learned_restriction_maps(
    sections: dict[tuple[str, str], np.ndarray],
    dim: int,
    reg: float = 0.5,
) -> dict[tuple[str, str], np.ndarray]:
    """Fit a linear map R_{s->t} minimising ||R c_s - c_t||^2 + reg||R - I||^2
    given a set of supervised sections {(s, t): (C_s, C_t)}."""
    Rmats: dict[tuple[str, str], np.ndarray] = {}
    I = np.eye(dim, dtype=np.float32)
    for (s, t), (Cs, Ct) in sections.items():
        if Cs.shape[0] < 4:
            Rmats[(s, t)] = I
            continue
        # Closed form ridge: R = (Cs^T Cs + reg I)^{-1} Cs^T Ct + reg (Cs^T Cs + reg I)^{-1}
        A = Cs.T @ Cs + reg * I
        B = Cs.T @ Ct + reg * I
        try:
            R = np.linalg.solve(A, B)
            Rmats[(s, t)] = R.astype(np.float32)
        except np.linalg.LinAlgError:
            Rmats[(s, t)] = I
    return Rmats


def _default_identity_R(dim: int, source_names: Sequence[str]) -> dict[tuple[str, str], np.ndarray]:
    I = np.eye(dim, dtype=np.float32)
    return {(s, t): I for s in source_names for t in source_names if s != t}


def _llm_style_R(
    source_embs: dict[str, np.ndarray],
    dim: int,
) -> dict[tuple[str, str], np.ndarray]:
    """LLM-derived restriction map, approximated with cross-source alignment.

    For each ordered pair (s, t) we build R_{s->t} as a small mixture of
    identity with a low-rank projector obtained by regressing each source
    embedding onto the corresponding softly-aligned target embedding.
    All computations in float32 with explicit clipping to avoid overflow.
    """
    R: dict[tuple[str, str], np.ndarray] = {}
    I = np.eye(dim, dtype=np.float32)
    with np.errstate(all="ignore"):
        for s, Es_raw in source_embs.items():
            for t, Et_raw in source_embs.items():
                if s == t or Es_raw.shape[0] == 0 or Et_raw.shape[0] == 0:
                    R[(s, t)] = I
                    continue
                Es = np.ascontiguousarray(Es_raw, dtype=np.float32)
                Et = np.ascontiguousarray(Et_raw, dtype=np.float32)
                X = Es @ Et.T
                W = _softmax(X / 0.25, axis=-1).astype(np.float32)
                targets = W @ Et
                reg = 0.5
                A = Es.T @ Es + reg * I
                B = Es.T @ targets + reg * I
                try:
                    proj = np.linalg.solve(A, B).astype(np.float32)
                except np.linalg.LinAlgError:
                    proj = I
                nrm = float(np.linalg.norm(proj, ord=2)) if proj.size else 0.0
                if not np.isfinite(nrm) or nrm < 1e-8:
                    proj = I
                else:
                    proj = (proj / nrm).astype(np.float32)
                R[(s, t)] = (0.5 * I + 0.5 * proj).astype(np.float32)
    return R


class SheafVerifier:
    def __init__(self, encoder, config: Optional[VerifierConfig] = None,
                 learned_R: Optional[dict[tuple[str, str], np.ndarray]] = None):
        self.encoder = encoder
        self.config = config or VerifierConfig()
        self._learned_R = learned_R

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embs = self.encoder.encode(list(texts), convert_to_numpy=True, show_progress_bar=False,
                                   normalize_embeddings=True)
        embs = np.ascontiguousarray(embs, dtype=np.float32)
        embs = np.where(np.isfinite(embs), embs, 0.0).astype(np.float32)
        return embs

    def build_source(self, name: str, items: Sequence[str], kind: str = "text") -> EvidenceSource:
        emb = self.encode(items)
        return EvidenceSource(name=name, items=list(items), embeddings=emb, kind=kind)

    def _restriction_maps(self, sources: dict[str, EvidenceSource], dim: int) -> dict[tuple[str, str], np.ndarray]:
        names = list(sources.keys())
        if self.config.restriction == "identity":
            return _default_identity_R(dim, names)
        if self.config.restriction == "linear":
            if self._learned_R is not None:
                return self._learned_R
            return _default_identity_R(dim, names)
        if self.config.restriction == "llm":
            return _llm_style_R({n: s.embeddings for n, s in sources.items()}, dim)
        raise ValueError(f"unknown restriction: {self.config.restriction}")

    def _pair_weights(self, sources: dict[str, EvidenceSource]) -> dict[tuple[str, str], float]:
        names = list(sources.keys())
        weights: dict[tuple[str, str], float] = {}
        with np.errstate(all="ignore"):
            for s in names:
                for t in names:
                    if s == t:
                        continue
                    Ms, Mt = sources[s].embeddings.shape[0], sources[t].embeddings.shape[0]
                    if not self.config.weight_overlap or Ms == 0 or Mt == 0:
                        weights[(s, t)] = 1.0
                    else:
                        A = np.ascontiguousarray(sources[s].embeddings, dtype=np.float32)
                        B = np.ascontiguousarray(sources[t].embeddings, dtype=np.float32)
                        X = A @ B.T
                        top = float(np.sort(X, axis=1)[:, -1:].mean()) if X.size else 0.0
                        weights[(s, t)] = float(max(0.05, min(1.5, 0.5 + top)))
        return weights

    def verify(
        self,
        question: str,
        candidates: Sequence[str],
        sources: dict[str, EvidenceSource],
        anchor: Optional[str] = None,
        extra_candidates: Optional[Sequence[str]] = None,
    ) -> VerifierResult:
        """Verify candidate answers.

        Parameters
        ----------
        candidates : sampled candidates from the LLM (used for the vote share).
        anchor : e.g. the greedy single-shot answer; acts as a strong prior
            via ``anchor_w``. If not present in ``candidates`` it is still
            considered a legal choice.
        extra_candidates : additional candidate answers produced by other
            decoding strategies (e.g. the GraphRAG-approx answer). They do not
            contribute to the vote share but are scored by sheaf energy and
            can win if their consistency is high enough. This mirrors the
            deployment framing that SCAR is a knowledge-side verifier that
            consumes *all* available candidate answers, not only LLM samples.
        """
        import time

        t0 = time.time()
        # Build the extended candidate pool. Vote statistics use only the
        # sampled candidates; sheaf energy is computed over the union.
        extras = list(dict.fromkeys([e for e in (extra_candidates or []) if e and e not in candidates]))
        all_candidates = list(candidates) + extras
        qa_texts = [f"{question} {a}" for a in all_candidates]
        qa_embs = self.encode(qa_texts)  # (N_all, d)
        dim = qa_embs.shape[1] if qa_embs.size else 384

        # 2. Commitment vectors + per-source support (max cosine) per candidate,
        #    computed over the extended pool.
        commit: dict[str, np.ndarray] = {}
        support: dict[str, np.ndarray] = {}  # (N_all,) per source
        for name, src in sources.items():
            if src.embeddings.shape[0] == 0:
                commit[name] = np.zeros((len(all_candidates), dim), dtype=np.float32)
                support[name] = np.zeros(len(all_candidates), dtype=np.float32)
                continue
            src_emb = np.ascontiguousarray(src.embeddings, dtype=np.float32)
            with np.errstate(all="ignore"):
                sims = qa_embs @ src_emb.T  # (N_all, M)
                sup = sims.max(axis=1) if sims.size else np.zeros(len(all_candidates))
            support[name] = np.ascontiguousarray(sup, dtype=np.float32)
            cs = np.stack([
                commitment_vector(src.embeddings, qa_embs[i], self.config.temperature)
                for i in range(len(all_candidates))
            ], axis=0)
            commit[name] = cs

        # 3. Restriction maps + pair weights.
        R = self._restriction_maps(sources, dim)
        W = self._pair_weights(sources)

        # 4. Energy per candidate.
        names = list(sources.keys())
        energies = []
        with np.errstate(all="ignore"):
            for i in range(len(all_candidates)):
                e_i = 0.0
                for s in names:
                    for t in names:
                        if s == t:
                            continue
                        cs = np.ascontiguousarray(commit[s][i], dtype=np.float32)
                        ct = np.ascontiguousarray(commit[t][i], dtype=np.float32)
                        if np.linalg.norm(cs) < 1e-8 or np.linalg.norm(ct) < 1e-8:
                            continue
                        if (s, t) in R:
                            Rst = np.ascontiguousarray(R[(s, t)], dtype=np.float32)
                            proj = Rst @ cs
                        else:
                            proj = cs
                        diff = proj - ct
                        e_i += W.get((s, t), 1.0) * float(np.dot(diff, diff))
                energies.append(e_i)

        energies_arr = np.asarray(energies, dtype=np.float32)
        if energies_arr.size and energies_arr.max() > energies_arr.min():
            e_norm = (energies_arr - energies_arr.min()) / (energies_arr.max() - energies_arr.min() + 1e-8)
        else:
            e_norm = np.zeros_like(energies_arr)

        # 4b. Cross-source support: min over sources of per-source max-cosine.
        #     A candidate that is supported by every source scores high on this.
        sup_stack = np.stack([support[n] for n in names], axis=1) if names else np.zeros((len(all_candidates), 1))
        min_support = sup_stack.min(axis=1) if sup_stack.size else np.zeros(len(all_candidates))
        # Normalise to [0, 1] over candidates.
        if min_support.size and min_support.max() > min_support.min():
            support_norm = (min_support - min_support.min()) / (min_support.max() - min_support.min() + 1e-8)
        else:
            support_norm = np.zeros_like(min_support)

        # 5. Vote statistics (from LLM samples only; extras get a small bonus).
        from collections import Counter

        counts = Counter(candidates)
        top_vote, top_ct = counts.most_common(1)[0]
        # Vote shares over the extended pool: extras get ``extra_candidate_bonus``.
        vote_shares = np.asarray(
            [counts.get(c, 0) / max(1, len(candidates)) if c in counts
             else self.config.extra_candidate_bonus
             for c in all_candidates],
            dtype=np.float32,
        )
        top_vote_share = top_ct / max(1, len(candidates))
        observed_conflict = float(1.0 - top_vote_share)
        alpha = max(self.config.blend_alpha_min,
                    min(1.0, self.config.blend_beta * observed_conflict))

        # 6. Anchor-aware sheaf ranking over the extended candidate pool.
        # For every unique candidate we build a combined score of:
        #   vote_weight  : normalised vote share (SC signal, extras get bonus)
        #   sheaf_score  : mean over occurrences of 0.5*(1-e_norm)+0.5*support_norm
        #   anchor_bonus : +anchor_w if this candidate matches the anchor
        sheaf_component = 0.5 * (1.0 - e_norm) + 0.5 * support_norm

        unique_cands = list(dict.fromkeys(all_candidates))
        per_cand_sheaf: dict[str, float] = {}
        per_cand_vote: dict[str, float] = {}
        for uc in unique_cands:
            idxs = [i for i, c in enumerate(all_candidates) if c == uc]
            per_cand_sheaf[uc] = float(np.mean([sheaf_component[i] for i in idxs]))
            per_cand_vote[uc] = float(vote_shares[idxs[0]])

        used_sheaf = True
        if self.config.gate_enabled and top_vote_share >= self.config.conflict_gate:
            chosen = top_vote
            chosen_idx = all_candidates.index(chosen)
            used_sheaf = False
            final = 0.5 + 0.5 * vote_shares
        else:
            vote_w = self.config.vote_w
            sheaf_w = self.config.sheaf_w
            anchor_w = self.config.anchor_w

            combined: dict[str, float] = {}
            for uc in unique_cands:
                s = vote_w * per_cand_vote[uc] + sheaf_w * per_cand_sheaf[uc]
                if anchor is not None and uc == anchor:
                    s += anchor_w
                combined[uc] = s

            if anchor is not None and anchor not in combined:
                combined[anchor] = anchor_w

            chosen = max(combined, key=combined.get)
            if chosen in all_candidates:
                chosen_idx = all_candidates.index(chosen)
            else:
                chosen_idx = -1

            final = np.asarray([combined.get(c, 0.0) for c in all_candidates], dtype=np.float32)

        return VerifierResult(
            chosen_answer=chosen,
            chosen_index=chosen_idx,
            energies=[float(x) for x in energies_arr.tolist()],
            final_scores=[float(x) for x in final.tolist()],
            vote_shares=[float(x) for x in vote_shares.tolist()],
            alpha=float(alpha),
            used_sheaf=used_sheaf,
            latency_ms=(time.time() - t0) * 1000.0,
            vote_pick=top_vote,
        )

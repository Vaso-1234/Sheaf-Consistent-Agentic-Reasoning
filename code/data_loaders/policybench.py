"""PolicyBench-Synth: a controlled decision-support benchmark.

Motivation. KBS papers rate a knowledge component on how well it handles
conflicting evidence in a decision-support setting. Multi-hop QA does not
give us principled control over conflict. PolicyBench-Synth does.

Construction. We define 10 policy domains, each with 5 numeric-parameter
templates (e.g. "Data classified as Confidential must be retained for X days").
For each of 500 questions, we sample a policy, produce the ground-truth
answer, and assemble 10 passages: two gold (paraphrased canonical text of
the correct policy), six same-domain distractors (different policies from
the same domain), and two out-of-domain distractors. If a conflict rate r
is specified, floor(r * 10) of the gold or same-domain passages are
replaced by a "poisoned" paraphrase whose numeric parameter has been
perturbed.

The whole benchmark is deterministic given the seed; SHA256 is logged
alongside the loaded split.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Optional

from .schema import Example, Passage


DOMAINS: dict[str, list[tuple[str, str, str, tuple[int, int, str]]]] = {
    "data_retention": [
        ("public_records",
         "Records classified as Public may be retained indefinitely and must be reviewed every {V} months.",
         "How often must Public-classified records be reviewed at {org}?",
         (12, 36, "months")),
        ("confidential_records",
         "Records classified as Confidential must be retained for exactly {V} days before secure deletion.",
         "For how many days must Confidential records be retained at {org}?",
         (30, 365, "days")),
        ("audit_logs",
         "Security audit logs at {org} must be preserved for at least {V} months in immutable storage.",
         "What is the minimum retention period for security audit logs at {org}?",
         (6, 84, "months")),
        ("employee_pii",
         "Employee PII may be retained for no more than {V} days after termination of employment.",
         "How long after termination may employee PII be retained at {org}?",
         (30, 730, "days")),
        ("customer_transactions",
         "Customer transaction data must be retained for a minimum of {V} years for regulatory compliance.",
         "What is the minimum retention period for customer transaction data at {org}?",
         (3, 15, "years")),
    ],
    "access_control": [
        ("failed_logins",
         "User accounts must be locked after {V} consecutive failed login attempts.",
         "After how many failed logins is a user account locked at {org}?",
         (3, 15, "attempts")),
        ("password_length",
         "System passwords must contain a minimum of {V} characters.",
         "What is the minimum password length required at {org}?",
         (8, 32, "characters")),
        ("session_timeout",
         "Idle authenticated sessions expire after {V} minutes.",
         "After how many minutes of inactivity do sessions expire at {org}?",
         (5, 120, "minutes")),
        ("mfa_grace",
         "New privileged users have {V} days to enroll in multi-factor authentication.",
         "How many days do new privileged users have to enroll in MFA at {org}?",
         (1, 30, "days")),
        ("revocation",
         "Access credentials for departing staff must be revoked within {V} hours of termination.",
         "Within how many hours must access be revoked for departing staff at {org}?",
         (1, 72, "hours")),
    ],
    "incident_response": [
        ("severity_one",
         "Severity-1 incidents at {org} must be acknowledged within {V} minutes.",
         "What is the maximum acknowledgement time for a Severity-1 incident at {org}?",
         (5, 60, "minutes")),
        ("post_mortem",
         "A written post-mortem is required within {V} business days of a major incident.",
         "Within how many business days must a post-mortem be filed after a major incident at {org}?",
         (3, 30, "business days")),
        ("breach_notification",
         "Regulatory breach notification must be filed within {V} hours of confirmed exposure.",
         "Within how many hours must a breach be reported to regulators at {org}?",
         (24, 168, "hours")),
        ("evidence_retention",
         "Incident forensic evidence must be preserved for {V} months.",
         "For how many months must forensic evidence from incidents be retained at {org}?",
         (6, 60, "months")),
        ("on_call_paging",
         "On-call engineers must respond to a page within {V} minutes.",
         "Within how many minutes must an on-call engineer respond to a page at {org}?",
         (5, 45, "minutes")),
    ],
    "vendor_management": [
        ("security_review",
         "Third-party vendors require a security review renewed every {V} months.",
         "How often are third-party vendor security reviews renewed at {org}?",
         (6, 36, "months")),
        ("contract_notice",
         "Termination of a vendor contract requires {V} days written notice.",
         "How many days of written notice are required to terminate a vendor contract at {org}?",
         (14, 180, "days")),
        ("data_processor_audit",
         "Data processors must be audited at least once every {V} months.",
         "How often at minimum must data processors be audited at {org}?",
         (6, 24, "months")),
        ("sla_uptime",
         "Vendor systems must meet a minimum uptime of {V} percent per calendar month.",
         "What is the minimum monthly uptime required from vendors at {org}?",
         (95, 100, "percent")),
        ("renewal_window",
         "Renewal decisions must be finalised at least {V} days before contract end.",
         "How many days before contract end must renewal decisions be finalised at {org}?",
         (14, 120, "days")),
    ],
    "licensing": [
        ("open_source_review",
         "New open-source dependencies require licence review within {V} business days of introduction.",
         "Within how many business days must new open-source dependencies be reviewed at {org}?",
         (1, 30, "business days")),
        ("copyleft_exception",
         "Copyleft licences require executive approval when used in more than {V} percent of shipped code.",
         "Above what percentage of shipped code do copyleft licences require executive approval at {org}?",
         (1, 25, "percent")),
        ("attribution_delay",
         "Attribution notices for third-party components must ship within {V} days of a release.",
         "Within how many days of a release must third-party attribution notices ship at {org}?",
         (7, 90, "days")),
        ("licence_audit",
         "Full licence audits occur every {V} months.",
         "How often are full licence audits conducted at {org}?",
         (6, 24, "months")),
        ("trademark_reuse",
         "External reuse of company trademarks requires legal approval at least {V} business days ahead.",
         "How many business days ahead is legal approval required for external trademark reuse at {org}?",
         (5, 45, "business days")),
    ],
    "safety": [
        ("emergency_drill",
         "Full emergency evacuation drills are held every {V} months.",
         "How often are full emergency evacuation drills held at {org}?",
         (3, 24, "months")),
        ("ergonomics",
         "Ergonomics assessments are offered every {V} months to desk-based staff.",
         "How often are ergonomics assessments offered to desk-based staff at {org}?",
         (6, 36, "months")),
        ("first_aid_kits",
         "First-aid kits must be inspected every {V} days.",
         "How often must first-aid kits be inspected at {org}?",
         (7, 90, "days")),
        ("chemical_storage",
         "Chemical storage inventory is audited every {V} weeks.",
         "How often is the chemical storage inventory audited at {org}?",
         (2, 26, "weeks")),
        ("training_refresher",
         "Safety training refreshers are required every {V} months.",
         "How often are safety training refreshers required at {org}?",
         (6, 36, "months")),
    ],
    "hr": [
        ("annual_leave",
         "Full-time employees accrue {V} days of annual leave per year.",
         "How many days of annual leave do full-time employees accrue per year at {org}?",
         (14, 40, "days")),
        ("probation",
         "New hires are subject to a probation period of {V} months.",
         "What is the probation period for new hires at {org}?",
         (1, 12, "months")),
        ("performance_review",
         "Performance reviews are conducted every {V} months for all staff.",
         "How often are performance reviews conducted at {org}?",
         (3, 24, "months")),
        ("notice_period",
         "Voluntary resignation requires {V} weeks of notice.",
         "How many weeks of notice are required for voluntary resignation at {org}?",
         (2, 26, "weeks")),
        ("training_budget",
         "Individual training budgets of {V} dollars are allocated per employee per year.",
         "What annual training budget is allocated per employee at {org}?",
         (500, 5000, "dollars")),
    ],
    "finance": [
        ("expense_reimbursement",
         "Employee expenses under {V} dollars are auto-approved without receipts.",
         "Below what dollar amount are employee expenses auto-approved without receipts at {org}?",
         (25, 500, "dollars")),
        ("purchase_order",
         "Purchase orders exceeding {V} dollars require finance director approval.",
         "Above what dollar amount do purchase orders require finance director approval at {org}?",
         (1000, 50000, "dollars")),
        ("invoice_payment",
         "Approved supplier invoices are paid within {V} business days.",
         "Within how many business days are approved supplier invoices paid at {org}?",
         (7, 60, "business days")),
        ("travel_advance",
         "Cash travel advances above {V} dollars require pre-approval.",
         "Above what dollar amount do cash travel advances require pre-approval at {org}?",
         (100, 2000, "dollars")),
        ("credit_limit",
         "Departmental corporate cards carry a default monthly limit of {V} dollars.",
         "What is the default monthly credit limit on departmental corporate cards at {org}?",
         (1000, 20000, "dollars")),
    ],
    "cloud_ops": [
        ("backup_frequency",
         "Production database backups run every {V} hours.",
         "How often do production database backups run at {org}?",
         (1, 24, "hours")),
        ("rto",
         "Recovery time objective for tier-1 services is {V} minutes.",
         "What is the recovery time objective for tier-1 services at {org}?",
         (5, 240, "minutes")),
        ("rpo",
         "Recovery point objective for customer data is {V} minutes.",
         "What is the recovery point objective for customer data at {org}?",
         (5, 120, "minutes")),
        ("patch_window",
         "Non-critical security patches are applied within {V} days of release.",
         "Within how many days are non-critical security patches applied at {org}?",
         (7, 60, "days")),
        ("dr_test",
         "Full disaster recovery drills are executed every {V} months.",
         "How often are full disaster recovery drills executed at {org}?",
         (3, 24, "months")),
    ],
    "research_ethics": [
        ("irb_response",
         "Institutional review board responses are issued within {V} business days.",
         "Within how many business days are IRB responses issued at {org}?",
         (5, 45, "business days")),
        ("consent_expiry",
         "Participant consent forms expire and require re-signing every {V} months.",
         "How often must participant consent forms be re-signed at {org}?",
         (6, 36, "months")),
        ("data_share",
         "External data-sharing agreements must be renewed every {V} months.",
         "How often are external data-sharing agreements renewed at {org}?",
         (6, 36, "months")),
        ("anonymisation_review",
         "Anonymisation methods are re-evaluated every {V} months.",
         "How often are anonymisation methods re-evaluated at {org}?",
         (6, 36, "months")),
        ("open_data_delay",
         "Datasets underlying published research are released within {V} months of publication.",
         "Within how many months of publication are underlying research datasets released at {org}?",
         (0, 24, "months")),
    ],
}


ORGS = [
    "Bluewave Holdings", "Aster Systems", "Northlake Analytics",
    "Meridian Robotics", "Ember Health", "Cinderwood Labs",
    "Quorum Networks", "Silvernote Bank", "Fernpath Foundation", "Ridgemark Consulting",
]


PARAPHRASE_TEMPLATES = {
    "data_retention": [
        "The {org} retention policy states: {text}",
        "As part of the {org} data governance handbook, {text}",
        "Per {org} policy DR-{code}, {text}",
    ],
    "access_control": [
        "According to the {org} access control standard AC-{code}, {text}",
        "The {org} authentication guideline notes that {text}",
        "Under section AC-{code} of the {org} handbook, {text}",
    ],
    "incident_response": [
        "The {org} incident response runbook (IR-{code}) records that {text}",
        "Per {org} on-call policy IR-{code}, {text}",
        "The {org} security operations centre documents: {text}",
    ],
    "vendor_management": [
        "The {org} vendor management policy VM-{code} states: {text}",
        "In the {org} supplier handbook (section VM-{code}), {text}",
        "According to the {org} third-party governance charter, {text}",
    ],
    "licensing": [
        "The {org} legal handbook (LIC-{code}) states: {text}",
        "Per {org} open source policy LIC-{code}, {text}",
        "According to the {org} intellectual property guidelines, {text}",
    ],
    "safety": [
        "The {org} workplace safety manual (SAF-{code}) records: {text}",
        "Per {org} emergency preparedness policy SAF-{code}, {text}",
        "The {org} facilities safety handbook states: {text}",
    ],
    "hr": [
        "The {org} employee handbook (HR-{code}) says: {text}",
        "Per {org} people operations policy HR-{code}, {text}",
        "The {org} human resources charter records that {text}",
    ],
    "finance": [
        "The {org} finance policy FIN-{code} states: {text}",
        "Per {org} controller's handbook (FIN-{code}), {text}",
        "According to the {org} spend management guideline, {text}",
    ],
    "cloud_ops": [
        "The {org} reliability policy OPS-{code} says: {text}",
        "Per {org} platform operations handbook (OPS-{code}), {text}",
        "The {org} site reliability charter documents: {text}",
    ],
    "research_ethics": [
        "The {org} research ethics manual (RE-{code}) states: {text}",
        "Per {org} data ethics policy RE-{code}, {text}",
        "According to the {org} research governance handbook, {text}",
    ],
}


def _rng_perturb(v: int, low: int, high: int, rng: random.Random) -> int:
    """Perturb a numeric value within its plausible range but distinct from v."""
    for _ in range(20):
        span = max(1, (high - low) // 3)
        delta = rng.randint(-span, span)
        cand = max(low, min(high, v + delta))
        if cand != v:
            return cand
    return low if v != low else high


def _paraphrase(domain: str, template: str, org: str, code: str, rng: random.Random) -> str:
    wrappers = PARAPHRASE_TEMPLATES[domain]
    wrapper = rng.choice(wrappers)
    body = template
    if body.endswith("."):
        body = body[:-1]
    body = body[0].lower() + body[1:] if body[:1].isupper() and not body.startswith("Severity") else body
    return wrapper.format(org=org, code=code, text=body) + "."


def _fill_template(template: str, org: str, value: int) -> str:
    return template.format(org=org, V=value)


def build_policybench(n: int = 500, seed: int = 42, conflict_rate: float = 0.0) -> list[Example]:
    """Generate n examples. conflict_rate in [0,1] specifies fraction of the 10 passages that are poisoned."""

    rng = random.Random(seed)
    domain_names = list(DOMAINS.keys())
    # Assemble a flat list of (domain, template_id, template, question, val_range)
    flat = []
    for dname, templates in DOMAINS.items():
        for tid, (name, ttext, qtext, vrange) in enumerate(templates):
            flat.append((dname, tid, name, ttext, qtext, vrange))

    out: list[Example] = []
    for i in range(n):
        dname, tid, name, ttext, qtext, (low, high, unit) = rng.choice(flat)
        org = rng.choice(ORGS)
        value = rng.randint(low, high)
        answer = f"{value} {unit}"
        question = qtext.format(org=org)

        code = f"{dname[:3].upper()}-{tid+1:02d}-{rng.randint(10,99)}"
        gold_text_a = _paraphrase(dname, _fill_template(ttext, org, value), org, code, rng)
        gold_text_b = _paraphrase(dname, _fill_template(ttext, org, value), org, code, rng)
        passages: list[Passage] = [
            Passage(title=f"{org} policy {code} (rev A)", text=gold_text_a, is_gold=True),
            Passage(title=f"{org} policy {code} (rev B)", text=gold_text_b, is_gold=True),
        ]

        # Six same-domain distractors (other policies in same domain, other values)
        others_same = [t for t in DOMAINS[dname] if t[0] != name]
        rng.shuffle(others_same)
        for j in range(min(6, len(others_same))):
            oname, otext, _, (olow, ohigh, ounit) = others_same[j]
            ov = rng.randint(olow, ohigh)
            other_org = rng.choice(ORGS)
            code2 = f"{dname[:3].upper()}-D{j+1:02d}-{rng.randint(10,99)}"
            text = _paraphrase(dname, _fill_template(otext, other_org, ov), other_org, code2, rng)
            passages.append(Passage(title=f"{other_org} policy {code2}", text=text, is_gold=False))

        # Two out-of-domain distractors
        other_domains = [d for d in domain_names if d != dname]
        for _ in range(2):
            od = rng.choice(other_domains)
            oname2, otext2, _, (olow2, ohigh2, ounit2) = rng.choice(DOMAINS[od])
            other_org2 = rng.choice(ORGS)
            ov2 = rng.randint(olow2, ohigh2)
            code3 = f"{od[:3].upper()}-X{rng.randint(10,99)}"
            text = _paraphrase(od, _fill_template(otext2, other_org2, ov2), other_org2, code3, rng)
            passages.append(Passage(title=f"{other_org2} policy {code3}", text=text, is_gold=False))

        # Conflict injection: replace floor(conflict_rate * 10) of the 10 passages
        # with "poisoned" paraphrases whose numeric value has been perturbed.
        n_conflict = int(round(conflict_rate * len(passages)))
        if n_conflict > 0:
            idxs = list(range(len(passages)))
            rng.shuffle(idxs)
            for k in idxs[:n_conflict]:
                p = passages[k]
                new_value = _rng_perturb(value if p.is_gold else rng.randint(low, high), low, high, rng)
                poisoned_text = _paraphrase(
                    dname, _fill_template(ttext, org if p.is_gold else rng.choice(ORGS), new_value),
                    org if p.is_gold else rng.choice(ORGS),
                    f"{dname[:3].upper()}-P{k:02d}-{rng.randint(10,99)}",
                    rng,
                )
                # Poisoned passage keeps the same title so it looks like a conflicting version.
                passages[k] = Passage(title=p.title + " (mirror)", text=poisoned_text, is_gold=False)

        rng.shuffle(passages)
        supporting_titles = sorted({p.title for p in passages if p.is_gold})
        ex = Example(
            qid=f"pb_{i:05d}",
            question=question,
            answer=answer,
            answer_aliases=[str(value), f"{value}", answer.lower()],
            passages=passages,
            supporting_titles=supporting_titles,
            dataset="policybench",
            qtype=dname,
        )
        out.append(ex)
    return out


def build_and_cache(n: int = 500, seed: int = 42, conflict_rate: float = 0.0, out_dir: Optional[Path] = None) -> Path:
    from ..common import POLICY_DIR
    out_dir = out_dir or POLICY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ex = build_policybench(n=n, seed=seed, conflict_rate=conflict_rate)
    from .schema import example_to_dict
    dumped = [example_to_dict(x) for x in ex]
    payload = json.dumps(dumped, indent=None)
    h = hashlib.sha256(payload.encode()).hexdigest()[:12]
    fname = f"policybench_n{n}_seed{seed}_cr{int(conflict_rate*100):02d}_{h}.json"
    path = out_dir / fname
    path.write_text(payload)
    return path


if __name__ == "__main__":
    xs = build_policybench(5, seed=42, conflict_rate=0.25)
    for e in xs:
        print(e.qid, "|", e.question)
        print("  ans:", e.answer)
        for p in e.passages:
            marker = "GOLD " if p.is_gold else "     "
            print(f"  {marker}{p.title}: {p.text[:110]}")
        print()

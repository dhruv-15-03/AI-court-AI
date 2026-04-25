"""
Elite Legal Intelligence Modules
==================================

Three advanced modules that make the AI Lawyer fight at an elite level:

1. JudgePatternAnalyzer — Analyzes judge-specific ruling patterns from case history
2. EvidenceAdmissibilityChecker — Validates evidence against Indian law rules
3. ProceduralChecklist — Per-case-type procedural requirements & deadlines

These are wired into the agent pipeline to provide extra intelligence
beyond the 3 core pillars (laws + case facts + precedents).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. JUDGE PATTERN ANALYZER
# ═══════════════════════════════════════════════════════════════════════════

class JudgePatternAnalyzer:
    """
    Analyzes judge-specific ruling patterns from the case database.
    
    Uses the 762K case corpus to extract:
    - Conviction rate per judge
    - Bail grant tendency
    - Average sentence severity
    - Case type specialization
    
    This gives the AI Lawyer insight into how a specific judge tends to rule,
    similar to what experienced lawyers know from courtroom experience.
    """

    def __init__(self, search_index=None):
        self.search_index = search_index
        self._judge_cache: Dict[str, Dict] = {}

    def analyze_judge(self, judge_name: str, court: str = "") -> Dict[str, Any]:
        """
        Analyze a judge's ruling patterns from the case database.
        
        Args:
            judge_name: Name of the judge (e.g., "Justice D.Y. Chandrachud")
            court: Court name filter (optional)
        
        Returns:
            Dict with judge's patterns, tendencies, and statistics.
        """
        cache_key = f"{judge_name}:{court}"
        if cache_key in self._judge_cache:
            return self._judge_cache[cache_key]

        if not self.search_index:
            return {"judge": judge_name, "available": False, "message": "Search index not loaded"}

        # Search for cases by this judge
        try:
            import numpy as np
            vectorizer = self.search_index.get("vectorizer")
            matrix = self.search_index.get("matrix")
            meta = self.search_index.get("meta", [])

            if vectorizer is None or matrix is None:
                return {"judge": judge_name, "available": False}

            query = f"Justice {judge_name} {court}".strip()
            qv = vectorizer.transform([query.lower()])
            scores = (matrix @ qv.T).toarray().ravel()
            top_idx = np.argsort(-scores)[:100]  # Top 100 matches

            # Analyze outcomes
            outcomes = {}
            case_types = {}
            for idx in top_idx:
                if idx < len(meta) and scores[idx] > 0.1:
                    m = meta[idx]
                    outcome = m.get("outcome", "Unknown")
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1

            total = sum(outcomes.values())
            if total < 3:
                return {"judge": judge_name, "available": False, "message": "Insufficient data"}

            # Calculate tendencies
            conviction_related = sum(v for k, v in outcomes.items()
                                     if "convict" in k.lower() or "upheld" in k.lower())
            acquittal_related = sum(v for k, v in outcomes.items()
                                    if "acquit" in k.lower() or "overturned" in k.lower())
            bail_granted = sum(v for k, v in outcomes.items()
                              if "bail granted" in k.lower())
            bail_denied = sum(v for k, v in outcomes.items()
                             if "bail denied" in k.lower() or "bail rejected" in k.lower())

            result = {
                "judge": judge_name,
                "court": court,
                "available": True,
                "total_cases_found": total,
                "outcome_distribution": outcomes,
                "tendencies": {
                    "conviction_rate": round(conviction_related / max(total, 1), 2),
                    "acquittal_rate": round(acquittal_related / max(total, 1), 2),
                    "bail_grant_rate": round(bail_granted / max(bail_granted + bail_denied, 1), 2) if (bail_granted + bail_denied) > 0 else None,
                },
                "insight": self._generate_insight(outcomes, total),
            }

            self._judge_cache[cache_key] = result
            return result

        except Exception as e:
            logger.warning(f"Judge analysis failed: {e}")
            return {"judge": judge_name, "available": False, "error": str(e)}

    def _generate_insight(self, outcomes: Dict[str, int], total: int) -> str:
        """Generate a human-readable insight about the judge's tendencies."""
        if total < 5:
            return "Insufficient data for reliable pattern analysis."

        top_outcome = max(outcomes, key=outcomes.get)
        top_pct = outcomes[top_outcome] / total * 100

        return (
            f"Based on {total} analyzed cases, the most common outcome is "
            f"'{top_outcome}' ({top_pct:.0f}%). "
        )

    def format_for_context(self, analysis: Dict[str, Any]) -> str:
        """Format judge analysis for inclusion in LLM prompt."""
        if not analysis.get("available"):
            return ""

        lines = [f"[Judge Pattern Analysis: {analysis['judge']}]"]
        lines.append(f"Cases analyzed: {analysis['total_cases_found']}")

        tend = analysis.get("tendencies", {})
        if tend.get("conviction_rate") is not None:
            lines.append(f"Conviction tendency: {tend['conviction_rate']:.0%}")
        if tend.get("acquittal_rate") is not None:
            lines.append(f"Acquittal tendency: {tend['acquittal_rate']:.0%}")
        if tend.get("bail_grant_rate") is not None:
            lines.append(f"Bail grant rate: {tend['bail_grant_rate']:.0%}")

        insight = analysis.get("insight", "")
        if insight:
            lines.append(f"Insight: {insight}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 2. EVIDENCE ADMISSIBILITY CHECKER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EvidenceIssue:
    """An issue found with a piece of evidence."""
    evidence_type: str
    issue: str
    severity: str  # "critical", "warning", "info"
    legal_basis: str  # Section reference
    recommendation: str


class EvidenceAdmissibilityChecker:
    """
    Validates evidence against Indian evidence law rules.
    
    Checks common admissibility issues that can make or break a case:
    - Electronic evidence (65B/BSA 63 certificate requirement)
    - Confessions to police (inadmissible under 25-26 Evidence Act / BSA 23-24)
    - Dying declarations (conditions for validity)
    - Hearsay rules
    - Expert evidence requirements
    - Chain of custody for physical evidence
    - Witness competency
    """

    # Evidence rules — each rule checks for a specific issue
    RULES = [
        {
            "id": "electronic_evidence_cert",
            "triggers": ["whatsapp", "email", "sms", "screenshot", "cctv", "video", "audio recording",
                         "digital", "electronic", "computer", "phone record", "call record", "cdr"],
            "severity": "critical",
            "issue": "Electronic evidence requires a Section 65B certificate (Indian Evidence Act) / Section 63 certificate (BSA) for admissibility. Without this certificate, electronic evidence is INADMISSIBLE.",
            "legal_basis": "Section 65B Indian Evidence Act / Section 63 Bharatiya Sakshya Adhiniyam",
            "recommendation": "Obtain a Section 65B/63 BSA certificate from the person in charge of the device/system that generated the electronic record. The certificate must identify the electronic record, describe the device, and certify the record's integrity.",
        },
        {
            "id": "confession_to_police",
            "triggers": ["confessed to police", "confession before police", "statement to police",
                         "police recorded confession", "admitted to inspector", "told the police"],
            "severity": "critical",
            "issue": "Confession made to a police officer is NOT admissible as evidence. Only confessions before a Magistrate (under Section 164 CrPC / BNSS 183) are admissible.",
            "legal_basis": "Section 25-26 Indian Evidence Act / Section 23-24 BSA; Section 164 CrPC / Section 183 BNSS",
            "recommendation": "If the accused made a statement to police, it cannot be used as a confession. However, a 'discovery statement' leading to recovery of evidence IS admissible under Section 27 Evidence Act / Section 25 BSA.",
        },
        {
            "id": "dying_declaration",
            "triggers": ["dying declaration", "statement before death", "last statement", "death bed statement"],
            "severity": "warning",
            "issue": "Dying declaration is admissible without oath if the person was in expectation of death and the statement relates to the cause of death or circumstances leading to it.",
            "legal_basis": "Section 32(1) Indian Evidence Act / Section 26 BSA",
            "recommendation": "Verify: (1) declarant was conscious and competent, (2) was under expectation of death, (3) statement relates to cause/circumstances of death. Preferably recorded by a Magistrate with doctor's fitness certificate.",
        },
        {
            "id": "hearsay",
            "triggers": ["told me that", "heard from", "someone said", "rumor", "informed by another",
                         "third party told", "was told by"],
            "severity": "warning",
            "issue": "Hearsay evidence (what someone else told the witness) is generally not admissible. Only direct knowledge is admissible as oral evidence.",
            "legal_basis": "Section 60 Indian Evidence Act / Section 56 BSA",
            "recommendation": "Produce the original source as a witness. Hearsay is only admissible under specific exceptions (dying declaration, entries in course of business, public documents, etc.).",
        },
        {
            "id": "expert_opinion",
            "triggers": ["medical report", "forensic report", "handwriting expert", "ballistic report",
                         "dna report", "chemical analysis", "post mortem", "autopsy"],
            "severity": "info",
            "issue": "Expert opinion evidence requires the expert to appear in court for cross-examination. A report alone without the expert's testimony has reduced evidentiary value.",
            "legal_basis": "Section 45-51 Indian Evidence Act / Section 39-45 BSA",
            "recommendation": "Ensure the expert is available for cross-examination. The court is not bound by expert opinion but gives it due weight.",
        },
        {
            "id": "recovery_panchnama",
            "triggers": ["recovered", "seizure", "recovery", "panchnama", "search", "weapon found",
                         "drugs found", "seized", "contraband"],
            "severity": "warning",
            "issue": "Recovery/seizure evidence requires a proper panchnama (attestation by independent witnesses). Recovery without independent panchnama witnesses is suspect.",
            "legal_basis": "Section 100 CrPC / Section 105 BNSS; Section 27 Evidence Act / Section 25 BSA",
            "recommendation": "Verify: (1) independent panchnama witnesses present, (2) seizure memo prepared at the spot, (3) proper chain of custody maintained, (4) FSL report obtained if applicable.",
        },
        {
            "id": "identification_parade",
            "triggers": ["identification", "test identification", "TIP", "identified in court",
                         "identified by witness", "pointed out"],
            "severity": "warning",
            "issue": "Court identification without prior Test Identification Parade (TIP) has reduced evidentiary value, especially when the accused was unknown to the witness before the incident.",
            "legal_basis": "Section 9 Indian Evidence Act / Section 8 BSA",
            "recommendation": "If the accused was a stranger, a TIP should have been conducted in jail under Magistrate supervision before the trial. Dock identification alone (pointing out in court) is weak evidence.",
        },
        {
            "id": "medical_evidence_delay",
            "triggers": ["medical examination", "medical report", "injury report", "mlc"],
            "severity": "info",
            "issue": "Delayed medical examination reduces the evidentiary value. Immediate medical examination after the incident strengthens the prosecution's case.",
            "legal_basis": "Section 53, 53A, 164A CrPC / Section 184 BNSS",
            "recommendation": "Check the time gap between incident and medical examination. Significant delay without explanation weakens the evidence.",
        },
    ]

    def check(self, case_text: str, evidence_descriptions: List[str] = None) -> List[EvidenceIssue]:
        """
        Check evidence in a case for admissibility issues.
        
        Args:
            case_text: Full case description or document text.
            evidence_descriptions: List of individual evidence item descriptions.
        
        Returns:
            List of EvidenceIssue objects with findings.
        """
        all_text = case_text.lower()
        if evidence_descriptions:
            all_text += " " + " ".join(e.lower() for e in evidence_descriptions)

        issues = []
        for rule in self.RULES:
            triggered = any(trigger in all_text for trigger in rule["triggers"])
            if triggered:
                issues.append(EvidenceIssue(
                    evidence_type=rule["id"],
                    issue=rule["issue"],
                    severity=rule["severity"],
                    legal_basis=rule["legal_basis"],
                    recommendation=rule["recommendation"],
                ))

        return issues

    def format_for_context(self, issues: List[EvidenceIssue]) -> str:
        """Format evidence issues for inclusion in LLM prompt."""
        if not issues:
            return ""

        lines = ["[Evidence Admissibility Alerts]"]
        for issue in issues:
            severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "•")
            lines.append(f"\n{severity_icon} {issue.evidence_type.upper()} [{issue.severity}]")
            lines.append(f"  Issue: {issue.issue}")
            lines.append(f"  Law: {issue.legal_basis}")
            lines.append(f"  Action: {issue.recommendation}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PROCEDURAL CHECKLISTS
# ═══════════════════════════════════════════════════════════════════════════

PROCEDURAL_CHECKLISTS = {
    "bail_application": {
        "title": "Bail Application Checklist",
        "case_types": ["criminal", "murder", "narcotics", "assault"],
        "steps": [
            {"step": "Determine bail type", "detail": "Regular bail (S.437/439 CrPC or S.478/479 BNSS) vs Anticipatory bail (S.438 CrPC or S.482 BNSS) vs Default bail (S.167(2) CrPC or S.187(2) BNSS)", "deadline": "Before arrest for anticipatory; anytime after arrest for regular"},
            {"step": "Check if offence is bailable or non-bailable", "detail": "Bailable offences: bail as of right. Non-bailable: court discretion", "deadline": "Immediate"},
            {"step": "Verify FIR/complaint copy", "detail": "Obtain copy of FIR, arrest memo, remand order", "deadline": "Within 24 hours of arrest"},
            {"step": "Check default bail eligibility", "detail": "If investigation not completed within 60 days (up to 7 years punishment) or 90 days (above 7 years), default bail is a RIGHT", "deadline": "On 61st/91st day"},
            {"step": "File bail application", "detail": "Draft with grounds: no flight risk, cooperation, medical grounds, long incarceration, weak prima facie case", "deadline": "ASAP after arrest"},
            {"step": "Attach documents", "detail": "ID proof, address proof, surety bond, FIR copy, medical reports if any, previous bail orders if any", "deadline": "At time of filing"},
            {"step": "Court hearing", "detail": "Present arguments on bail. Prosecution gets notice (usually). For anticipatory bail, presence may not be required", "deadline": "As per court schedule"},
            {"step": "Bail conditions compliance", "detail": "If granted: execute bail bond, provide sureties, surrender passport if ordered, mark attendance", "deadline": "Immediately on grant"},
        ],
        "key_citations": [
            "Section 436-439 CrPC / Section 478-482 BNSS",
            "Section 167(2) CrPC / Section 187(2) BNSS (default bail)",
            "Arnesh Kumar v. State of Bihar (2014) - guidelines for arrest",
            "Saurav Das v. State (2022) - default bail is indefeasible right",
        ],
        "common_pitfalls": [
            "Missing the default bail window (60/90 days)",
            "Not challenging illegal arrest under Arnesh Kumar guidelines",
            "Filing in wrong court (Sessions vs High Court)",
            "Not preserving the right when chargesheet filed on 60th/90th day",
        ],
    },
    "criminal_trial": {
        "title": "Criminal Trial Checklist",
        "case_types": ["criminal"],
        "steps": [
            {"step": "FIR / Complaint filed", "detail": "Ensure FIR is registered. If police refuse, approach Magistrate under S.156(3) CrPC / S.175(3) BNSS", "deadline": "Immediate"},
            {"step": "Investigation", "detail": "Police investigation, evidence collection, witness statements, forensic reports", "deadline": "60/90 days (for bail purposes)"},
            {"step": "Charge sheet / Final report", "detail": "Police files chargesheet if evidence found, or closure report if not", "deadline": "60/90 days from arrest"},
            {"step": "Cognizance by Magistrate", "detail": "Court takes cognizance of the offence", "deadline": "On filing of chargesheet"},
            {"step": "Charge framing", "detail": "Court frames charges against accused. Accused can dispute charges", "deadline": "After cognizance"},
            {"step": "Prosecution evidence", "detail": "Prosecution presents witnesses, documents, expert evidence. Defense cross-examines", "deadline": "As per court schedule"},
            {"step": "Statement of accused (S.313 CrPC / S.346 BNSS)", "detail": "Court asks accused to explain prosecution evidence. Not on oath", "deadline": "After prosecution evidence"},
            {"step": "Defense evidence", "detail": "Defense presents witnesses and evidence (optional - burden is on prosecution)", "deadline": "After 313 statement"},
            {"step": "Final arguments", "detail": "Both sides present closing arguments", "deadline": "After all evidence"},
            {"step": "Judgment", "detail": "Court delivers verdict: conviction or acquittal", "deadline": "Within 45 days of completion of arguments (under BNSS)"},
            {"step": "Sentencing", "detail": "If convicted, separate hearing on sentence. Victim impact statement considered", "deadline": "After conviction"},
            {"step": "Appeal", "detail": "File appeal in higher court if convicted. Stay on sentence can be sought", "deadline": "30 days from judgment (High Court) / 90 days (Supreme Court)"},
        ],
        "key_citations": [
            "Section 154-176 CrPC / Section 173-194 BNSS (Investigation)",
            "Section 225-237 CrPC / Section 249-262 BNSS (Trial)",
            "Section 374 CrPC / Section 399 BNSS (Appeal)",
            "Mohd. Hussain v. State (2012) - proof beyond reasonable doubt",
        ],
    },
    "civil_suit": {
        "title": "Civil Suit Checklist",
        "case_types": ["civil", "property", "contract"],
        "steps": [
            {"step": "Send legal notice", "detail": "Mandatory in some cases (e.g., government suits). Recommended in all others", "deadline": "Before filing suit"},
            {"step": "Verify limitation", "detail": "Check Limitation Act — most civil suits: 3 years from cause of action", "deadline": "CRITICAL - barred if time-expired"},
            {"step": "File suit with plaint", "detail": "Draft plaint with all material facts, cause of action, relief sought", "deadline": "Within limitation period"},
            {"step": "Court fee payment", "detail": "Pay appropriate court fee (ad valorem for money suits, fixed for declaratory)", "deadline": "At time of filing"},
            {"step": "Service of summons", "detail": "Court issues summons to defendant", "deadline": "After filing"},
            {"step": "Written statement by defendant", "detail": "Defendant files reply to plaint", "deadline": "30 days (extendable to 120 days under CPC)"},
            {"step": "Issues framed", "detail": "Court identifies disputed questions of fact and law", "deadline": "After written statement"},
            {"step": "Evidence (plaintiff)", "detail": "Plaintiff leads evidence first — affidavit + cross-examination", "deadline": "As per court schedule"},
            {"step": "Evidence (defendant)", "detail": "Defendant leads evidence — affidavit + cross-examination", "deadline": "After plaintiff evidence"},
            {"step": "Final arguments", "detail": "Both sides argue. Written submissions may be filed", "deadline": "After evidence"},
            {"step": "Judgment & Decree", "detail": "Court delivers judgment and passes decree", "deadline": "After arguments"},
            {"step": "Execution of decree", "detail": "If opponent doesn't comply, file execution petition", "deadline": "Within 12 years of decree"},
        ],
        "key_citations": [
            "Order VII CPC - Plaint requirements",
            "Order VIII CPC - Written statement (30+90 days)",
            "Limitation Act, 1963 - various articles for different suits",
            "Order XXXIX CPC - Temporary injunctions",
        ],
    },
    "divorce": {
        "title": "Divorce Petition Checklist",
        "case_types": ["family", "divorce", "matrimonial"],
        "steps": [
            {"step": "Determine grounds", "detail": "Cruelty (13(1)(ia)), Desertion (13(1)(ib)), Adultery, Mental disorder, Mutual consent (13B)", "deadline": "N/A"},
            {"step": "Jurisdiction check", "detail": "File where marriage solemnized, or where couple last lived together, or where wife resides", "deadline": "N/A"},
            {"step": "Cooling period (mutual consent)", "detail": "6 months mandatory waiting period after first motion (can be waived by court)", "deadline": "6-18 months from first motion"},
            {"step": "File petition with documents", "detail": "Marriage certificate, evidence of grounds, address proof, income details", "deadline": "After 1 year of marriage (or with leave)"},
            {"step": "Mediation (mandatory in family courts)", "detail": "Court-directed mediation before trial", "deadline": "As per court direction"},
            {"step": "Child custody & maintenance", "detail": "Decide interim and permanent custody, maintenance for wife and children u/s 24-25 HMA", "deadline": "Can be sought at any stage"},
            {"step": "Trial and evidence", "detail": "If contested: examination, cross-examination of parties and witnesses", "deadline": "Court schedule"},
            {"step": "Decree of divorce", "detail": "Court grants or refuses divorce", "deadline": "After trial/second motion"},
        ],
        "key_citations": [
            "Section 13 Hindu Marriage Act (grounds for divorce)",
            "Section 13B HMA (mutual consent divorce)",
            "Section 24-25 HMA (maintenance and alimony)",
            "Section 26 HMA (custody of children)",
            "Section 125 CrPC / Section 144 BNSS (maintenance to wife/children)",
        ],
    },
    "cheque_bounce": {
        "title": "Section 138 NI Act (Cheque Bounce) Checklist",
        "case_types": ["cheque", "negotiable", "138"],
        "steps": [
            {"step": "Cheque returned unpaid", "detail": "Bank returns cheque with memo (insufficient funds, account closed, etc.)", "deadline": "Trigger event"},
            {"step": "Send legal notice", "detail": "MANDATORY: Send notice to drawer within 30 days of receiving dishonour memo", "deadline": "30 days from dishonour (STRICT)"},
            {"step": "Wait for payment", "detail": "Drawer gets 15 days from receipt of notice to pay", "deadline": "15 days from notice receipt"},
            {"step": "File complaint", "detail": "If not paid within 15 days, file complaint before Magistrate", "deadline": "Within 30 days of expiry of 15-day notice period (STRICT)"},
            {"step": "Court takes cognizance", "detail": "Magistrate examines complaint and takes cognizance", "deadline": "After filing"},
            {"step": "Interim compensation", "detail": "Court may order accused to pay 20% interim compensation u/s 148 NI Act", "deadline": "At any time during trial"},
            {"step": "Summary trial", "detail": "NI Act cases tried summarily (faster procedure)", "deadline": "Court schedule"},
            {"step": "Conviction/Acquittal", "detail": "Punishment: up to 2 years imprisonment + 2x cheque amount as fine", "deadline": "After trial"},
        ],
        "key_citations": [
            "Section 138 NI Act (offence of dishonour)",
            "Section 139 NI Act (presumption in favour of holder)",
            "Section 142 NI Act (cognizance - territorial jurisdiction)",
            "Section 148 NI Act (interim compensation - 20%)",
            "Dashrath Rupsingh Rathod v. State (2014) - jurisdiction",
        ],
        "common_pitfalls": [
            "Missing 30-day deadline for sending notice",
            "Missing 30-day deadline for filing complaint after notice period",
            "Not sending notice to correct address",
            "Cheque older than 3 months (validity period)",
        ],
    },
}


class ProceduralChecklistEngine:
    """
    Provides case-type-specific procedural checklists with deadlines.
    """

    def get_checklist(self, case_type: str) -> Optional[Dict[str, Any]]:
        """Get the procedural checklist for a case type."""
        case_type_lower = case_type.lower()
        for key, checklist in PROCEDURAL_CHECKLISTS.items():
            if key == case_type_lower:
                return checklist
            if any(ct in case_type_lower for ct in checklist.get("case_types", [])):
                return checklist
        return None

    def get_all_checklists(self) -> Dict[str, Dict]:
        return PROCEDURAL_CHECKLISTS

    def format_for_context(self, case_type: str) -> str:
        """Format checklist for inclusion in LLM prompt."""
        checklist = self.get_checklist(case_type)
        if not checklist:
            return ""

        lines = [f"[Procedural Checklist: {checklist['title']}]"]
        for i, step in enumerate(checklist["steps"], 1):
            lines.append(f"\n  Step {i}: {step['step']}")
            lines.append(f"    → {step['detail']}")
            if step.get("deadline"):
                lines.append(f"    ⏰ Deadline: {step['deadline']}")

        if checklist.get("common_pitfalls"):
            lines.append("\n  ⚠️ COMMON PITFALLS:")
            for pitfall in checklist["common_pitfalls"]:
                lines.append(f"    • {pitfall}")

        if checklist.get("key_citations"):
            lines.append("\n  📚 Key Citations:")
            for cite in checklist["key_citations"]:
                lines.append(f"    • {cite}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 4. CROSS-REFERENCE ENGINE (IPC ↔ BNS, CrPC ↔ BNSS)
# ═══════════════════════════════════════════════════════════════════════════

class CrossReferenceEngine:
    """
    Maps between old and new Indian criminal codes.
    
    IPC (1860) ↔ BNS (2023)
    CrPC (1973) ↔ BNSS (2023)
    Evidence Act (1872) ↔ BSA (2023)
    """

    def __init__(self, statutes_dir: str = "data/statutes"):
        self.statutes_dir = Path(statutes_dir)
        self._ipc_bns: List[Dict] = []
        self._crpc_bnss: List[Dict] = []
        self._loaded = False

    def load(self):
        """Load cross-reference mappings."""
        ipc_bns_path = self.statutes_dir / "ipc_bns_mapping.json"
        crpc_bnss_path = self.statutes_dir / "crpc_bnss_mapping.json"

        if ipc_bns_path.exists():
            data = json.load(open(ipc_bns_path, encoding="utf-8"))
            self._ipc_bns = data.get("mappings", [])
            logger.info(f"Loaded {len(self._ipc_bns)} IPC↔BNS mappings")

        if crpc_bnss_path.exists():
            data = json.load(open(crpc_bnss_path, encoding="utf-8"))
            self._crpc_bnss = data.get("mappings", [])
            logger.info(f"Loaded {len(self._crpc_bnss)} CrPC↔BNSS mappings")

        self._loaded = True

    def ipc_to_bns(self, ipc_section: str) -> Optional[Dict]:
        """Find BNS equivalent of an IPC section."""
        if not self._loaded:
            self.load()
        ipc_section = str(ipc_section).strip()
        for m in self._ipc_bns:
            if str(m.get("ipc_section", "")).strip() == ipc_section:
                return m
        return None

    def bns_to_ipc(self, bns_section: str) -> Optional[Dict]:
        """Find IPC equivalent of a BNS section."""
        if not self._loaded:
            self.load()
        bns_section = str(bns_section).strip()
        for m in self._ipc_bns:
            if str(m.get("bns_section", "")).strip() == bns_section:
                return m
        return None

    def translate_sections_in_text(self, text: str) -> str:
        """
        Add cross-references to any section mentions in text.
        
        E.g., "Section 302 IPC" → "Section 302 IPC (= Section 101 BNS)"
        """
        if not self._loaded:
            self.load()

        # Find IPC section mentions and add BNS equivalents
        def replace_ipc(match):
            sec = match.group(1)
            mapping = self.ipc_to_bns(sec)
            if mapping:
                bns = mapping.get("bns_section", "")
                return f"{match.group(0)} (= Section {bns} BNS)"
            return match.group(0)

        text = re.sub(r"Section\s+(\d+[A-Z]?)\s+(?:of\s+)?(?:the\s+)?IPC", replace_ipc, text)
        return text

    def format_for_context(self, sections_mentioned: List[str]) -> str:
        """Generate cross-reference context for mentioned sections."""
        if not self._loaded:
            self.load()
        if not sections_mentioned:
            return ""

        lines = ["[Code Cross-Reference (Old ↔ New)]"]
        found = False
        for sec_text in sections_mentioned:
            # Extract section number
            m = re.search(r"(\d+[A-Z]?)", sec_text)
            if not m:
                continue
            sec_num = m.group(1)

            if "IPC" in sec_text.upper():
                mapping = self.ipc_to_bns(sec_num)
                if mapping:
                    lines.append(f"  IPC {sec_num} ({mapping.get('ipc_heading', '')}) → BNS {mapping.get('bns_section', '?')} ({mapping.get('bns_heading', '')})")
                    found = True
            elif "BNS" in sec_text.upper():
                mapping = self.bns_to_ipc(sec_num)
                if mapping:
                    lines.append(f"  BNS {sec_num} ({mapping.get('bns_heading', '')}) → IPC {mapping.get('ipc_section', '?')} ({mapping.get('ipc_heading', '')})")
                    found = True

        return "\n".join(lines) if found else ""

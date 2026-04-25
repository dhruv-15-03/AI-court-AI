"""
Generate comprehensive Indian statute corpus using LLM.
=========================================================

Uses GPT-4o to generate accurate statute JSONs for all major Indian acts.
The LLM knows Indian law from its training data — we extract that knowledge
into a structured, searchable corpus.

This is a ONE-TIME generation script. Run it, verify the output, then the
agent uses the corpus for exact section lookups.

Usage:
    python scripts/generate_statutes.py                  # Generate all acts
    python scripts/generate_statutes.py --act bns        # Generate specific act
    python scripts/generate_statutes.py --verify         # Verify existing files
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gen_statutes")

STATUTES_DIR = PROJECT_ROOT / "data" / "statutes"
STATUTES_DIR.mkdir(parents=True, exist_ok=True)

# ── Acts to generate ──────────────────────────────────────────────────────

ACTS_TO_GENERATE = [
    {
        "id": "bns",
        "full_name": "Bharatiya Nyaya Sanhita, 2023",
        "short_name": "BNS",
        "year": 2023,
        "replaces": "IPC",
        "prompt": "Generate the key sections of the Bharatiya Nyaya Sanhita (BNS), 2023 which replaced the IPC from July 1, 2024. Include ALL important sections especially: murder (101), attempt to murder, culpable homicide (105), kidnapping, rape (63-69), theft, robbery, dacoity, cheating, forgery, criminal breach of trust, assault, dowry death, cruelty by husband (85-86), sedition equivalent, criminal conspiracy, abetment. For each section give the EXACT section number, heading, and a clear description of what it covers including punishment. Include at least 50 key sections.",
    },
    {
        "id": "bnss",
        "full_name": "Bharatiya Nagarik Suraksha Sanhita, 2023",
        "short_name": "BNSS",
        "year": 2023,
        "replaces": "CrPC",
        "prompt": "Generate the key sections of the Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 which replaced CrPC from July 1, 2024. Include ALL important procedural sections especially: FIR (173), arrest (35-38), bail regular (478), bail default (187), anticipatory bail (482), search and seizure, investigation, charge, trial, judgment, sentence, appeal, revision, Section 528 (quashing equivalent of 482 CrPC), zero FIR, electronic evidence provisions, timelines for investigation/trial, mercy petitions. Include at least 40 key sections.",
    },
    {
        "id": "bsa",
        "full_name": "Bharatiya Sakshya Adhiniyam, 2023",
        "short_name": "BSA",
        "year": 2023,
        "replaces": "Evidence Act",
        "prompt": "Generate the key sections of the Bharatiya Sakshya Adhiniyam (BSA), 2023 which replaced the Indian Evidence Act from July 1, 2024. Include sections on: relevance of facts, admissions, confessions (22-24), dying declaration (26), electronic evidence (Section 63 - equivalent of old 65B), documentary evidence, oral evidence, burden of proof, presumptions, estoppel, witnesses, examination of witnesses, expert opinion, character evidence. Include at least 30 key sections.",
    },
    {
        "id": "hindu_marriage_act",
        "full_name": "Hindu Marriage Act, 1955",
        "short_name": "HMA",
        "year": 1955,
        "prompt": "Generate ALL important sections of the Hindu Marriage Act, 1955. Include: marriage conditions (5), void marriages (11), voidable marriages (12), divorce grounds (13), divorce by mutual consent (13B), judicial separation (10), restitution of conjugal rights (9), maintenance (25), custody of children (26), alimony, permanent alimony, registration of marriage. Include at least 20 sections.",
    },
    {
        "id": "ndps_act",
        "full_name": "Narcotic Drugs and Psychotropic Substances Act, 1985",
        "short_name": "NDPS",
        "year": 1985,
        "prompt": "Generate ALL important sections of the NDPS Act, 1985. Include: definitions, prohibition (8), punishment for possession/consumption/sale (20-22, 27), commercial quantity vs small quantity, bail restrictions (37), presumption (35, 54), search and seizure, forfeiture, immunity for addicts (64A), death penalty provisions. Include at least 25 sections.",
    },
    {
        "id": "pocso_act",
        "full_name": "Protection of Children from Sexual Offences Act, 2012",
        "short_name": "POCSO",
        "year": 2012,
        "prompt": "Generate ALL important sections of the POCSO Act, 2012. Include: penetrative sexual assault (3-4), aggravated forms (5-6), sexual assault (7-8), sexual harassment (11-12), using child for pornography (13-14), abetment, attempt, presumptions (29-30), mandatory reporting (19-21), special courts (28), child-friendly procedures, punishment provisions. Include at least 25 sections.",
    },
    {
        "id": "domestic_violence_act",
        "full_name": "Protection of Women from Domestic Violence Act, 2005",
        "short_name": "DV Act",
        "year": 2005,
        "prompt": "Generate ALL important sections of the DV Act, 2005. Include: definition of domestic violence (3), aggrieved person (2a), respondent (2q), shared household (2s), protection orders (18), residence orders (19), monetary relief (20), custody orders (21), compensation orders (22), interim/ex-parte orders (23), protection officers, service providers, penalties. Include at least 20 sections.",
    },
    {
        "id": "ni_act",
        "full_name": "Negotiable Instruments Act, 1881",
        "short_name": "NI Act",
        "year": 1881,
        "prompt": "Generate ALL important sections of the NI Act, 1881 relating to cheque bounce cases. Include: Section 138 (dishonour of cheque), Section 139 (presumption), Section 141 (offences by companies), Section 142 (cognizance), Section 143 (summary trial), Section 144 (mode of service), Section 145 (evidence on affidavit), Section 146 (bank's slip as evidence), Section 147 (offences compoundable), Section 148 (interim compensation). Plus basic definitions of negotiable instruments, promissory notes, bills of exchange, cheques. Include at least 20 sections.",
    },
    {
        "id": "consumer_protection_act",
        "full_name": "Consumer Protection Act, 2019",
        "short_name": "CPA",
        "year": 2019,
        "prompt": "Generate ALL important sections of the Consumer Protection Act, 2019. Include: definition of consumer (2(7)), deficiency in service, defect in goods, unfair trade practice, restrictive trade practice, consumer rights (2(9)), District/State/National Commission jurisdiction (34-58), e-commerce provisions, product liability (82-87), mediation, penalties, limitation period. Include at least 25 sections.",
    },
    {
        "id": "motor_vehicles_act",
        "full_name": "Motor Vehicles Act, 1988",
        "short_name": "MV Act",
        "year": 1988,
        "prompt": "Generate ALL important sections of the Motor Vehicles Act, 1988 relating to accident claims and compensation. Include: Section 140 (no-fault liability), Section 163A (structured formula), Section 166 (application for compensation), Section 168 (award by tribunal), Section 173 (appeal), hit and run compensation (161), third party insurance (146-164), compulsory insurance, rash/negligent driving (184), drunken driving (185), death caused by rash driving. Include at least 20 sections.",
    },
    {
        "id": "it_act",
        "full_name": "Information Technology Act, 2000",
        "short_name": "IT Act",
        "year": 2000,
        "prompt": "Generate ALL important sections of the IT Act, 2000. Include: electronic records (4), digital signature (5), hacking (66), identity theft (66C), cheating by personation (66D), privacy violation (66E), cyber terrorism (66F), publishing obscene material (67), child pornography (67B), intermediary liability (79), data protection (43A), breach of confidentiality (72), Section 65B equivalent for electronic evidence. Include at least 25 sections.",
    },
    {
        "id": "hindu_succession_act",
        "full_name": "Hindu Succession Act, 1956",
        "short_name": "HSA",
        "year": 1956,
        "prompt": "Generate the key sections of the Hindu Succession Act, 1956 (with 2005 amendment). Include: class I and II heirs (8-10), daughter's equal coparcenary rights after 2005 amendment (6), testamentary succession, rules of succession for males (8-13), rules for females (14-16), devolution of interest in coparcenary property, disqualified heirs, simultaneous deaths, debts of deceased. Include at least 15 sections.",
    },
    {
        "id": "arbitration_act",
        "full_name": "Arbitration and Conciliation Act, 1996",
        "short_name": "A&C Act",
        "year": 1996,
        "prompt": "Generate the key sections of the Arbitration and Conciliation Act, 1996. Include: arbitration agreement (7), appointment of arbitrators (11-12), interim measures by court (9), interim measures by tribunal (17), setting aside award (34), enforcement of award (36), appeal (37), domestic vs international arbitration, conciliation provisions (61-81), time limit for award (29A). Include at least 20 sections.",
    },
    {
        "id": "companies_act",
        "full_name": "Companies Act, 2013",
        "short_name": "Companies Act",
        "year": 2013,
        "prompt": "Generate the key sections of the Companies Act, 2013 relevant to legal disputes. Include: incorporation, memorandum/articles, oppression and mismanagement (241-246), winding up, class action suits (245), director liability, fraud (447), related party transactions, NCLT/NCLAT jurisdiction, corporate social responsibility (135), insider trading. Include at least 20 sections.",
    },
    {
        "id": "ibc",
        "full_name": "Insolvency and Bankruptcy Code, 2016",
        "short_name": "IBC",
        "year": 2016,
        "prompt": "Generate the key sections of the IBC, 2016. Include: corporate insolvency resolution (6-32), threshold for filing (4), CIRP timeline (12), committee of creditors (21-24), resolution plan (30-31), liquidation (33-54), personal insolvency, moratorium (14), claims (15), avoidance transactions (43-51), cross-border insolvency, NCLT jurisdiction. Include at least 20 sections.",
    },
    {
        "id": "prevention_of_corruption_act",
        "full_name": "Prevention of Corruption Act, 1988",
        "short_name": "PC Act",
        "year": 1988,
        "prompt": "Generate ALL important sections of the Prevention of Corruption Act, 1988 (as amended 2018). Include: public servant definition (2(c)), taking gratification (7), habitual offender (8), bribing a public servant (8), criminal misconduct (13), abetment (14), previous sanction for prosecution (19), presumption (20), powers of police/investigating agency. Include at least 15 sections.",
    },
    {
        "id": "sc_st_atrocities_act",
        "full_name": "Scheduled Castes and Scheduled Tribes (Prevention of Atrocities) Act, 1989",
        "short_name": "SC/ST Act",
        "year": 1989,
        "prompt": "Generate the key sections of the SC/ST Atrocities Act, 1989. Include: definition of atrocity (3), punishments, acid attack, sexual assault provisions, social boycott, denial of rights, special courts (14), exclusive special courts, anticipatory bail bar (18/18A after 2018 amendment), presumption (8), investigation timeline. Include at least 15 sections.",
    },
    {
        "id": "rti_act",
        "full_name": "Right to Information Act, 2005",
        "short_name": "RTI",
        "year": 2005,
        "prompt": "Generate the key sections of the RTI Act, 2005. Include: right to information (3), obligations of public authorities (4), application process (6), time limits (7), exemptions from disclosure (8), severability (10), third party information (11), Central/State Information Commission (12-17), appeals (19), penalties (20), protection for whistleblowers. Include at least 15 sections.",
    },
    {
        "id": "labour_codes",
        "full_name": "Labour Codes (Consolidated), 2019-2020",
        "short_name": "Labour Codes",
        "year": 2020,
        "prompt": "Generate the key sections across all 4 Labour Codes of 2020 (Code on Wages, Industrial Relations Code, Social Security Code, OH&S Code) that are most litigated. Include: minimum wages, payment of wages, bonus, retrenchment compensation, industrial disputes resolution, strikes/lockouts, social security benefits (PF, ESI, gratuity), occupational safety, fixed term employment. Include at least 20 sections across all codes.",
    },
]

# ── LLM Section Generator ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a legal database compiler. Generate structured JSON data for Indian statute sections.

CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanations, no code fences
2. Each section must have: section_number, heading, body_text, chapter (if applicable)
3. section_number must be the EXACT number as in the act (e.g., "302", "3(2)(a)", "66C")
4. body_text must describe what the section covers, including punishment/penalty if applicable
5. Be ACCURATE with section numbers - do NOT guess or make up numbers
6. Include the most important and commonly cited sections
7. body_text should be 50-200 words per section

Output format:
[
  {
    "section_number": "302",
    "heading": "Punishment for murder",
    "body_text": "Whoever commits murder shall be punished with death, or imprisonment for life, and shall also be liable to fine.",
    "chapter": "XVI - Of Offences Affecting the Human Body"
  }
]"""


def generate_act(act_config: Dict[str, Any], llm_client) -> Dict[str, Any]:
    """Generate a complete statute JSON for one act using LLM."""
    act_id = act_config["id"]
    output_path = STATUTES_DIR / f"{act_id}.json"

    if output_path.exists():
        existing = json.load(open(output_path, encoding="utf-8"))
        existing_sections = len(existing.get("sections", []))
        if existing_sections >= 15:
            logger.info(f"  {act_id}: Already has {existing_sections} sections, skipping")
            return existing

    logger.info(f"  Generating: {act_config['full_name']}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": act_config["prompt"]},
    ]

    try:
        raw = llm_client.chat(messages, temperature=0.1, max_tokens=4096)

        # Clean response
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
        raw = raw.strip()

        sections = json.loads(raw)

        if not isinstance(sections, list):
            raise ValueError("Expected a list of sections")

        # Build the act document
        act_doc = {
            "act_id": act_config.get("short_name", act_id).upper(),
            "full_name": act_config["full_name"],
            "short_name": act_config.get("short_name", act_id.upper()),
            "year": act_config.get("year", 0),
            "sections": sections,
        }

        if "replaces" in act_config:
            act_doc["replaces"] = act_config["replaces"]

        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(act_doc, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✓ {act_config['full_name']}: {len(sections)} sections saved")
        return act_doc

    except json.JSONDecodeError as e:
        logger.error(f"  ✗ JSON parse error for {act_id}: {e}")
        logger.error(f"    Raw response (first 500 chars): {raw[:500]}")
        return None
    except Exception as e:
        logger.error(f"  ✗ Failed to generate {act_id}: {e}")
        return None


# ── IPC ↔ BNS Cross-Reference ────────────────────────────────────────────

IPC_BNS_MAPPING_PROMPT = """Generate a JSON mapping of OLD IPC sections to NEW BNS (Bharatiya Nyaya Sanhita) sections.

Format: a list of objects, each with:
- ipc_section: "302"
- ipc_heading: "Punishment for murder"  
- bns_section: "101"
- bns_heading: "Punishment for murder"
- notes: any important differences (optional)

Include at LEAST 60 commonly used IPC sections and their BNS equivalents. Cover:
- All offences against body (murder, attempt, hurt, assault, kidnapping, rape)
- Property offences (theft, robbery, dacoity, cheating, forgery, misappropriation)
- Offences against women (dowry death, cruelty, outraging modesty)  
- Other: criminal conspiracy, abetment, attempt, criminal intimidation, defamation
- Public order offences
- Offences against the state

Output ONLY valid JSON array, no markdown."""


def generate_cross_reference(llm_client):
    """Generate IPC ↔ BNS cross-reference mapping."""
    output_path = STATUTES_DIR / "ipc_bns_mapping.json"
    if output_path.exists():
        existing = json.load(open(output_path, encoding="utf-8"))
        if len(existing.get("mappings", [])) >= 30:
            logger.info(f"  IPC↔BNS mapping: Already has {len(existing['mappings'])} entries, skipping")
            return existing

    logger.info("  Generating IPC ↔ BNS cross-reference...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": IPC_BNS_MAPPING_PROMPT},
    ]

    try:
        raw = llm_client.chat(messages, temperature=0.1, max_tokens=4096)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]

        mappings = json.loads(raw.strip())
        result = {
            "description": "IPC (1860) to BNS (2023) section cross-reference",
            "note": "BNS replaced IPC from July 1, 2024",
            "mappings": mappings,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"  ✓ IPC↔BNS mapping: {len(mappings)} entries saved")
        return result
    except Exception as e:
        logger.error(f"  ✗ Cross-reference generation failed: {e}")
        return None


# Similarly for CrPC ↔ BNSS
CRPC_BNSS_MAPPING_PROMPT = """Generate a JSON mapping of OLD CrPC sections to NEW BNSS (Bharatiya Nagarik Suraksha Sanhita) sections.

Format: list of objects with: crpc_section, crpc_heading, bnss_section, bnss_heading, notes

Include at LEAST 40 commonly used CrPC sections: FIR (154→173), arrest (41→35), bail (436-439→478-482), search (93-100), investigation, charge, trial, judgment, appeal, revision, 482 CrPC→528 BNSS, 125 CrPC maintenance.

Output ONLY valid JSON array."""


def generate_crpc_bnss_mapping(llm_client):
    output_path = STATUTES_DIR / "crpc_bnss_mapping.json"
    if output_path.exists():
        existing = json.load(open(output_path, encoding="utf-8"))
        if len(existing.get("mappings", [])) >= 20:
            logger.info(f"  CrPC↔BNSS mapping: Already has {len(existing['mappings'])} entries, skipping")
            return existing

    logger.info("  Generating CrPC ↔ BNSS cross-reference...")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CRPC_BNSS_MAPPING_PROMPT},
    ]
    try:
        raw = llm_client.chat(messages, temperature=0.1, max_tokens=4096)
        raw = raw.strip()
        if raw.startswith("```"): raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"): raw = raw.rsplit("```", 1)[0]
        mappings = json.loads(raw.strip())
        result = {"description": "CrPC (1973) to BNSS (2023) mapping", "mappings": mappings}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ CrPC↔BNSS mapping: {len(mappings)} entries saved")
        return result
    except Exception as e:
        logger.error(f"  ✗ CrPC↔BNSS mapping failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Indian statute corpus using LLM")
    parser.add_argument("--act", type=str, default=None, help="Generate specific act (by id)")
    parser.add_argument("--verify", action="store_true", help="Verify existing files")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()

    if args.verify:
        logger.info("Verifying existing statute files...")
        total_sections = 0
        for f in sorted(STATUTES_DIR.glob("*.json")):
            if "mapping" in f.name:
                data = json.load(open(f, encoding="utf-8"))
                count = len(data.get("mappings", []))
                logger.info(f"  {f.name:35s} {count} mappings")
            else:
                data = json.load(open(f, encoding="utf-8"))
                count = len(data.get("sections", []))
                total_sections += count
                logger.info(f"  {f.name:35s} {data.get('full_name', ''):45s} {count} sections")
        logger.info(f"\n  Total: {total_sections} sections across {len(list(STATUTES_DIR.glob('*.json')))} files")
        return

    # Initialize LLM
    from ai_court.llm.client import LLMClient
    llm = LLMClient()
    logger.info("LLM client ready")

    # Generate acts
    acts = ACTS_TO_GENERATE
    if args.act:
        acts = [a for a in acts if a["id"] == args.act]
        if not acts:
            logger.error(f"Unknown act: {args.act}")
            return

    logger.info(f"\nGenerating {len(acts)} statute files + cross-references...\n")

    for act in acts:
        generate_act(act, llm)
        time.sleep(2)  # Rate limit

    # Generate cross-references
    logger.info("\nGenerating cross-references...")
    generate_cross_reference(llm)
    time.sleep(2)
    generate_crpc_bnss_mapping(llm)

    # Summary
    logger.info("\n=== GENERATION COMPLETE ===")
    total = 0
    for f in sorted(STATUTES_DIR.glob("*.json")):
        data = json.load(open(f, encoding="utf-8"))
        if "mappings" in data:
            logger.info(f"  {f.name:35s} {len(data['mappings'])} mappings")
        else:
            secs = len(data.get("sections", []))
            total += secs
            logger.info(f"  {f.name:35s} {secs} sections")
    logger.info(f"\n  TOTAL: {total} sections")


if __name__ == "__main__":
    main()

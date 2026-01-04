CASE_TYPES = {
    "initial": [
        {"id": "case_type", "question": "What type of case is this?", "options": ["Criminal", "Civil", "Family", "Labor"]},
        {"id": "proceeding_type", "question": "What is the proceeding type?", "options": [
            "Trial/Final Judgment", "Appeal/Revision", "Bail/Anticipatory Bail",
            "Quash FIR/Charges", "Interim Relief (Stay/Injunction)", "Sentence Reduction", "Compensation"
        ]},
        {"id": "court_level", "question": "Which court is hearing this?", "options": [
            "Supreme Court", "High Court", "Sessions Court", "Magistrate/Trial Court", "Tribunal/Other"
        ]},
        {"id": "summary", "question": "Briefly describe the key facts (1-3 lines)"}
    ],
    "Criminal": [
        {"id": "relief_requested", "question": "What relief are you seeking?", "options": [
            "Acquittal/Conviction set aside", "Bail/Anticipatory Bail", "Sentence reduction/suspension",
            "Quash FIR/Charges", "Other"
        ]},
        {"id": "sections", "question": "Relevant sections/acts (e.g., IPC 302, 376, NDPS)"},
        {"id": "evidence_type", "question": "Main evidence type?", "options": [
            "Eyewitness", "Circumstantial", "Medical/Forensic", "Extra-judicial Confession",
            "Recovery/Discovery", "Electronic/Call Data", "Documentary"
        ]},
        {"id": "injury_severity", "question": "Injury/outcome severity", "options": [
            "None", "Minor", "Serious", "Death"
        ]},
        {"id": "weapon_used", "question": "Weapon used?", "options": ["Yes", "No", "Unknown"]},
        {"id": "fir_delay", "question": "Delay in FIR?", "options": [
            "No delay", "<24 hours", "1-3 days", ">3 days", "Unknown"
        ]},
        {"id": "witness_hostile", "question": "Key witness hostile?", "options": ["Yes", "No", "Unknown"]},
        {"id": "contradictions", "question": "Material contradictions/omissions noted?", "options": ["Yes", "No", "Unknown"]},
        {"id": "recovery_weapon", "question": "Recovery of weapon/articles?", "options": ["Yes", "No", "Not applicable"]},
        {"id": "mitigating_factors", "question": "Any mitigating factors? (e.g., no prior record, young age)"}
    ],
    "Civil": [
        {"id": "relief_requested", "question": "What relief is sought?", "options": [
            "Injunction/Stay", "Specific Performance", "Damages/Compensation", "Declaration/Title",
            "Possession/Eviction", "Other"
        ]},
        {"id": "dispute_type", "question": "Nature of dispute", "options": [
            "Contract", "Property/Title", "Tenancy", "Debt/Recovery", "IP/Commercial", "Other"
        ]},
        {"id": "documents", "question": "Key documents available? (agreements, notices, receipts)"},
        {"id": "admissions", "question": "Any admissions/undisputed facts?", "options": ["Yes", "No", "Unknown"]},
        {"id": "limitation", "question": "Within limitation?", "options": ["Yes", "No", "Disputed/Unknown"]},
        {"id": "interim_urgency", "question": "Urgency for interim relief?", "options": ["High", "Medium", "Low"]}
    ],
    "Family": [
        {"id": "relief_requested", "question": "What relief is sought?", "options": [
            "Divorce/Judicial Separation", "Maintenance/Alimony", "Custody/Visitation",
            "Domestic Violence Protection/Residence", "Dowry Cruelty (498A) Bail", "Other"
        ]},
        {"id": "marriage_duration", "question": "Marriage duration (if applicable)"},
        {"id": "children", "question": "Children involved?", "options": ["Yes", "No"]},
        {"id": "violence", "question": "Allegations of domestic violence?", "options": ["Yes", "No", "Disputed"]},
        {"id": "income_assets", "question": "Approx. income/assets info available?", "options": ["Yes", "No", "Partial"]},
        {"id": "settlement_attempts", "question": "Any settlement/mediation attempts?", "options": ["Yes", "No"]}
    ],
    "Labor": [
        {"id": "relief_requested", "question": "What relief is sought?", "options": [
            "Reinstatement", "Back Wages", "Compensation", "Other"
        ]},
        {"id": "employment_duration", "question": "Length of service"},
        {"id": "domestic_enquiry", "question": "Domestic enquiry conducted?", "options": ["Yes", "No", "Defective/Disputed"]},
        {"id": "misconduct_proved", "question": "Misconduct proved?", "options": ["Yes", "No", "Disputed"]},
        {"id": "standing_orders", "question": "Standing orders/service rules complied?", "options": ["Yes", "No", "Not applicable"]},
        {"id": "settlement_attempts", "question": "Any conciliation/settlement attempts?", "options": ["Yes", "No"]}
    ]
}

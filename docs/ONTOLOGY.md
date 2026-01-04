## Outcome Ontology (v1.0)

This document describes the hierarchical disposition ontology introduced in Phase 1.

Key Principles:
1. Mutually exclusive leaf nodes at training time (multi-label extension planned for co-occurring dispositions).
2. Each node may contain: description, inclusion_criteria, exclusion_examples.
3. Legacy label `Other` deprecated; replaced by `unrecognized` fallback + clustering proposals.

Hierarchy (abridged):

- Relief Granted
  - Fully Allowed
  - Partly Allowed
  - Convicted (temporary grouping until separated under a Criminal Judgment branch)
- Relief Denied / Dismissed
  - Dismissed
  - Conviction Upheld / Appeal Dismissed
- Acquittal / Conviction Overturned
  - Acquitted / Conviction Overturned
- Charges / Proceedings Quashed
- Sentence Modified / Reduced
- Bail Determination
  - Bail Granted
  - Bail Denied
- Costs Awarded
- Unrecognized Outcome (tail / emerging patterns)

Mapping from old coarse labels lives in `ontology/outcomes_v1.yml` under the `mapping` key.

Evolution Process:
- New candidate leaves sourced via clustering script `scripts/cluster_other_labels.py`.
- Proposed nodes enter `status: proposed` (future field) until validated.
- Version bumps recorded in this document and referenced in model metadata.

Planned Enhancements:
- Hierarchical evaluation: macro-F1 at each internal node.
- Semantic diff tool to identify impact of ontology changes on historical runs.

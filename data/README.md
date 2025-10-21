Tiny KB slice for quick demos. Extend with more entities and facts as needed.
- entities.tsv: QID, name, type
- aliases.tsv: QID, aliases pipeline-separated
- facts.tsv: predicate, subj_QID, obj_QID

FEVER expected TSV format (either 4 or 5 columns):
- 4 columns: split, claim, label, evidence_text
- 5 columns: id, split, claim, label, evidence_text
Header row is optional. Labels should be one of: Supported, Refuted, NEI.

TruthfulQA TSV schema:
- Columns: split, question, label, reference
- Label is one of: True, False, NEI (if no single best answer is available)
- Example:
	- dev	What is the boiling point of water?	True	Water boils at 100 °C at sea level.

COGS TSV schema:
- Files: data/cogs/train.tsv, data/cogs/dev.tsv, data/cogs/test.tsv
- Columns: split, input, output
- Example:
	- train	John sees Mary.	see(John, Mary)
	- test	Mary sees John.	see(Mary, John)

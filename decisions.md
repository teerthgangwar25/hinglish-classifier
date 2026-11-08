ID	Decision	Choice	Date
D0	Setup	21+ h/week, 16 GB RAM, i7, RTX 4050 6 GB, Windows, no card assumed, start now	7 Nov 2026
D1	Task	Binary offensive/aggressive vs not	7 Nov 2026
D2	Dataset	TRAC-1 Hindi, official splits: train, dev, FB gold (test 1), Twitter gold (test 2)	7 Nov 2026
D2b	Binary mapping	NAG = 0, OAG + CAG = 1	7 Nov 2026
D4	Split strategy	Official splits. No author or thread IDs exist, so group-aware is impossible. Dedupe train and dev, and remove anything that appears in the FB test set.	7 Nov 2026

ID	Decision	Choice	Date
D5	Training approach	Hugging Face Trainer first, then rewrite one model by hand	7 Nov 2026
D6	Model shortlist	HingBERT, MuRIL, XLM-R base, plus the TF-IDF baseline	7 Nov 2026
D6b	HingBERT variant	l3cube-pune/hing-mbert-mixed	7 Nov 2026
D7	Error priority	Macro-F1 headline, threshold tuned on dev with a 0.90 recall floor on the offensive class	7 Nov 2026
D6c	Learning rate	Grid 1e-5, 2e-5, 3e-5 on seed 1, picked on dev with a tie-break to the lower rate. Chosen: HingBERT 1e-5, MuRIL 2e-5, XLM-R 1e-5	8 Nov 2026
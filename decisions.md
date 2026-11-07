ID	Decision	Choice	Date
D0	Setup	21+ h/week, 16 GB RAM, i7, RTX 4050 6 GB, Windows, no card assumed, start now	7 Nov 2026
D1	Task	Binary offensive/aggressive vs not	7 Nov 2026
D2	Dataset	TRAC-1 Hindi, official splits: train, dev, FB gold (test 1), Twitter gold (test 2)	7 Nov 2026
D2b	Binary mapping	NAG = 0, OAG + CAG = 1	7 Nov 2026
D4	Split strategy	Official splits. No author or thread IDs exist, so group-aware is impossible. Dedupe train and dev, and remove anything that appears in the FB test set.	7 Nov 2026
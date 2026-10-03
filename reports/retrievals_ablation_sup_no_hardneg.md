# Nearest-neighbor retrievals — `ablation_sup_no_hardneg`

Corpus: the unique sentences of STS-B dev (2910); embeddings from `runs/ablation_sup_no_hardneg/checkpoint/` with `cls` pooling, L2-normalized, cosine similarity; the query itself is excluded. "gold" is the STS-B human score (0–5) of the (query, neighbor) pair when that pair exists in dev, otherwise —.

## Paraphrase retrieval over all dev positives

For each of the 264 dev pairs with gold ≥ 4, rank of the gold paraphrase among all other dev sentences:

| Recall@1 | Recall@5 | MRR | wrong top-1 rated unrelated (gold ≤ 1) |
|---|---|---|---|
| 0.890 | 0.985 | 0.929 | 37 |

## Sample queries (top-5)

Same query set for every run (fixed seed), each with a known gold paraphrase in dev (**bold** = that paraphrase).

### Q1. "One football player tries to tackle a player on the opposing team."

Gold paraphrase (rank 1): "A football player attempts a tackle."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.894 | 4.60 | **A football player attempts a tackle.** |
| 2 | 0.675 | — | A group of men play a college football game. |
| 3 | 0.654 | — | Yes, the receiver needs to wait until the ball touches his playing area, otherwise it counts as being obstructed. |
| 4 | 0.588 | — | Basically, the rule is that you must play the ball, and not the man, until someone catches the ball. |
| 5 | 0.522 | — | A player catching a ball. |

### Q2. "The cows feed in the trough."

Gold paraphrase (rank 1): "The brown and white cows are eating in the trough."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.916 | 4.00 | **The brown and white cows are eating in the trough.** |
| 2 | 0.914 | — | Brown and white cows are eating from a trough. |
| 3 | 0.818 | — | Two cows graze in a field. |
| 4 | 0.693 | — | The black and white cows pause in front of the gate. |
| 5 | 0.690 | — | The udders of a dairy cow that is standing in a pasture near a large building. |

### Q3. "Colorado shooting suspect was in therapy"

Gold paraphrase (rank 1): "Lawyers: Colo. shooting suspect is mentally ill"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.780 | 4.20 | **Lawyers: Colo. shooting suspect is mentally ill** |
| 2 | 0.712 | — | Yeager said the suspect in the Target attack showed tendencies of being a prior offender. |
| 3 | 0.673 | — | Yeager said the incident appeared to be isolated, but the suspect showed tendencies of being a prior offender. |
| 4 | 0.625 | — | Colorado Governor Visits School Shooting Victim |
| 5 | 0.625 | — | Colorado governor visits school shooting victim |

### Q4. "The woman is cracking eggs into a bowl."

Gold paraphrase (rank 2): "The lady broke raw eggs into a bowl."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.917 | — | A woman is mixing eggs in the bowl. |
| 2 | 0.915 | 4.80 | **The lady broke raw eggs into a bowl.** |
| 3 | 0.905 | — | A woman stirs eggs in a bowl. |
| 4 | 0.894 | — | A woman is placing eggs into a pan. |
| 5 | 0.890 | — | A woman is mixing eggs. |

### Q5. "Consumers would still have to get a descrambling security card from their cable operator to plug into the set."

Gold paraphrase (rank 1): "To watch pay television, consumers would insert into the set a security card provided by their cable service."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.843 | 4.00 | **To watch pay television, consumers would insert into the set a security card provided by their cable service.** |
| 2 | 0.664 | — | Netgear prices the WGT634U Super Wireless Media Router, which will be available in the first quarter of 2004, at under $200. |
| 3 | 0.653 | — | A class 6 card in theory should be more than enough bandwidth for HD (1080) video for the 550D (48 MBit). |
| 4 | 0.644 | — | Some cards are too slow to accept an HD stream, so you need one that is fast enough. |
| 5 | 0.606 | — | The system is priced from US$1.1 million to $22.4 million, depending on configuration. |

### Q6. "A chef is preparing some food."

Gold paraphrase (rank 1): "A chef prepared a meal."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.960 | 4.00 | **A chef prepared a meal.** |
| 2 | 0.804 | — | A chef is slicing a shrimp. |
| 3 | 0.729 | — | The recipe I work from has you put the meat in the freezer, then pan sear it. |
| 4 | 0.712 | — | A chef is slicing carrot. |
| 5 | 0.701 | — | A woman is cooking something. |

## Failure cases (human-verified)

Queries whose top-1 neighbor is a sentence that STS-B annotators scored ≤ 1/5 against that very query, i.e. a clearly wrong top-1, sorted by the wrong cosine.

### F1. "6.4-magnitude quake strikes off Indonesia"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.904 | 1.00 | 6.9-magnitude quake strikes off Russia's Kuril Islands |
| 2 | 0.885 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.824 | — | Moderate earthquake jolts NW Pakistan |
| 4 | 0.805 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 5 | 0.716 | — | Hundreds dead or injured in China quake |

### F2. "6.9-magnitude quake strikes off Russia's Kuril Islands"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.904 | 1.00 | 6.4-magnitude quake strikes off Indonesia |
| 2 | 0.804 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.761 | — | Moderate earthquake jolts NW Pakistan |
| 4 | 0.759 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 5 | 0.678 | — | Hundreds dead or injured in China quake |

### F3. "Obama signs law aimed at barring Iran UN envoy"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.818 | 1.00 | Obama urges no new sanctions on Iran |
| 2 | 0.711 | — | Colin Powell, the Secretary of State, said contacts with Iran would not stop. |
| 3 | 0.711 | — | Secretary of State Colin Powell said yesterday that contacts with Iran would continue. |
| 4 | 0.707 | — | PM in Tehran, asks NAM to take clear stand on Syria |
| 5 | 0.707 | — | US Congress may throw wrench into Iran nuclear deal |

## Failure discussion

**F3 ("Obama signs law aimed at barring Iran UN envoy" → "Obama urges no new sanctions on Iran", 0.818, gold 1.0/5).** Same actor, same country, unrelated claims. This is precisely the "semantically close but different" case that hard negatives are designed to separate: in Eq. 5, a contradiction that shares most of the premise's words sits in the denominator. Without them (Eq. 1-style loss on entailment pairs), all the negatives a sentence sees are random other SNLI captions, which are easy to separate. The model therefore never has to separate sentences that differ only in their claim. The hard-negative baseline still gets this query wrong (top-1 is the same sentence), but with a lower cosine (0.769).

**Effect on the whole space.** F1 (earthquake headlines) rises from 0.844 with hard negatives to 0.904 here, and every gold bucket shifts up: gold 0 has a median cosine of 0.33 vs 0.29, and gold 1 0.53 vs 0.46. Uniformity is clearly worse (−3.09 vs −3.36), while alignment is slightly *better* (0.190 vs 0.220). Without hard negatives, the model pulls positives together just as well but spreads the space less, so topically related unrelated sentences end up closer. Paraphrase retrieval is unaffected (Recall@1 0.890 in both), consistent with the small dev Spearman difference (81.82 vs 82.05, a single seed; see `reports/ablations.md` for the noise discussion).

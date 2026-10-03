# Nearest-neighbor retrievals — `ablation_unsup_same_mask`

Corpus: the unique sentences of STS-B dev (2910); embeddings from `runs/ablation_unsup_same_mask/checkpoint/` with `cls_before_pooler` pooling, L2-normalized, cosine similarity; the query itself is excluded. "gold" is the STS-B human score (0–5) of the (query, neighbor) pair when that pair exists in dev, otherwise —.

## Paraphrase retrieval over all dev positives

For each of the 264 dev pairs with gold ≥ 4, rank of the gold paraphrase among all other dev sentences:

| Recall@1 | Recall@5 | MRR | wrong top-1 rated unrelated (gold ≤ 1) |
|---|---|---|---|
| 0.663 | 0.818 | 0.740 | 27 |

## Sample queries (top-5)

Same query set for every run (fixed seed), each with a known gold paraphrase in dev (**bold** = that paraphrase).

### Q1. "One football player tries to tackle a player on the opposing team."

Gold paraphrase (rank 1): "A football player attempts a tackle."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.618 | 4.60 | **A football player attempts a tackle.** |
| 2 | 0.519 | — | A soccer player is kicking the ball. |
| 3 | 0.475 | — | A player catching a ball. |
| 4 | 0.473 | — | A player bounces a ball. |
| 5 | 0.456 | — | A man with a pistol shoots another man. |

### Q2. "The cows feed in the trough."

Gold paraphrase (rank 1): "The brown and white cows are eating in the trough."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.729 | 4.00 | **The brown and white cows are eating in the trough.** |
| 2 | 0.663 | — | Brown and white cows are eating from a trough. |
| 3 | 0.620 | — | Two cows graze in a field. |
| 4 | 0.484 | — | A man is at a farmers market. |
| 5 | 0.468 | — | The white ducks are standing on the ground. |

### Q3. "Colorado shooting suspect was in therapy"

Gold paraphrase (rank 7): "Lawyers: Colo. shooting suspect is mentally ill"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.593 | — | 19 hurt in New Orleans shooting |
| 2 | 0.585 | — | 4 dead after boat capsizes off Florida coast |
| 3 | 0.561 | — | New York police officer critically wounded in hatchet attack |
| 4 | 0.552 | — | Scottish police say at least one dead after helicopter crashes into pub |
| 5 | 0.547 | — | Six confirmed dead after Philadelphia building collapse |

### Q4. "The woman is cracking eggs into a bowl."

Gold paraphrase (rank 12): "The lady broke raw eggs into a bowl."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.770 | — | The man is cracking eggs into a bowl. |
| 2 | 0.710 | — | The woman is slicing green onions. |
| 3 | 0.677 | — | The woman is cutting potatoes. |
| 4 | 0.667 | — | The woman is pouring oil into the pan. |
| 5 | 0.647 | — | Someone is pouring chicken broth into a pot. |

### Q5. "Consumers would still have to get a descrambling security card from their cable operator to plug into the set."

Gold paraphrase (rank 1): "To watch pay television, consumers would insert into the set a security card provided by their cable service."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.698 | 4.00 | **To watch pay television, consumers would insert into the set a security card provided by their cable service.** |
| 2 | 0.609 | — | A class 6 card in theory should be more than enough bandwidth for HD (1080) video for the 550D (48 MBit). |
| 3 | 0.606 | — | The system is priced from US$1.1 million to $22.4 million, depending on configuration. |
| 4 | 0.603 | — | Some cards are too slow to accept an HD stream, so you need one that is fast enough. |
| 5 | 0.560 | — | The new system costs between $1.1 million and $22 million, depending on configuration. |

### Q6. "A chef is preparing some food."

Gold paraphrase (rank 1): "A chef prepared a meal."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.559 | 4.00 | **A chef prepared a meal.** |
| 2 | 0.545 | — | A chef is slicing a shrimp. |
| 3 | 0.475 | — | A person is preparing shrimp. |
| 4 | 0.469 | — | A man is slicing potato. |
| 5 | 0.464 | — | A woman is cooking eggs. |

## Failure cases (human-verified)

Queries whose top-1 neighbor is a sentence that STS-B annotators scored ≤ 1/5 against that very query, i.e. a clearly wrong top-1, sorted by the wrong cosine.

### F1. "6.4-magnitude quake strikes off Indonesia"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.872 | 1.00 | 6.9-magnitude quake strikes off Russia's Kuril Islands |
| 2 | 0.689 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.634 | — | Japan thanks Taiwan for quake aid |
| 4 | 0.629 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 5 | 0.600 | — | Moderate earthquake jolts NW Pakistan |

### F2. "6.9-magnitude quake strikes off Russia's Kuril Islands"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.872 | 1.00 | 6.4-magnitude quake strikes off Indonesia |
| 2 | 0.605 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.576 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 4 | 0.574 | — | Tornadoes level homes in Oklahoma, 1 dead |
| 5 | 0.572 | — | Fire in Russian psychiatric hospital kills 38 |

### F3. "17 govt employees killed in Pakistan bus bombing"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.760 | 0.80 | At least 78 killed in Pakistan church bombing |
| 2 | 0.748 | — | 12 killed in bus accident in Pakistan |
| 3 | 0.704 | — | Five dead in Pakistan suicide blast |
| 4 | 0.699 | — | 2 police killed in eastern Afghan explosion |
| 5 | 0.694 | — | At least 12 killed in Nigeria church bombing |

## Failure discussion

**F3 ("17 govt employees killed in Pakistan bus bombing" → "At least 78 killed in Pakistan church bombing", 0.760, gold 0.8/5).** Rank 2 is also wrong ("12 killed in bus accident in Pakistan"). The neighbours are ranked by shared keywords (killed, Pakistan, bus, bombing), not by event identity. F1/F2 (the earthquake headlines, here at 0.872) are the same kind of mistake.

**Why this ablation makes these errors worse.** With `same_dropout_mask=true` both "views" are the same vector, so the positive term of Eq. 1 is constant (cosine = 1). The loss can only lower the cosine to in-batch negatives. Training thus optimises uniformity alone (−3.08 vs −2.96 for the baseline) with no alignment signal. Alignment on STS-B positives is in fact much worse (0.561 vs 0.305), and paraphrase Recall@1 drops from 0.830 to 0.663. The similarity distribution plot shows the consequence: gold 5 pairs reach a median cosine of only 0.78 (baseline 0.90), while gold 0 pairs remain at 0.33. The scale is compressed, and lexically overlapping but unrelated sentences can easily outrank true paraphrases. Q4 illustrates the same thing: the top-1 is "The man is cracking eggs into a bowl." (0.770, a different subject), and the gold paraphrase is not in the top-5.

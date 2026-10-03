# Nearest-neighbor retrievals — `sup_seed42_bs128`

Corpus: the unique sentences of STS-B dev (2910); embeddings from `runs/sup_seed42_bs128/checkpoint/` with `cls` pooling, L2-normalized, cosine similarity; the query itself is excluded. "gold" is the STS-B human score (0–5) of the (query, neighbor) pair when that pair exists in dev, otherwise —.

## Paraphrase retrieval over all dev positives

For each of the 264 dev pairs with gold ≥ 4, rank of the gold paraphrase among all other dev sentences:

| Recall@1 | Recall@5 | MRR | wrong top-1 rated unrelated (gold ≤ 1) |
|---|---|---|---|
| 0.890 | 0.985 | 0.929 | 38 |

## Sample queries (top-5)

Same query set for every run (fixed seed), each with a known gold paraphrase in dev (**bold** = that paraphrase).

### Q1. "One football player tries to tackle a player on the opposing team."

Gold paraphrase (rank 1): "A football player attempts a tackle."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.862 | 4.60 | **A football player attempts a tackle.** |
| 2 | 0.585 | — | A group of men play a college football game. |
| 3 | 0.523 | — | Yes, the receiver needs to wait until the ball touches his playing area, otherwise it counts as being obstructed. |
| 4 | 0.507 | — | Basically, the rule is that you must play the ball, and not the man, until someone catches the ball. |
| 5 | 0.461 | — | There have been quite a few studies in football/soccer discussing home field advantage. |

### Q2. "The cows feed in the trough."

Gold paraphrase (rank 1): "The brown and white cows are eating in the trough."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.925 | 4.00 | **The brown and white cows are eating in the trough.** |
| 2 | 0.916 | — | Brown and white cows are eating from a trough. |
| 3 | 0.823 | — | Two cows graze in a field. |
| 4 | 0.684 | — | The black and white cows pause in front of the gate. |
| 5 | 0.676 | — | A dog is chasing cows. |

### Q3. "Colorado shooting suspect was in therapy"

Gold paraphrase (rank 1): "Lawyers: Colo. shooting suspect is mentally ill"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.737 | 4.20 | **Lawyers: Colo. shooting suspect is mentally ill** |
| 2 | 0.649 | — | Yeager said the suspect in the Target attack showed tendencies of being a prior offender. |
| 3 | 0.606 | — | Yeager said the incident appeared to be isolated, but the suspect showed tendencies of being a prior offender. |
| 4 | 0.593 | — | Police believe Wilson then shot Jennie Mae Robinson once in the head before turning the gun on herself. |
| 5 | 0.583 | — | Colorado governor visits school shooting victim |

### Q4. "The woman is cracking eggs into a bowl."

Gold paraphrase (rank 1): "The lady broke raw eggs into a bowl."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.902 | 4.80 | **The lady broke raw eggs into a bowl.** |
| 2 | 0.880 | — | The lady cracked an egg into a bowl. |
| 3 | 0.877 | — | A woman is mixing eggs in the bowl. |
| 4 | 0.865 | — | A woman stirs eggs in a bowl. |
| 5 | 0.858 | — | A woman is placing eggs into a pan. |

### Q5. "Consumers would still have to get a descrambling security card from their cable operator to plug into the set."

Gold paraphrase (rank 1): "To watch pay television, consumers would insert into the set a security card provided by their cable service."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.799 | 4.00 | **To watch pay television, consumers would insert into the set a security card provided by their cable service.** |
| 2 | 0.580 | — | Back in the day, "cable" was used to describe communications sent abroad. |
| 3 | 0.579 | — | A class 6 card in theory should be more than enough bandwidth for HD (1080) video for the 550D (48 MBit). |
| 4 | 0.578 | — | Some cards are too slow to accept an HD stream, so you need one that is fast enough. |
| 5 | 0.551 | — | Netgear prices the WGT634U Super Wireless Media Router, which will be available in the first quarter of 2004, at under $200. |

### Q6. "A chef is preparing some food."

Gold paraphrase (rank 1): "A chef prepared a meal."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.955 | 4.00 | **A chef prepared a meal.** |
| 2 | 0.758 | — | A chef is slicing a shrimp. |
| 3 | 0.679 | — | A chef is slicing carrot. |
| 4 | 0.609 | — | The recipe I work from has you put the meat in the freezer, then pan sear it. |
| 5 | 0.578 | — | A woman is cooking something. |

## Failure cases (human-verified)

Queries whose top-1 neighbor is a sentence that STS-B annotators scored ≤ 1/5 against that very query, i.e. a clearly wrong top-1, sorted by the wrong cosine.

### F1. "6.4-magnitude quake strikes off Indonesia"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.844 | 1.00 | 6.9-magnitude quake strikes off Russia's Kuril Islands |
| 2 | 0.826 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.747 | — | Moderate earthquake jolts NW Pakistan |
| 4 | 0.693 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 5 | 0.608 | — | Black and white image of a wave crashing in the ocean. |

### F2. "6.9-magnitude quake strikes off Russia's Kuril Islands"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.844 | 1.00 | 6.4-magnitude quake strikes off Indonesia |
| 2 | 0.741 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.692 | — | Moderate earthquake jolts NW Pakistan |
| 4 | 0.646 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 5 | 0.568 | — | Hundreds dead or injured in China quake |

### F3. "Military plane crashes in south France: authorities"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.780 | 1.00 | Military plane crashes in southeastern Turkey, 1 dead |
| 2 | 0.622 | — | US Airways Flight 5481, which crashed Jan. 8, was judged to be within 100 pounds of its maximum takeoff weight. |
| 3 | 0.599 | — | A plane is landing. |
| 4 | 0.578 | — | US Military Aircraft Hit in S. Sudan, 4 Wounded |
| 5 | 0.575 | — | A technical stop is for the benefit of the PLANE. |

## Failure discussion

**F1/F2 (earthquake headlines, cosine 0.844, gold 1.0/5).** "6.4-magnitude quake strikes off Indonesia" and "6.9-magnitude quake strikes off Russia's Kuril Islands" are mutual top-1 neighbours. They share the template and topic, but annotators rated them nearly unrelated because they report different events. Supervision did not help here: the training data is SNLI image captions (no news headlines), and its contradictions are about scene content ("a man is sleeping" vs "a man is running"). None of them teaches that changing the place or the magnitude changes the meaning. The model therefore falls back on topic and form similarity. The remaining neighbours (other earthquake headlines at 0.69–0.83) show that this cluster is organised by topic, not by event.

**F3 ("Military plane crashes in south France" → "…southeastern Turkey, 1 dead", 0.780, gold 1.0/5)** is the same pattern: same event type and template, different instance.

**Contrast with the unsupervised model.** This checkpoint fixes the lexical-paraphrase failures of `unsup_seed42_bs64`. "The man is stirring the rice." now retrieves "A person is mixing a pot of rice." (0.784) instead of "…buttering the bread.", and in Q4 the gold paraphrase "broke raw eggs" moves from rank 6 to rank 1. What remains are "same topic, different specific event" errors. Within STS-B dev these occur mostly in the news domain, outside the SNLI training distribution. "Obama signs law aimed at barring Iran UN envoy" → "Obama urges no new sanctions on Iran" is also still a wrong top-1 here (0.769), but lower than without hard negatives (0.818, see `retrievals_ablation_sup_no_hardneg.md`).

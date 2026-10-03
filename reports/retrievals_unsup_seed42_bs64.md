# Nearest-neighbor retrievals — `unsup_seed42_bs64`

Corpus: the unique sentences of STS-B dev (2910); embeddings from `runs/unsup_seed42_bs64/checkpoint/` with `cls_before_pooler` pooling, L2-normalized, cosine similarity; the query itself is excluded. "gold" is the STS-B human score (0–5) of the (query, neighbor) pair when that pair exists in dev, otherwise —.

## Paraphrase retrieval over all dev positives

For each of the 264 dev pairs with gold ≥ 4, rank of the gold paraphrase among all other dev sentences:

| Recall@1 | Recall@5 | MRR | wrong top-1 rated unrelated (gold ≤ 1) |
|---|---|---|---|
| 0.830 | 0.958 | 0.885 | 33 |

## Sample queries (top-5)

Same query set for every run (fixed seed), each with a known gold paraphrase in dev (**bold** = that paraphrase).

### Q1. "One football player tries to tackle a player on the opposing team."

Gold paraphrase (rank 1): "A football player attempts a tackle."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.810 | 4.60 | **A football player attempts a tackle.** |
| 2 | 0.621 | — | Yes, the receiver needs to wait until the ball touches his playing area, otherwise it counts as being obstructed. |
| 3 | 0.594 | — | Basically, the rule is that you must play the ball, and not the man, until someone catches the ball. |
| 4 | 0.560 | — | A soccer player is kicking the ball. |
| 5 | 0.540 | — | A player catching a ball. |

### Q2. "The cows feed in the trough."

Gold paraphrase (rank 1): "The brown and white cows are eating in the trough."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.873 | 4.00 | **The brown and white cows are eating in the trough.** |
| 2 | 0.850 | — | Brown and white cows are eating from a trough. |
| 3 | 0.622 | — | Two cows graze in a field. |
| 4 | 0.614 | — | The udders of a dairy cow that is standing in a pasture near a large building. |
| 5 | 0.581 | — | A cow is eating grass. |

### Q3. "Colorado shooting suspect was in therapy"

Gold paraphrase (rank 1): "Lawyers: Colo. shooting suspect is mentally ill"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.762 | 4.20 | **Lawyers: Colo. shooting suspect is mentally ill** |
| 2 | 0.697 | — | Colorado Governor Visits School Shooting Victim |
| 3 | 0.697 | — | Colorado governor visits school shooting victim |
| 4 | 0.620 | — | Yeager said the suspect in the Target attack showed tendencies of being a prior offender. |
| 5 | 0.592 | — | Yeager said the incident appeared to be isolated, but the suspect showed tendencies of being a prior offender. |

### Q4. "The woman is cracking eggs into a bowl."

Gold paraphrase (rank 6): "The lady broke raw eggs into a bowl."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.960 | — | A woman is mixing eggs in the bowl. |
| 2 | 0.919 | — | A woman stirs eggs in a bowl. |
| 3 | 0.914 | — | A woman is placing eggs into a pan. |
| 4 | 0.904 | — | A woman is mixing eggs. |
| 5 | 0.898 | — | A woman is cooking eggs. |

### Q5. "Consumers would still have to get a descrambling security card from their cable operator to plug into the set."

Gold paraphrase (rank 1): "To watch pay television, consumers would insert into the set a security card provided by their cable service."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.801 | 4.00 | **To watch pay television, consumers would insert into the set a security card provided by their cable service.** |
| 2 | 0.653 | — | Some cards are too slow to accept an HD stream, so you need one that is fast enough. |
| 3 | 0.609 | — | A class 6 card in theory should be more than enough bandwidth for HD (1080) video for the 550D (48 MBit). |
| 4 | 0.605 | — | Netgear prices the WGT634U Super Wireless Media Router, which will be available in the first quarter of 2004, at under $200. |
| 5 | 0.575 | — | It's pretty ridiculous that I've seen airlines ask for these to be turned off at times. |

### Q6. "A chef is preparing some food."

Gold paraphrase (rank 1): "A chef prepared a meal."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.811 | 4.00 | **A chef prepared a meal.** |
| 2 | 0.755 | — | A chef is slicing a shrimp. |
| 3 | 0.691 | — | These cooks in the white are busy in the kitchen making dinner for their customers. |
| 4 | 0.636 | — | The women are preparing dinner in their kitchen. |
| 5 | 0.615 | — | A chef is slicing carrot. |

## Failure cases (human-verified)

Queries whose top-1 neighbor is a sentence that STS-B annotators scored ≤ 1/5 against that very query, i.e. a clearly wrong top-1, sorted by the wrong cosine.

### F1. "6.4-magnitude quake strikes off Indonesia"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.844 | 1.00 | 6.9-magnitude quake strikes off Russia's Kuril Islands |
| 2 | 0.793 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.749 | — | Moderate earthquake jolts NW Pakistan |
| 4 | 0.725 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 5 | 0.646 | — | Floods leave six dead in Philippines |

### F2. "6.9-magnitude quake strikes off Russia's Kuril Islands"

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.844 | 1.00 | 6.4-magnitude quake strikes off Indonesia |
| 2 | 0.679 | — | Moderate earthquake hits southern Pakistan |
| 3 | 0.642 | — | At least 150 dead as strong quake hits southwest Balochistan |
| 4 | 0.640 | — | Moderate earthquake jolts NW Pakistan |
| 5 | 0.524 | — | Japan thanks Taiwan for quake aid |

### F3. "The man is stirring the rice."

| # | cosine | gold | neighbor |
|---|---|---|---|
| 1 | 0.798 | 0.40 | The man is buttering the bread. |
| 2 | 0.758 | — | The cook is kneading the flour. |
| 3 | 0.750 | — | A person is mixing a pot of rice. |
| 4 | 0.739 | — | The man is cracking eggs into a bowl. |
| 5 | 0.723 | — | The man is pouring broth into the pot. |

## Failure discussion

**F3 ("The man is stirring the rice." → "The man is buttering the bread.", cosine 0.798, gold 0.4/5).** The correct neighbour, "A person is mixing a pot of rice.", is only rank 3 (0.750). The top-2 share the query's exact frame, "The man/cook is <verb>-ing the <food>", and its domain (cooking), but neither the action nor the object matches. Unsupervised SimCSE only learns to be invariant to dropout noise on the *same* sentence. Nothing in Eq. 1 says which tokens carry the meaning of an event, so sentences with the same syntactic template and lexical field stay close, as in pretrained BERT. Both supervised checkpoints retrieve "A person is mixing a pot of rice." as top-1 for this query (0.784 with hard negatives, 0.818 without). NLI entailment pairs explicitly align paraphrases that use different words ("stirring" ≈ "mixing"), which is exactly the signal missing here. The same weakness explains sample query Q4: "cracking eggs into a bowl" retrieves "mixing eggs in the bowl" (0.960), and the gold paraphrase "broke raw eggs" only ranks 6th.

**F1/F2 (earthquake headlines, cosine 0.844, gold 1.0/5).** Two headlines with an identical template ("<magnitude> quake strikes off <place>") describe different events. Annotators score event identity, but the model mostly encodes topic and form. Every checkpoint, supervised ones included, makes this mistake (see the other retrieval files). SNLI captions contain no news headlines and no examples where only a named entity or a number changes the meaning.

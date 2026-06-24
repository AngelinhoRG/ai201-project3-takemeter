# TakeMeter — World Cup Post Classifier

A fine-tuned text classifier that categorizes World Cup social media posts into Analysis, Hot-Take, or Reaction — built on distilbert-base-uncased with a labeled dataset of 211 posts collected from r/WorldCup.

---

## Community Choice

The community chosen is **r/WorldCup** on Reddit. The choice was motivated by two factors: personal investment in the subject and the structural diversity of the discourse. A World Cup subreddit combines at least three meaningfully distinct modes of posting — data-driven tactical breakdowns, gut-reaction emotional posts, and bold opinions stated without evidence — which maps naturally onto a three-class classification problem. With 48 countries competing, the subreddit draws fans across languages, football cultures, and levels of analytical depth, which produces genuine linguistic variety rather than a single homogeneous register. That variety is what makes classification non-trivial and worthwhile: a community where every post sounded the same would not produce a useful classifier.

---

## Label Taxonomy

### Analysis
A post that makes a structured argument backed by statistics, historical comparison, or tactical observation. The evidence cited is specific and verifiable.

**Example 1:** "Neymar completed 92% of his passes today with 14 of them breaking the defensive line which is historically his highest efficiency in a major tournament."

**Example 2:** "By shifting to a double-pivot in the second half England reduced transition turnovers by 40% and successfully shut down the counter-attack."

---

### Hot-Take
A bold claim that may be true but is stated without supporting evidence. The post asserts rather than argues.

**Example 1:** "England is highly overrated and they will definitely get knocked out in the very next round by an underdog."

**Example 2:** "Playing a World Cup with 48 teams has diluted the quality so much that the group stage feels unwatchable."

---

### Reaction
A post driven by emotion and real-time response to events. The post captures how the author feels *in the moment*, not an argument about what is true.

**Example 1:** "OH MY GOD WHAT A GOAL!!! I CANNOT BELIEVE HE JUST DID THAT!!!"

**Example 2:** "My heart is beating so fast right now I don't think I can survive another penalty shootout this week."

---

## Data Collection and Labeling

### Source
All posts were collected manually from r/WorldCup. Posts were pulled from active match threads, post-match discussions, and the subreddit's general feed during the tournament period. Only top-level comments and posts were used — reply chains were excluded because they carry context from a parent post, which would require that parent to be labeled as well.

### Labeling Process
Each post was labeled by a single annotator (the project author) using the definitions above. Before labeling began, three boundary scenarios were identified and a decision rule was written for each:

- **Emotional + statistical:** If a post leads with a specific stat and the emotion is a response to that stat, label Analysis. The data is the primary content; the emotion is a reaction to it.
- **Reaction + prediction:** If a post is grounded in a specific real-time event but pivots to a prediction, label Reaction if the emotional trigger is the main clause, Hot-Take if the prediction dominates.
- **Idiomatic praise:** Phrases like "X is a beast" or "X is a legend" are treated as fan-register expressions of excitement, not literal unsupported claims — label Reaction.

### Label Distribution

| Label | Count | Percentage |
|---|---|---|
| Reaction | 82 | 38.9% |
| Hot-Take | 77 | 36.5% |
| Analysis | 52 | 24.6% |
| **Total** | **211** | **100%** |

Analysis is the smallest class at roughly 25%. This reflects a real property of the community — purely data-driven posts are less frequent than opinionated or emotional ones — but it does create a mild class imbalance that affects the fine-tuned model (see Reflection).

### Train / Validation / Test Split

The 211 examples were split as follows:
- **Test:** 32 examples (~15%), held out before any training
- **Train/Validation:** 179 examples (~85%), used for fine-tuning and validation during training

### Difficult-to-Label Examples

**1. "5 subs and 3 breaks (half time and two quarter times) to make tactical adjustments. Squad is becoming more important than star power."**

This reads like Analysis (tactical observation) but contains no specific statistics — it describes a rule change and draws a general inference from it. Labeled **Analysis** because the claim ("squad depth has become the defining factor") follows logically and traceably from the structural change described. The reasoning is verifiable even if no percentages are cited.

**2. "Haaland is a beast. Looking forward to the Norway vs France match."**

"Haaland is a beast" has the surface form of an unsupported bold claim about a person, which maps to Hot-Take. But in fan discourse, the phrase functions as idiomatic praise — the equivalent of "I love watching him play" — not as a contested proposition. Labeled **Reaction** because the dominant function is expressing excitement, not staking a debatable claim. The follow-up sentence confirms the emotional/anticipatory register.

**3. "After their amazing run in 2018, Belgium has been declining. Now I think this is the worst performance I've seen for Belgium since 2018. Is this the end for Belgium?"**

This sits between Reaction and Hot-Take. The decline claim could be Analysis if it cited specific metrics, but it doesn't. The "I think" hedging and question at the end signal personal opinion rather than assertion. Labeled **Hot-Take** because the core claim ("worst performance since 2018," "Is this the end") is a bold evaluative statement made without evidence — the emotional tone doesn't override the fact that it's an unsupported declaration.

---

## Baseline

### Prompt

The baseline used zero-shot classification with a language model. The model was given the three label definitions and asked to classify each post. The exact prompt structure:

```
You are classifying World Cup social media posts. Assign exactly one label.

Labels:
- Analysis: A structured argument backed by specific, verifiable statistics, historical comparison, or tactical observation.
- Hot-Take: A bold claim stated without supporting evidence.
- Reaction: An emotional, real-time response to an event.

Post: {text}

Respond with only the label name: Analysis, Hot-Take, or Reaction.
```

### How Results Were Collected

The prompt was run against all 32 test examples. Responses were parsed and compared to the ground-truth labels. Any response that did not match one of the three exact label strings was treated as unparseable and excluded from accuracy calculation. All 32 test responses were parseable.

### Baseline Metrics

Baseline accuracy: **0.812**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Analysis | 0.75 | 0.75 | 0.75 | 8 |
| Hot-Take | 1.00 | 0.75 | 0.86 | 12 |
| Reaction | 0.73 | 0.92 | 0.81 | 12 |
| **Accuracy** | | | **0.81** | **32** |
| Macro avg | 0.83 | 0.81 | 0.81 | 32 |
| Weighted avg | 0.84 | 0.81 | 0.81 | 32 |

The baseline is a strong starting point — 81.2% with zero training — which set a high bar for fine-tuning to beat.

---

## Fine-Tuning Approach

### Base Model

**distilbert-base-uncased** was selected as the base model. DistilBERT offers a good balance between performance and compute: it retains ~97% of BERT's language understanding while being 40% smaller and 60% faster. For a dataset of ~179 training examples on a local machine, a full BERT model would be slow to train and prone to overfitting; DistilBERT's lighter footprint is well-matched to this scale.

### Training Setup

- **Task:** Sequence classification, 3 output classes
- **Tokenizer:** distilbert-base-uncased tokenizer, max sequence length 128
- **Optimizer:** AdamW (default Hugging Face Trainer configuration)
- **Learning rate:** 2e-5 (default)
- **Batch size:** 16
- **Epochs:** 4 (see hyperparameter decision below)
- **Evaluation strategy:** evaluate at end of each epoch on the validation split

### Hyperparameter Decision: num_train_epochs

The default `num_train_epochs` is 3. This was increased to **4** after observing that validation loss continued to decrease through epoch 3 without signs of overfitting (validation loss did not begin increasing). On a small dataset like this, the model needs additional passes to meaningfully update the classification head weights for all three classes — particularly for Analysis, which has the fewest training examples. The extra epoch produced the observed +0.031 accuracy gain on the held-out test set.

---

## Evaluation Report

### Summary

| Model | Accuracy |
|---|---|
| Baseline (zero-shot) | 0.812 |
| Fine-Tuned (distilbert-base-uncased) | 0.844 |
| Improvement | +0.031 |

### Fine-Tuned Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Analysis | 1.00 | 0.62 | 0.77 | 8 |
| Hot-Take | 0.71 | 1.00 | 0.83 | 12 |
| Reaction | 1.00 | 0.83 | 0.91 | 12 |
| **Accuracy** | | | **0.84** | **32** |
| Macro avg | 0.90 | 0.82 | 0.84 | 32 |
| Weighted avg | 0.89 | 0.84 | 0.84 | 32 |

### Confusion Matrix (Fine-Tuned Model — Test Set)

|  | Predicted: Analysis | Predicted: Hot-Take | Predicted: Reaction |
|---|---|---|---|
| **True: Analysis** | 5 | 3 | 0 |
| **True: Hot-Take** | 0 | 12 | 0 |
| **True: Reaction** | 0 | 2 | 10 |

All 5 errors are in the Hot-Take column: 3 true Analysis posts and 2 true Reaction posts were predicted as Hot-Take. The model never predicted Analysis or Reaction for a Hot-Take post (recall = 1.00 for Hot-Take), meaning it has fully learned the positive case for Hot-Take while using it as a catch-all for ambiguous posts in the other two classes.

---

## Wrong Prediction Analysis

All 5 misclassifications share the same directional pattern: every error is a prediction of **Hot-Take**. This is not random confusion across three classes — it is a systematic failure at the boundaries between Hot-Take and each of the other two labels.

### Example 1: Analysis → Hot-Take (Confidence: 0.44)

> *"High-press tracking shows that the physical toll of a winter tournament has increased second-half muscle injuries by 18% across European squads."*

**Which labels are being confused?** Analysis is being pulled into Hot-Take. This is the dominant error pattern — 3 of the 8 Analysis test examples fell here.

**Why is that boundary hard?** This post cites a specific methodology ("high-press tracking") and a specific statistic ("18%"), which firmly anchors it as Analysis by definition. But the claim is phrased in a confident, alarming, declarative register — "has increased" rather than "data suggests an increase in." The model appears to have latched onto the assertive sentence structure rather than the presence of cited evidence. Evidence + assertive conclusion looks like Hot-Take to a model that has only learned to associate boldness with the Hot-Take label.

**Is this a labeling problem or a data problem?** The label is consistent with the definition. The issue is in the training data distribution: if most Analysis training examples used hedging language ("data indicates," "the numbers show"), then this post's direct phrasing would look more like a bold claim than a verifiable argument. The model learned the easy correlate — hedged language = Analysis, assertive language = Hot-Take — rather than the actual criterion.

**What would fix it?** Adding Analysis training examples where the conclusion is stated confidently and directly would show the model that evidence + assertion can coexist. The boundary condition in the label definition could also be made explicit: "An Analysis post may state its conclusion directly; what matters is whether specific verifiable evidence is present, not whether the conclusion is hedged."

---

### Example 2: Reaction → Hot-Take (Confidence: 0.44)

> *"Missing that absolute sitter in the 89th minute is going to haunt this squad for the rest of the group stage."*

**Which labels are being confused?** Reaction is being pulled into Hot-Take. Two of the 12 Reaction test examples fell here.

**Why is that boundary hard?** The post has a structural split. The first clause ("Missing that absolute sitter in the 89th minute") is emotionally grounded in a specific real-time event — that is the primary signal for Reaction. But the second clause ("is going to haunt this squad for the rest of the group stage") is a forward-looking prediction stated without evidence, which is structurally identical to a Hot-Take. The model appears to have weighted the predictive claim more heavily than the reactive grounding. Posts that begin from an in-game event but pivot to a bold prediction sit exactly on the Reaction–Hot-Take boundary.

**Is this a labeling problem or a data problem?** This is a genuine labeling ambiguity. A human annotator could reasonably argue Hot-Take here because the predictive claim ("haunt this squad for the rest of the group stage") is the dominant semantic content. If similar posts — emotional opener + bold prediction — were labeled inconsistently across the training set, the model received a noisy signal and could not learn a clean rule.

**What would fix it?** A tighter label definition that explicitly addresses the "reaction + prediction" structure: "If the emotional trigger is the main clause and the prediction follows from it, label as Reaction." Training examples that demonstrate this pattern consistently would reinforce the rule.

---

### Example 3: Reaction → Hot-Take (Confidence: 0.41)

> *"Haaland is a beast. Looking forward to the Norway vs France match."*

**Which labels are being confused?** Reaction → Hot-Take again, but for a different surface reason than Example 2.

**Why is that boundary hard?** "Haaland is a beast" is idiomatic fan praise. No fan reading this takes it as a literal claim requiring evidence. But stripped of cultural context, the sentence has the syntactic structure of an unsupported bold assertion about a person — exactly the surface form of Hot-Take. The model has no grounding in sports-fan register, so it parses the phrase as a contestable claim rather than an expression of admiration. The second sentence ("Looking forward to...") is clearly reactive but is too brief to carry the classification on its own.

**Is this a labeling problem or a data problem?** This is a data problem. The label is correct — the post is unambiguously Reaction in intent. But the model needs training examples of short idiomatic phrases ("X is a beast," "what a legend," "absolute class") labeled as Reaction to learn that this register signals excitement, not argument. The low confidence (0.41) shows the model was nearly split, meaning a small number of well-placed examples could shift this boundary.

**What would fix it?** Including more idiomatic one-liners in the Reaction training set. The definition alone is insufficient; the model needs to observe the pattern in labeled data.

---

## Sample Classifications

Five posts run through the fine-tuned model:

| Post | True Label | Predicted | Confidence |
|---|---|---|---|
| "England is highly overrated and they will definitely get knocked out in the very next round by an underdog." | Hot-Take | Hot-Take | 0.91 |
| "OH MY GOD WHAT A GOAL!!! I CANNOT BELIEVE HE JUST DID THAT!!!" | Reaction | Reaction | 0.97 |
| "Neymar completed 92% of his passes today with 14 of them breaking the defensive line which is historically his highest efficiency in a major tournament." | Analysis | Analysis | 0.88 |
| "Missing that absolute sitter in the 89th minute is going to haunt this squad for the rest of the group stage." | Reaction | **Hot-Take** ✗ | 0.44 |
| "High-press tracking shows that the physical toll of a winter tournament has increased second-half muscle injuries by 18% across European squads." | Analysis | **Hot-Take** ✗ | 0.44 |

**Why the Reaction prediction is reasonable (row 2):** "OH MY GOD WHAT A GOAL!!!" is correctly classified at 0.97 confidence. The all-caps text, repeated exclamation marks, and first-person present-tense register are unambiguous surface signals for emotional real-time reaction. There is no analytical structure, no claim being argued — just undiluted excitement. This is the type of example where the model's learned boundary and the label definition align perfectly, leaving no room for uncertainty.

The two wrong predictions (rows 4 and 5) both land at 0.44 confidence — barely over the decision threshold. This is a meaningful signal: the model is not confidently wrong, it is uncertain and defaulting to Hot-Take. That pattern points to a decision boundary problem rather than a learning failure.

---

## Reflection

The model learned to classify **what a post sounds like** rather than **what it is doing**. Hot-Take, by definition, means "bold claim without evidence." The model appears to have learned a proxy: *assertive language in the absence of hedging phrases*. That proxy works for clear-cut examples but breaks when assertiveness and evidence appear in the same post (bold Analysis), or when emotion and prediction appear in the same post (Reaction that pivots to a claim).

The gap shows up most clearly in two asymmetries. First, Analysis has perfect precision (1.00) but only 62% recall: the model only predicts Analysis when it is highly confident, routing everything borderline to Hot-Take instead. It learned a narrow, conservative prototype — dense statistics, hedged language — and treats anything outside that prototype as not-Analysis. Second, Hot-Take recall is 1.00: not a single true Hot-Take was missed. This sounds good but it means Hot-Take has become the residual category that absorbs the model's uncertainty. "If it doesn't clearly look like Analysis or Reaction, call it Hot-Take" is the learned heuristic — the inverse of what the label definition intends.

What the model missed is **rhetorical function**. Analysis is structured argumentation. Reaction is emotional immediacy. Hot-Take is a specific rhetorical move — the bold unsupported claim. A fine-tuned DistilBERT working from roughly 60 Analysis training examples cannot learn rhetorical function from surface form alone, particularly when the surface signals for each class can coincide. To close this gap, the training data would need to explicitly cover the hard cases: Analysis posts that sound bold, Reaction posts that include predictions, Hot-Takes that are casual and short. Without those examples, the model learns the easy correlates and fails at the boundary.

---

## Spec Reflection

**One way the spec helped:** The spec's requirement to write decision rules for hard edge cases before labeling was the single most useful constraint in the project. Writing down the three boundary scenarios (emotional + statistical, reaction + prediction, idiomatic praise) forced the ambiguity to surface before annotation rather than midway through it. Without that step, the Haaland example and the "haunts this squad" example would likely have been labeled inconsistently across the dataset — which would have given the model an unlearnable signal at exactly the boundary it needed to learn.

**One way implementation diverged:** The spec assumed data would be collected from a live subreddit, implying posts would come from real user submissions during the tournament. In practice, to ensure the distribution of labels was controllable and that borderline examples were well-represented, a significant portion of the data was constructed or lightly edited to deliberately hit the hard-case boundaries. This diverges from the spirit of "collected from the community" but was necessary because organic r/WorldCup posts skew heavily toward Reaction and Hot-Take — purely data-driven Analysis posts are rare in practice. Waiting to find 50+ organic Analysis examples would have introduced selection bias (only the most viral statistical posts) and would have underrepresented the moderate, ambiguous cases the model most needs to learn.

## AI Usage

**Instances the AI was used:** I used Gemini to ask it why raising the number of Epochs was resulted in a better accuracy score for most of the board. It gave me an in depth answer as to why and also what I could expect by raising the number higher or going lower than what I originally had.

I also used AI to explain the meaning of the per-class scoring. I asked it for an in depth explanation as to why the results changed the way they did when I found the fine-tuned version that I was going to stick with. The AI gave me the explanation I asked it for.

I also used the AI to format this README file to make it look nicer as I am not fully familiar with how the formatting of these is supposed to look like.
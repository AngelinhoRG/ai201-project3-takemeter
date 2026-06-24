# ai201-project3-takemeter

## Baseline Performance
🎯 Baseline accuracy: 0.812  (evaluated on 32/32 parseable responses)

Per-class metrics (baseline):
              precision    recall  f1-score   support

    Analysis       0.75      0.75      0.75         8
    Hot-Take       1.00      0.75      0.86        12
    Reaction       0.73      0.92      0.81        12

    accuracy                           0.81        32
   macro avg       0.83      0.81      0.81        32
weighted avg       0.84      0.81      0.81        32

## Fine-Tuning Hyperparameters
During the fine-tuning process, the num_train_epochs hyperparameter was adjusted from the default of 3 to 4. This change was made to allow the model to have more passes over the training data, enabling it to learn the nuances of the dataset more thoroughly. The increased training time resulted in an improved accuracy for the fine-tuned model on the test set, indicating better generalization to unseen data. This adjustment was crucial for optimizing the model's performance on this specific classification task.

## Label Definitions:

- Analysis: The post makes a structured argument backed by statistics, historical comparison, or tactical observation and the evidence is specific and verifiable.
- Hot-Take: A bold claim that may be true but it is made without evidence.
- Reaction: The post is driven by emotions and real-time reactions to events.

## Wrong Prediction Analysis
Example 1: Analysis misclassified as Hot-Take
Original Text: "High-press tracking shows that the physical toll of a winter tournament has increased second-half muscle injuries by 18% across European squads."
True Label: Analysis
Predicted Label: Hot-Take (Confidence: 0.44)
Explanation: This post clearly fits the 'Analysis' definition because it cites specific, verifiable evidence ("High-press tracking shows," "increased...by 18%") to support its claim. The model likely misclassified this as a 'Hot-Take' due to the strong, declarative nature of the statement about the "physical toll" and "increased injuries." While data-backed, the phrasing might have led the model to interpret it as a bold, attention-grabbing assertion rather than a neutral, evidence-based observation. This highlights a challenge in distinguishing between assertive analytical statements and unsupported hot-takes based solely on wording.

Example 2: Reaction misclassified as Hot-Take
Original Text: "Missing that absolute sitter in the 89th minute is going to haunt this squad for the rest of the group stage."
True Label: Reaction
Predicted Label: Hot-Take (Confidence: 0.44)
Explanation: This text is an immediate, emotionally charged response to a specific in-game event ("Missing that absolute sitter"). It expresses disappointment and an opinion about future consequences stemming directly from that moment. The true label is 'Reaction' as it's driven by real-time emotion. The model, however, labeled it a 'Hot-Take', possibly because the statement "is going to haunt this squad..." is a strong, somewhat speculative claim about the future. It struggled to prioritize the emotional, event-driven nature over the predictive, albeit unsupported, element.

Example 3: Reaction misclassified as Hot-Take
Original Text: "Haaland is a beast. Looking forward to the Norway vs France match."
True Label: Reaction
Predicted Label: Hot-Take (Confidence: 0.41)
Explanation: This post is a short, enthusiastic expression of personal sentiment and anticipation. "Haaland is a beast" is a common idiom used to express admiration and excitement, making the post primarily a 'Reaction'. The follow-up, "Looking forward to the Norway vs France match," further reinforces its reactive and personal nature. The model's classification as 'Hot-Take' suggests it interpreted "Haaland is a beast" as a bold, unsupported claim rather than a simple expression of fan excitement. This indicates difficulty in discerning general enthusiastic remarks from actual 'Hot-Take' statements that propose a specific, debatable idea without evidence.
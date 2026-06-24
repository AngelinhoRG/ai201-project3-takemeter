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
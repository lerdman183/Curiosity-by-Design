# Curiosity by Design: An LLM-based Coding Assistant Asking Clarification Questions

## Datasets

### `gerrit-cleaning/filtered_questions.json`

A dataset of real developer code comments and questions for a given piece of code formatted into prompt(code) and question pairs. Used in training the llama clarification module in clarification_module.py.

### `classifier_train_dataset.json`

A synthetic labeled dataset of user prompts with accompanying metadata indicating the level of ambiguity or need for clarification (on a graded scale from 1 to 4). Intended for training the Intent Classifier to detect whether a prompt requires a clarification question.


## Training Scripts

### `intent_classifier.py`

Training script for the Intent Classifier Module, responsible for grading ambiguity of user prompts on a scale of 1 to 4.

### `clarification_module.py`

Training script for the Clarification Module, responsible for asking clarification questions to user prompts flagged as ambigious or under-specified by the Intent Classifier Module.


## Testing Scripts

### `test_classifier.py`

Testing script for the Intent Classifier Module on data it has not encountered in the training phase.

### `test_clarification_module.py`

Testing script for the Clarification Module on data it has not encountered in the training phase.
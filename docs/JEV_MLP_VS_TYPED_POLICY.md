# Generic MLP versus typed policy

This ablation separates generic function approximation from semantic action
conditioning.

- **Question-conditioned generic MLP:** state plus question embedding, one
  ordinary output head over the legal actions.
- **Action-conditioned scorer without question:** semantic action tokens but no
  question embedding.
- **Question-conditioned fixed head:** question-conditioned fixed action head
  without the full shared compatibility interaction.
- **Full JEV:** shared state encoder, question embedding, semantic action
  embeddings, compatibility scorer, and runtime legal-action masking.

The comparison must report decision NLL, Brier, ECE, risk-coverage,
best-action accuracy, action frequencies, and final tracking metrics. Training
and calibration are performed on the same policy train/validation records;
official test is held until the selection lock.


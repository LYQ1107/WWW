"""Use the same native constrained solver; never introduce a new candidate."""
from gtr.modeling.jev_stage2.assignment import lawful_choice


def commit_choice(logits, legal):
    return lawful_choice(logits, legal)

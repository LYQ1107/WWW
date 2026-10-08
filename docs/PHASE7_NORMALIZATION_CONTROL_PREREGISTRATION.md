# Additional strong MLP control — before reading new heldout outcomes

Source-only reason: B2 uses LayerNorm twice on a state with unbounded raw accumulated GMT scores;
the first generic MLP does not. Add a second ordinary MLP with two LayerNorm/GELU layers:
64→152→152→3, **34,203 parameters**, 0.36% above B2. Same exact historical records, weights,
three actions, optimizer, epochs, updates, seed and calibration; no architecture or width search.

Retain the originally preregistered unnormalized MLP. The additional normalized MLP receives own
native validator, common legacy validator, and the same B2-count budget diagnostic on all three
frozen heldouts, plus the identical-start paired interventions. No candidate-conditioned or new
B2 architecture is fitted. Include both MLPs in the attribution table; require JEV's architecture
gate against the normalized MLP as well. Do not claim the generic MLP family is ineffective if
only one controller fails. This addition follows the source audit, not heldout outcomes.

WHAT DID WE LEARN? Capacity and input equality do not ensure equal optimization conditions;
normalization is a relevant ordinary baseline property on the inherited state representation.

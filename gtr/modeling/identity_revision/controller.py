"""Phase-0 controller. Unsupported actions fail closed before any state write."""


class IdentityRevisionController:
    def __init__(self, cfg):
        self.enabled = bool(cfg.ENABLED)
        self.mode = str(cfg.MODE)
        if self.enabled and self.mode not in ('off', 'keep'):
            raise ValueError('CIR mode has not passed native qualification: ' + self.mode)
        if cfg.ENABLE_PROVISIONAL or cfg.ENABLE_REVISION or cfg.ENABLE_RELINK or cfg.REVISION_WINDOW:
            raise ValueError('CIR revision/buffer not qualified in Phase 0')
        self.reset()

    def reset(self):
        self.calls = 0

    def propose(self, native_scores):
        # KEEP must preserve the very same tensor and consume no RNG.
        self.calls += 1
        return native_scores

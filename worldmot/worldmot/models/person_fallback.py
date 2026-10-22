"""Compatibility import for the former optional detector module.

New code should import :mod:`worldmot.models.person_detector`.  This alias
prevents downstream code from accidentally retaining the old no-op class.
"""

from .person_detector import PersonDetection, PersonDetector

PersonFallback = PersonDetector

__all__ = ["PersonDetection", "PersonDetector", "PersonFallback"]

"""Deflection -> dataset class mapping."""
import config


def classify_turn(total_deflection: float):
    """
    Maps accumulated horizontal deflection to a dataset class:
      - < DEFLECTION_DISCARD        -> None (negligible drift, discarded)
      - < DEFLECTION_MINOR_MAX      -> "Minor"
      - < DEFLECTION_MAJOR_MAX      -> "Major"
      - otherwise                   -> "Sharp"
    """
    abs_def = abs(total_deflection)

    if abs_def < config.DEFLECTION_DISCARD:
        return None
    if abs_def < config.DEFLECTION_MINOR_MAX:
        return "Minor"
    if abs_def < config.DEFLECTION_MAJOR_MAX:
        return "Major"
    return "Sharp"

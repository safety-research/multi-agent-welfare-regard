"""Judge registry. Every judge is a class with `name`, `version`, and `async judge(rec, model, cache_dir)`."""

from rwr.judges.base import Judge, available_judges, get_judge, judge_call

__all__ = ["Judge", "available_judges", "get_judge", "judge_call"]

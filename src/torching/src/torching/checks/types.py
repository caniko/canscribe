from __future__ import annotations

from typing import Literal, Optional


Status = Literal["OK", "WARN", "FAIL", "INFO"]


class CheckResult:
    __slots__ = ("status", "name", "detail", "explanation", "fix")

    def __init__(
        self,
        status: Status,
        name: str,
        detail: str,
        explanation: Optional[str] = None,
        fix: Optional[str] = None,
    ) -> None:
        self.status = status
        self.name = name
        self.detail = detail
        self.explanation = explanation
        self.fix = fix

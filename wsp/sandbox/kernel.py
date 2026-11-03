"""Persistent JSON-lines Python kernel executed inside the sandbox."""

from __future__ import annotations

import io
import json
import signal
import sys
import traceback
from typing import Any


class ExecutionTimedOut(TimeoutError):
    pass


class LimitedTextIO(io.TextIOBase):
    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)
        self.parts: list[str] = []
        self.length = 0
        self.truncated = False

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        value = str(value)
        available = self.limit - self.length
        if available > 0:
            kept = value[:available]
            self.parts.append(kept)
            self.length += len(kept)
        if len(value) > max(0, available):
            self.truncated = True
        return len(value)

    def getvalue(self) -> str:
        value = "".join(self.parts)
        if self.truncated:
            value += "\n[output capture truncated by sandbox]"
        return value


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise ExecutionTimedOut("execution timed out")


def _safe_budget(wsp: Any) -> dict[str, Any] | None:
    try:
        value = wsp.budget()
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def main() -> None:
    import wsp_sdk as wsp

    submitted = {"value": False}
    original_submit = getattr(wsp, "submit", None)
    if callable(original_submit):
        def tracked_submit(*args: Any, **kwargs: Any) -> Any:
            result = original_submit(*args, **kwargs)
            submitted["value"] = True
            return result

        wsp.submit = tracked_submit

    namespace: dict[str, Any] = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "wsp": wsp,
    }
    signal.signal(signal.SIGALRM, _timeout_handler)
    print(json.dumps({"event": "ready"}), flush=True)

    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if request.get("op") == "shutdown":
                print(json.dumps({"id": request.get("id"), "ok": True}), flush=True)
                return
            if request.get("op") != "exec":
                raise ValueError("unsupported kernel operation")

            request_id = request.get("id")
            code = str(request.get("code", ""))
            timeout_s = max(1.0, float(request.get("timeout_s", 300.0)))
            output_limit = max(1, int(request.get("max_output_chars", 100_000)))
            stdout = LimitedTextIO(output_limit)
            stderr = LimitedTextIO(output_limit)
            ok = True
            trace = ""
            old_stdout, old_stderr = sys.stdout, sys.stderr
            try:
                sys.stdout, sys.stderr = stdout, stderr
                signal.setitimer(signal.ITIMER_REAL, timeout_s)
                exec(compile(code, "<agent-turn>", "exec"), namespace, namespace)
            except BaseException:
                ok = False
                trace = traceback.format_exc()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                sys.stdout, sys.stderr = old_stdout, old_stderr

            budget = None if submitted["value"] else _safe_budget(wsp)
            response = {
                "id": request_id,
                "ok": ok,
                "stdout": stdout.getvalue(),
                "stderr": stderr.getvalue(),
                "traceback": trace,
                "submitted": submitted["value"],
                "budget": budget,
                "restarted": False,
            }
        except BaseException:
            response = {
                "id": locals().get("request_id"),
                "ok": False,
                "stdout": "",
                "stderr": "",
                "traceback": traceback.format_exc(),
                "submitted": submitted["value"],
                "budget": None,
                "restarted": False,
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

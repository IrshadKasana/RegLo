"""Name -> constructor registries, connecting the YAML ``name:`` fields to code.

    from reglo.registry import OPTIMIZEES

    @OPTIMIZEES.register("convnet4")
    class ConvNet4(nn.Module):
        ...

    model = OPTIMIZEES.build(cfg.optimizee.name, cfg.optimizee)

An unknown name raises at once and lists the valid ones, so a misspelt config
can never fall back to a default without anyone noticing.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, TypeVar

__all__ = ["DATASETS", "OPTIMIZEES", "OPTIMIZERS", "REGULARIZERS", "Registry", "UnknownNameError"]

T = TypeVar("T", bound=Callable[..., Any])


class UnknownNameError(KeyError):
    """A name that no module has registered."""

    def __str__(self) -> str:  # KeyError would wrap the message in quotes
        return str(self.args[0]) if self.args else ""


class Registry:
    """A named collection of classes or factory functions of one kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator that records ``obj`` under ``name``."""

        def decorator(obj: T) -> T:
            existing = self._entries.get(name)
            if existing is not None and existing is not obj:
                raise KeyError(f"{self.kind} '{name}' is already registered by {existing!r}")
            self._entries[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._entries[name]
        except KeyError:
            known = ", ".join(self.names()) or "none yet (is the defining module imported?)"
            raise UnknownNameError(f"Unknown {self.kind} '{name}'. Registered: {known}") from None

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Look up ``name`` and call it with the given arguments."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._entries)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


DATASETS = Registry("dataset")
OPTIMIZEES = Registry("optimizee")
OPTIMIZERS = Registry("optimizer")
REGULARIZERS = Registry("regularizer")

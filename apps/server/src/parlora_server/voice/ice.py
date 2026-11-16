"""ICE on a published address (design D11, amends ADR 0003).

Where the container cannot reach the browser's candidates, the browser must
start the ICE checks against an address it can reach: a UDP port range
published on the host's LAN address. Two pieces make that work:

- `ice_port_range` makes aioice bind its host sockets inside the published
  range. aioice always binds port 0, so its `asyncio` name is swapped for a
  shim whose event loop picks the port. Only `aioice.ice` sees the shim; the
  rest of the process keeps the real loop.
- `rewrite_host_candidates` puts the published address in the answer's host
  candidates (and connection line), in place of the container's own address.
"""

import asyncio
import re
import types
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import aioice.ice


class IcePortRangeExhaustedError(OSError):
    """Every port of the ICE range is taken: the call cannot gather candidates."""


class _PortRangeLoop:
    """An event loop proxy whose ephemeral datagram endpoints bind inside `ports`."""

    def __init__(self, loop: asyncio.AbstractEventLoop, ports: range) -> None:
        self._loop = loop
        self._ports = ports

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loop, name)

    async def create_datagram_endpoint(
        self,
        protocol_factory: Callable[[], asyncio.BaseProtocol],
        local_addr: tuple[str, int] | None = None,
        **kwargs: Any,
    ) -> tuple[asyncio.BaseTransport, asyncio.BaseProtocol]:
        if local_addr is None or local_addr[1] != 0:
            return await self._loop.create_datagram_endpoint(
                protocol_factory, local_addr=local_addr, **kwargs
            )
        host = local_addr[0]
        for port in self._ports:
            try:
                return await self._loop.create_datagram_endpoint(
                    protocol_factory, local_addr=(host, port), **kwargs
                )
            except OSError:
                continue
        raise IcePortRangeExhaustedError(
            f"no free UDP port in {self._ports.start}-{self._ports.stop - 1}"
        )


# The name `aioice.ice` imports asyncio under; swapped, never the real module.
_AIOICE_ASYNCIO = "asyncio"


def _asyncio_shim(ports: range) -> types.ModuleType:
    shim = types.ModuleType("asyncio")

    def get_event_loop() -> _PortRangeLoop:
        return _PortRangeLoop(asyncio.get_event_loop(), ports)

    def passthrough(name: str) -> Any:
        return getattr(asyncio, name)

    setattr(shim, "get_event_loop", get_event_loop)  # noqa: B010 - ModuleType has no such slot
    setattr(shim, "__getattr__", passthrough)  # noqa: B010 - module-level __getattr__ (PEP 562)
    return shim


def _swap_aioice_asyncio(module: types.ModuleType) -> types.ModuleType:
    previous: types.ModuleType = getattr(aioice.ice, _AIOICE_ASYNCIO)
    setattr(aioice.ice, _AIOICE_ASYNCIO, module)
    return previous


@contextmanager
def ice_port_range(ports: range) -> Iterator[None]:
    """While active, aioice's host sockets bind inside `ports` (scoped to `aioice.ice`)."""
    previous = _swap_aioice_asyncio(_asyncio_shim(ports))
    try:
        yield
    finally:
        _swap_aioice_asyncio(previous)


def install_ice_port_range(ports: range) -> None:
    """Keeps aioice's host sockets inside `ports` for the life of the process."""
    _swap_aioice_asyncio(_asyncio_shim(ports))


_HOST_CANDIDATE = re.compile(
    r"^(a=candidate:\S+ \d+ udp \d+ )(\S+) (\d+)( typ host\b.*)$", re.IGNORECASE
)
_CONNECTION = re.compile(r"^c=IN IP4 \S+$")


def rewrite_host_candidates(sdp: str, address: str, ports: range) -> str:
    """The SDP with `address` in its host candidates whose port is in `ports`."""
    newline = "\r\n" if "\r\n" in sdp else "\n"
    lines = sdp.split(newline)
    rewritten = False
    for i, line in enumerate(lines):
        match = _HOST_CANDIDATE.match(line)
        if match and int(match.group(3)) in ports:
            lines[i] = f"{match.group(1)}{address} {match.group(3)}{match.group(4)}"
            rewritten = True
    if rewritten:
        lines = [f"c=IN IP4 {address}" if _CONNECTION.match(line) else line for line in lines]
    return newline.join(lines)

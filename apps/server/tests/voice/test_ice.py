"""ICE on a published address (design D11): port range and candidate rewrite."""

import asyncio
import socket
from collections.abc import Iterator

import aioice.ice
import pytest

from parlora_server.voice.ice import (
    IcePortRangeExhaustedError,
    ice_port_range,
    rewrite_host_candidates,
)

pytestmark = pytest.mark.unit

ANSWER = (
    "v=0\r\n"
    "o=- 3912 3912 IN IP4 0.0.0.0\r\n"
    "m=audio 40002 UDP/TLS/RTP/SAVPF 111\r\n"
    "c=IN IP4 172.18.0.3\r\n"
    "a=candidate:9f3 1 udp 2130706431 172.18.0.3 40002 typ host\r\n"
    "a=candidate:7aa 1 udp 2130706431 172.18.0.3 53111 typ host\r\n"
    "a=candidate:1b2 1 udp 1694498815 203.0.113.5 40003 typ srflx raddr 172.18.0.3 rport 40003\r\n"
    "a=end-of-candidates\r\n"
)


class TestRewriteHostCandidates:
    def test_rewrites_host_candidates_in_the_range_and_the_connection_line(self) -> None:
        rewritten = rewrite_host_candidates(ANSWER, "192.0.2.10", range(40000, 40020))

        lines = rewritten.split("\r\n")
        assert "a=candidate:9f3 1 udp 2130706431 192.0.2.10 40002 typ host" in lines
        assert "c=IN IP4 192.0.2.10" in lines

    def test_leaves_every_other_line_unchanged(self) -> None:
        rewritten = rewrite_host_candidates(ANSWER, "192.0.2.10", range(40000, 40020))

        before, after = ANSWER.split("\r\n"), rewritten.split("\r\n")
        changed = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
        assert [a for a, _ in changed] == [
            "c=IN IP4 172.18.0.3",
            "a=candidate:9f3 1 udp 2130706431 172.18.0.3 40002 typ host",
        ]

    def test_keeps_line_endings(self) -> None:
        rewritten = rewrite_host_candidates(
            ANSWER.replace("\r\n", "\n"), "192.0.2.10", range(40000, 40020)
        )

        assert "\r" not in rewritten
        assert "192.0.2.10 40002 typ host" in rewritten


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


@pytest.fixture
def ports() -> Iterator[range]:
    """Three consecutive free UDP ports on loopback."""
    for start in range(45000, 45900, 7):
        candidate = range(start, start + 3)
        if all(_port_free(p) for p in candidate):
            yield candidate
            return
    pytest.skip("no free UDP port range on this machine")


async def _gather_port() -> int:
    """Binds a host socket the way aioice does and returns its port."""
    loop = aioice.ice.asyncio.get_event_loop()
    transport, _ = await loop.create_datagram_endpoint(
        asyncio.DatagramProtocol, local_addr=("127.0.0.1", 0)
    )
    try:
        return int(transport.get_extra_info("sockname")[1])
    finally:
        transport.close()


class TestIcePortRange:
    def test_aioice_host_sockets_bind_inside_the_range(self, ports: range) -> None:
        with ice_port_range(ports):
            port = asyncio.run(_gather_port())

        assert port == ports[0]

    def test_a_taken_port_moves_to_the_next_one(self, ports: range) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as taken:
            taken.bind(("127.0.0.1", ports[0]))
            with ice_port_range(ports):
                port = asyncio.run(_gather_port())

        assert port == ports[1]

    def test_a_full_range_fails_instead_of_using_a_random_port(self, ports: range) -> None:
        held = []
        try:
            for p in ports:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.bind(("127.0.0.1", p))
                held.append(s)
            with ice_port_range(ports), pytest.raises(IcePortRangeExhaustedError):
                asyncio.run(_gather_port())
        finally:
            for s in held:
                s.close()

    def test_restores_aioice_and_leaves_other_code_alone(self, ports: range) -> None:
        original = aioice.ice.asyncio
        with ice_port_range(ports):
            assert asyncio.get_event_loop_policy() is not None
            assert aioice.ice.asyncio.Event is asyncio.Event, "other names pass through"

            async def outside_aioice() -> int:
                loop = asyncio.get_running_loop()
                transport, _ = await loop.create_datagram_endpoint(
                    asyncio.DatagramProtocol, local_addr=("127.0.0.1", 0)
                )
                port = int(transport.get_extra_info("sockname")[1])
                transport.close()
                return port

            assert asyncio.run(outside_aioice()) not in ports
        assert aioice.ice.asyncio is original

    def test_explicit_ports_are_not_redirected(self, ports: range) -> None:
        async def explicit() -> int:
            loop = aioice.ice.asyncio.get_event_loop()
            port = ports[2]
            transport, _ = await loop.create_datagram_endpoint(
                asyncio.DatagramProtocol, local_addr=("127.0.0.1", port)
            )
            transport.close()
            return port

        with ice_port_range(ports):
            assert asyncio.run(explicit()) == ports[2]

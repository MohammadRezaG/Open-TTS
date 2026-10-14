"""IP address parsing shared by API and UI access checks."""

import ipaddress
from typing import Iterable


def client_ip(host: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if not host:
        return None
    try:
        return ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return None


def is_loopback(host: str | None) -> bool:
    address = client_ip(host)
    return address is not None and address.is_loopback


def is_network_allowed(host: str | None, networks: Iterable[str]) -> bool:
    address = client_ip(host)
    if address is None:
        return False
    if address.is_loopback:
        return True
    for value in networks:
        try:
            if address in ipaddress.ip_network(value, strict=False):
                return True
        except ValueError as exc:
            raise ValueError(f"Invalid allowed network CIDR: {value}") from exc
    return False

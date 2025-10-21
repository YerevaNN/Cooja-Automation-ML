from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    mote_type: str
    target: str
    server_binary: str
    client_binary: str
    interfaces: List[str] = field(default_factory=list)


PLATFORMS: Dict[str, PlatformSpec] = {
    "z1": PlatformSpec(
        name="z1",
        mote_type="org.contikios.cooja.mspmote.Z1MoteType",
        target="z1",
        server_binary="battery_server.z1",
        client_binary="battery_client.z1",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.RimeAddress",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspDefaultSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput"
        ],
    ),
    # Preview entries below; validate in your environment and add other interfaces as needed before production sweeps.
    "sky": PlatformSpec(
        name="sky",
        mote_type="org.contikios.cooja.mspmote.SkyMoteType",
        target="sky",
        server_binary="battery_server.sky",
        client_binary="battery_client.sky",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.RimeAddress",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.SkyButton",
            "org.contikios.cooja.mspmote.interfaces.SkyFlash",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspDefaultSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput",
            "org.contikios.cooja.mspmote.interfaces.SkyLED"
        ],
    ),
    "wismote": PlatformSpec(
        name="wismote",
        mote_type="org.contikios.cooja.mspmote.WismoteMoteType",
        target="wismote",
        server_binary="battery_server.wismote",
        client_binary="battery_client.wismote",
        interfaces=[
            "org.contikios.cooja.interfaces.Position",
            "org.contikios.cooja.interfaces.RimeAddress",
            "org.contikios.cooja.interfaces.IPAddress",
            "org.contikios.cooja.interfaces.Mote2MoteRelations",
            "org.contikios.cooja.interfaces.MoteAttributes",
            "org.contikios.cooja.mspmote.interfaces.MspClock",
            "org.contikios.cooja.mspmote.interfaces.MspMoteID",
            "org.contikios.cooja.mspmote.interfaces.Msp802154Radio",
            "org.contikios.cooja.mspmote.interfaces.MspDefaultSerial",
            "org.contikios.cooja.mspmote.interfaces.MspLED",
            "org.contikios.cooja.mspmote.interfaces.MspDebugOutput"
        ],
    ),
}


def get_platform(name: str) -> PlatformSpec:
    key = (name or "z1").lower()
    if key not in PLATFORMS:
        raise ValueError(f"Unknown platform '{name}'. Available: {sorted(PLATFORMS)}")
    return PLATFORMS[key]



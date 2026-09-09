"""Check server identity/contract without ROS, calibration, sessions, or motion."""
from __future__ import annotations

import argparse
import json

from tau0vla_calibrated_protocol import CalibratedHttpClient, ProtocolError


def check_server(server_url: str, *, route: str, experiment: str, protocol_version: str):
    client = CalibratedHttpClient(
        server_url, experiment=experiment, protocol_version=protocol_version,
        calibration=None, robot_id="preflight-only",
    )
    try:
        health = client.health()
        if health.get("route") != route:
            raise ProtocolError(f"route {health.get('route')!r} does not match {route!r}")
        contract = client.policy_contract()
        if contract.get("model_id") != health.get("model_id"):
            raise ProtocolError("model changed between health and contract checks")
        if contract.get("checkpoint_sha256") != health.get("checkpoint_sha256"):
            raise ProtocolError("checkpoint changed between health and contract checks")
        if contract.get("rtc_enabled", False):
            raise ProtocolError("calibrated inference requires RTC disabled")
        return {"health": health, "contract": contract}
    finally:
        client.session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--protocol-version", required=True)
    args = parser.parse_args()
    try:
        result = check_server(args.server_url, route=args.route, experiment=args.experiment,
                              protocol_version=args.protocol_version)
    except Exception as error:
        parser.exit(1, f"SERVER_PREFLIGHT_REFUSED: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    print("SERVER_PREFLIGHT_OK (read-only; no robot hardware or session started)")


if __name__ == "__main__":
    main()

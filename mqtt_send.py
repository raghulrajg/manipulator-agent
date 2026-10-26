"""
mqtt_send.py
-------------
Send a command to the manipulator agent (the digital twin started with
--mqtt) by hand, the way the vision agent does, and print its replies.
Same broker settings as the rest (MQTT_BROKER_HOST, MQTT_BROKER_PORT,
MQTT_USERNAME, MQTT_PASSWORD, MQTT_TLS, MQTT_CA_FILE, MQTT_TRANSPORT) or
the options below; works with EMQX, Mosquitto or any MQTT 3.1.1 broker.

    python3 mqtt_send.py idle  --position 0.02 -0.03 0.30 0     # just show the object in the twin
    python3 mqtt_send.py start pick  --position 0.02 -0.03 0.30 0 --product tomato
    python3 mqtt_send.py start place --position -0.10 0.05 0.32 0
    python3 mqtt_send.py stop
    python3 mqtt_send.py --json '{"state":"start","product":"tomato","action":"pick","position":[0.02,-0.03,0.30,0]}'

position: x y z w in the wrist camera's frame (meters; x right, y down,
z along the view, as FoundationPose), w the turn about the view axis in
degrees. Moving needs the twin armed in a browser.

The same JSON can be sent from the EMQX dashboard (WebSocket client),
MQTTX or mosquitto_pub to the topic manipulator/command.
"""

import argparse
import json
import queue
import sys
import time
import uuid

from mqtt_link import COMMAND_TOPIC, STATUS_TOPIC, MessageError, MqttLink, command_message, parse_command


def main():
    p = argparse.ArgumentParser(description="Send a command to the manipulator agent over MQTT.")
    p.add_argument("state", nargs="?", choices=("start", "stop", "idle"))
    p.add_argument("action", nargs="?", choices=("pick", "place"))
    p.add_argument("--position", type=float, nargs=4, metavar=("X", "Y", "Z", "W"),
                   help="camera frame: x y z [m], w [deg]")
    p.add_argument("--product", default="")
    p.add_argument("--json", help="send this JSON as it is instead")
    p.add_argument("--wait", type=float, default=10.0, help="print replies for this many seconds (default 10)")
    p.add_argument("--host", help="broker (default MQTT_BROKER_HOST or localhost)")
    p.add_argument("--port", type=int, help="default MQTT_BROKER_PORT or 1883")
    p.add_argument("--username")
    p.add_argument("--password")
    p.add_argument("--tls", action="store_true", default=None, help="TLS (on by default on port 8883)")
    p.add_argument("--ca-file", help="the broker's CA certificate (TLS)")
    args = p.parse_args()

    if args.json:
        text = args.json
    elif args.state:
        text = json.dumps(command_message(args.state, args.action, args.position, args.product))
    else:
        p.error("give a state (start / stop / idle) or --json")
    try:
        parse_command(text)   # what the agent will check; refuse here, before sending
    except MessageError as e:
        p.error(str(e))

    link = MqttLink(client_id=f"mqtt_send_{uuid.uuid4().hex[:6]}", host=args.host, port=args.port,
                    username=args.username, password=args.password, tls=args.tls, ca_file=args.ca_file)
    replies = link.subscribe(STATUS_TOPIC)
    try:
        link.connect()
    except (ConnectionError, OSError) as e:
        sys.exit(f"MQTT: {e}")
    print(f"{link.host}:{link.port} {COMMAND_TOPIC} <- {text}")
    link.publish(COMMAND_TOPIC, text)
    end, final = time.monotonic() + args.wait, False
    try:
        while not final and time.monotonic() < end:
            try:
                reply = replies.get(timeout=max(0.05, end - time.monotonic()))
            except queue.Empty:
                break
            text = reply.decode("utf-8", errors="replace")
            print(f"{STATUS_TOPIC} -> {text}")
            try:
                final = json.loads(text).get("state") in ("done", "failed", "stopped", "refused", "updated", "idle")
            except (ValueError, AttributeError):
                pass
        if not final:
            print(f"(no final reply in {args.wait:.0f} s: is digital_twin.py running with --mqtt?)")
    finally:
        link.close()


if __name__ == "__main__":
    main()

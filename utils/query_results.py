"""
utils/query_results.py
--------------------------------------------------------------
CLI tool to inspect the face tracker database.

Usage:
  python utils/query_results.py             # summary
  python utils/query_results.py --count     # unique visitor count only
  python utils/query_results.py --faces     # all registered faces
  python utils/query_results.py --events 30 # last 30 events
  python utils/query_results.py --export out.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db_manager import DatabaseManager


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="config/config.json")
    p.add_argument("--count",   action="store_true")
    p.add_argument("--faces",   action="store_true")
    p.add_argument("--events",  type=int, metavar="N")
    p.add_argument("--export",  metavar="FILE")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    db = DatabaseManager(cfg["database"]["path"])

    if args.count:
        print(f"\n  UNIQUE VISITORS: {db.get_unique_visitor_count()}\n")
        return

    if args.faces:
        faces = db.get_all_faces()
        print(f"\n{'─'*72}")
        print(f"  REGISTERED FACES  ({len(faces)} total)")
        print(f"{'─'*72}")
        print(f"{'Face ID':<32} {'First Seen':<21} {'Visits'}")
        print(f"{'─'*72}")
        for f in faces:
            print(f"{f['id']:<32} {f['first_seen'][:19]:<21} {f['visit_count']}")
        print()
        return

    if args.events is not None:
        evs = db.get_recent_events(args.events)
        print(f"\n{'─'*82}")
        print(f"  RECENT EVENTS  (last {args.events})")
        print(f"{'─'*82}")
        print(f"{'ID':<5} {'Type':<12} {'Face ID':<32} {'Timestamp':<21} {'Conf'}")
        print(f"{'─'*82}")
        for ev in reversed(evs):
            print(f"{ev['id']:<5} {ev['event_type']:<12} {ev['face_id']:<32} "
                  f"{ev['timestamp'][:19]:<21} {ev['confidence']:.2f}")
        print()
        return

    if args.export:
        evs = db.get_recent_events(100_000)
        with open(args.export, "w") as f:
            json.dump(evs, f, indent=2)
        print(f"Exported {len(evs)} events to {args.export}")
        return

    # default summary
    st = db.get_stats()
    print(f"""
╔══════════════════════════════════════════╗
║     FACE TRACKER — SESSION SUMMARY       ║
╠══════════════════════════════════════════╣
║  Unique Visitors : {st.get('unique_visitors','0'):<22} ║
║  Total Entries   : {st.get('total_entries','0'):<22} ║
║  Total Exits     : {st.get('total_exits','0'):<22} ║
║  Session Start   : {st.get('session_start','N/A')[:19]:<22} ║
╚══════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()

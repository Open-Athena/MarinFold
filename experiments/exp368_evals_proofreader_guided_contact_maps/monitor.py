"""Read the shared worker's durable state and preserve monitoring evidence."""

import argparse
import json
from datetime import UTC, datetime

from runtime import HERE, ROOT, load_release


def main() -> None:
    """Print compact, actionable task state; keep full logs in the saved snapshot."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', default=ROOT + '/server-v1')
    args = parser.parse_args()
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(args.control)
    state = json.loads(fs.cat(root + '/status.json'))
    snapshot = dict(observed_at=datetime.now(UTC).isoformat(), control=args.control, **state)
    path = HERE / '_cache/server_status.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2))
    print(f"active={len(state['active'])} finished={len(state['finished'])} "
          f"artifact_transfer_GB={state['artifact_transfer_bytes']/1e9:.3f}")
    for name, item in state['active'].items():
        tail = item.get('log_tail', '').strip().splitlines()
        progress = [line for line in tail if '[exp368]' in line]
        print(name, 'gpu=' + str(item['gpu']), (progress or tail or ['starting'])[-1][-240:])
    for name, item in state['finished'].items():
        print(name, item['state'], 'exit=' + str(item['exit_code']))


if __name__ == '__main__':
    main()

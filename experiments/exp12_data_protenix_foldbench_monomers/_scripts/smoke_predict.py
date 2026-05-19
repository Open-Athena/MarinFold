"""1-protein x 1-mode x 1-seed x 1-sample smoke test on Modal.

Validates the full inference wiring (image build, weights symlink,
config construction, InferenceRunner load, distogram hook fire,
output copy to Volume) before kicking off the real 10-protein run.

Uses 5sbj_A (smallest at 30 aa), single_seq mode (no MSA needed),
seed=1, n_sample=1, n_cycle=10. Expected time: ~3-5 min on H100.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal_app


def main() -> None:
    stem = "5sbj_A"
    job_path = Path("inputs/jobs") / f"{stem}.json"
    job_str = job_path.read_text()

    print(f"Smoke: {stem} single_seq seed=1 n_sample=1 n_cycle=10")
    with modal_app.app.run():
        worker = modal_app.ProtenixWorker()
        result = worker.predict_one.remote(
            job_json_str=job_str,
            stem=stem,
            mode="single_seq",
            seeds=[1],
            n_sample=1,
            n_cycle=10,
        )
    print("Result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

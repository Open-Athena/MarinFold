"""Independent fixtures for the distinction between proximity and correction."""

import unittest

from analyze import analyze_one


class ContactCorrectionTest(unittest.TestCase):
    """Check that oracle matching cannot reuse an already-recovered contact."""

    def item(self, votes: list[list[int]]) -> dict:
        return {"truth": {"stem": "fixture", "L": 40, "resolved": list(range(40)),
                          "contacts": [[0, 10, 0.5], [20, 30, 0.5]]},
                "votes": votes, "is_viral": False}

    def test_nearby_duplicate_cannot_rescue_missing_remote_contact(self) -> None:
        row, _, _ = analyze_one(self.item([[0, 10, 90], [1, 11, 80], [20, 30, 1]]))
        self.assertEqual(row["r_precision"], 0.5)
        self.assertEqual(row["near_2_precision"], 1.0)
        self.assertEqual(row["one_to_one_2_precision"], 0.5)
        self.assertEqual(row["true_seen_fraction"], 1.0)

    def test_distinct_shifted_pairs_can_each_match_once(self) -> None:
        row, _, _ = analyze_one(self.item([[1, 11, 90], [21, 31, 80]]))
        self.assertEqual(row["r_precision"], 0.0)
        self.assertEqual(row["one_to_one_1_precision"], 1.0)
        self.assertEqual(row["true_seen_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()

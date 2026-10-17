import unittest

from training.card_recognizer.identity_evidence import assess_identity_evidence, extract_printed_codes


def token(game, kind, value):
    return {"game": game, "kind": kind, "token": value}


class PrintedCodeTests(unittest.TestCase):
    def test_extracts_game_scoped_codes_without_correcting_characters(self):
        self.assertEqual(
            extract_printed_codes("OP01-001 / ST-12-003; Pokemon TG01 / 30; LOB-EN001 12345678"),
            [
                token("onepiece", "printed_number", "OP01-001"),
                token("onepiece", "printed_number", "ST-12-003"),
                token("pokemon", "collector_number", "TG01/30"),
                token("yugioh", "set_code", "LOB-EN001"),
                token("yugioh", "passcode", "12345678"),
            ],
        )

    def test_plain_pokemon_number_is_not_unique_evidence(self):
        self.assertEqual(extract_printed_codes("card 12; 12/198"), [token("pokemon", "collector_number", "12/198")])

    def test_onepiece_promo_and_optional_set_prefix_hyphen(self):
        self.assertEqual(
            extract_printed_codes("P-001 OP-01-001"),
            [token("onepiece", "printed_number", "P-001"), token("onepiece", "printed_number", "OP-01-001")],
        )

    def test_pokemon_alphanumeric_denominator_is_preserved(self):
        self.assertEqual(
            extract_printed_codes("TG01/TG30 GG01/GG70"),
            [token("pokemon", "collector_number", "TG01/TG30"), token("pokemon", "collector_number", "GG01/GG70")],
        )


class EvidenceDecisionTests(unittest.TestCase):
    def test_duplicate_templates_collapse_before_margin(self):
        result = assess_identity_evidence(
            [
                {"identity": "op:OP01-001", "similarity": 0.91},
                {"identity": "op:OP01-001", "similarity": 0.90},
                {"identity": "op:OP01-002", "similarity": 0.80},
            ],
            "OP01-001",
            {"op:OP01-001": {"expected_printed_tokens": [token("onepiece", "printed_number", "OP01-001")] }},
            min_margin=0.05,
        )
        self.assertEqual(result["status"], "corroborated_candidate")
        self.assertAlmostEqual(result["margin"], 0.11)
        self.assertFalse(result["ground_truth_verified"])

    def test_conflicting_readable_ocr_is_not_overridden_by_similarity(self):
        result = assess_identity_evidence(
            [{"identity": "op:OP01-001", "similarity": 0.95}],
            "OP01-002",
            {"op:OP01-001": {"expected_printed_tokens": [token("onepiece", "printed_number", "OP01-001")]}},
        )
        self.assertEqual(result["status"], "conflicting_evidence")

    def test_pokemon_collector_number_requires_exact_denominator(self):
        result = assess_identity_evidence(
            [
                {"identity": "pokemon:sv1-12", "similarity": 0.90},
                {"identity": "pokemon:other", "similarity": 0.80},
            ],
            "12/198",
            {"pokemon:sv1-12": {"expected_printed_tokens": [token("pokemon", "collector_number", "12/198")] }},
        )
        self.assertEqual(result["status"], "corroborated_candidate")
        self.assertEqual(extract_printed_codes("12"), [])

    def test_onepiece_base_code_does_not_confirm_p1_variant(self):
        base = token("onepiece", "printed_number", "OP01-001")
        result = assess_identity_evidence(
            [{"identity": "op:OP01-001_p1", "similarity": 0.95}],
            "OP01-001",
            {
                "op:OP01-001_p1": {"expected_printed_tokens": [base], "printing_variant": "alternate_art"},
                "op:OP01-001": {"expected_printed_tokens": [base]},
            },
        )
        self.assertEqual(result["status"], "unresolved/visual_candidate")
        self.assertTrue(any("does not establish" in item for item in result["limitations"]))

    def test_exact_token_shared_by_catalog_identities_is_ambiguous(self):
        base = token("onepiece", "printed_number", "OP01-001")
        result = assess_identity_evidence(
            [
                {"identity": "op:OP01-001", "similarity": 0.95},
                {"identity": "op:OP01-002", "similarity": 0.75},
            ],
            "OP01-001",
            {
                "op:OP01-001": {"expected_printed_tokens": [base]},
                "op:OP01-001_p1": {"expected_printed_tokens": [base]},
            },
        )
        self.assertEqual(result["status"], "unresolved/visual_candidate")
        self.assertTrue(result["ambiguous_tokens"])

    def test_yugioh_passcode_alone_does_not_establish_printing(self):
        result = assess_identity_evidence(
            [
                {"identity": "ygo:LOB-EN001", "similarity": 0.94},
                {"identity": "ygo:other", "similarity": 0.80},
            ],
            "12345678",
            {"ygo:LOB-EN001": {"expected_printed_tokens": [token("yugioh", "passcode", "12345678")]}},
        )
        self.assertEqual(result["status"], "unresolved/visual_candidate")
        self.assertTrue(any("passcode alone" in item for item in result["limitations"]))

    def test_finite_negative_cosine_scores_are_accepted(self):
        result = assess_identity_evidence(
            [{"identity": "op:a", "similarity": -0.2}, {"identity": "op:b", "similarity": -0.4}],
            "",
            {},
        )
        self.assertEqual(result["best_similarity"], -0.2)

    def test_similarity_rejects_nonfinite_and_large_overflow(self):
        for invalid in (float("nan"), float("inf"), 1.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                assess_identity_evidence([{"identity": "a", "similarity": invalid}], "", {})

    def test_low_distinct_identity_margin_stays_unresolved(self):
        result = assess_identity_evidence(
            [
                {"identity": "op:OP01-001", "similarity": 0.82},
                {"identity": "op:OP01-002", "similarity": 0.80},
            ],
            "OP01-001",
            {"op:OP01-001": {"expected_printed_tokens": [token("onepiece", "printed_number", "OP01-001")]}},
            min_margin=0.05,
        )
        self.assertEqual(result["status"], "unresolved/visual_candidate")
        self.assertFalse(result["gates_pass"])


if __name__ == "__main__":
    unittest.main()

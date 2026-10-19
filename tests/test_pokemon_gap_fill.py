import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "catalog"))
from fill_pokemon_gallery_gaps import match_card, norm_name, norm_number  # noqa: E402


class GapFillMatchingTests(unittest.TestCase):
    def test_number_normalization_strips_leading_zeros_only(self):
        self.assertEqual(norm_number("SV001"), "SV1")
        self.assertEqual(norm_number("001"), "1")
        self.assertEqual(norm_number("146a"), "146A")
        self.assertEqual(norm_number("%3F"), "?")
        self.assertNotEqual(norm_number("50a"), norm_number("50b"))

    def test_name_normalization_ignores_punctuation_and_accents(self):
        self.assertEqual(norm_name("Dialga GX"), norm_name("Dialga-GX"))
        self.assertEqual(norm_name("Pokémon Center"), norm_name("Pokemon Center"))

    def test_requires_number_and_name_and_uniqueness(self):
        record = {"local_id": "SV001", "name": "Rowlet"}
        self.assertEqual(match_card(record, [{"number": "SV1", "name": "Rowlet"}])[0], "matched")
        self.assertEqual(match_card(record, [{"number": "SV1", "name": "Dartrix"}])[0], "name_mismatch")
        self.assertEqual(match_card(record, [{"number": "SV2", "name": "Rowlet"}])[0], "no_number_match")
        twins = [{"number": "SV001", "name": "Rowlet"}, {"number": "SV1", "name": "Rowlet"}]
        self.assertEqual(match_card(record, twins)[0], "ambiguous")

    def test_basic_energy_prefix_is_equivalent(self):
        self.assertEqual(norm_name("Basic Grass Energy"), norm_name("Grass Energy"))
        self.assertNotEqual(norm_name("Basic Grass Energy"), norm_name("Fire Energy"))

    def test_name_only_fallback_requires_unique_name(self):
        record = {"local_id": "CC002", "name": "Charizard"}
        pool = [{"number": "4", "name": "Charizard"}, {"number": "2", "name": "Blastoise"}]
        self.assertEqual(match_card(record, pool)[0], "no_number_match")
        self.assertEqual(match_card(record, pool, name_only_fallback=True)[0], "matched_by_unique_name")
        halves = [{"number": "1", "name": "Darkrai & Cresselia LEGEND"}, {"number": "2", "name": "Darkrai & Cresselia LEGEND"}]
        self.assertEqual(match_card({"local_id": "019", "name": "Darkrai & Cresselia LEGEND"}, halves, True)[0], "ambiguous")


if __name__ == "__main__":
    unittest.main()

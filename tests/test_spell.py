"""Tests for spell correction."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.spell import SpellCorrector, edit_distance


def test_edit_distance_identical():
    assert edit_distance("python", "python") == 0


def test_edit_distance_one_substitution():
    assert edit_distance("pythan", "python") == 1


def test_edit_distance_one_deletion():
    assert edit_distance("pythoon", "python") == 1


def test_edit_distance_one_insertion():
    assert edit_distance("pyton", "python") == 1


def test_edit_distance_transposition():
    assert edit_distance("pytohn", "python") == 2


def make_corrector(words):
    index = {w: {"doc_ids": list(range(10)), "tf": {}}
             for w in words}
    return SpellCorrector(index)


def test_spell_correct_typo():
    sc  = make_corrector(["distributed", "computing", "parallel", "python"])
    fix = sc.correct("distribted")
    assert fix == "distributed"


def test_spell_correct_valid_word_returns_none():
    sc  = make_corrector(["distributed", "computing"])
    fix = sc.correct("distributed")
    assert fix is None


def test_spell_correct_query_multi_word():
    sc = make_corrector(["distributed", "computing", "parallel", "python"])
    corrected, fixes = sc.correct_query("distribted compuing")
    assert "distributed" in corrected
    assert "computing"   in corrected
    assert len(fixes)    == 2


def test_spell_no_correction_for_short_words():
    sc  = make_corrector(["distributed", "computing"])
    fix = sc.correct("ab")
    assert fix is None
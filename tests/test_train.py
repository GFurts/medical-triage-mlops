from src.train import passes_quality_gate


def test_quality_gate_blocks_low_recall():
    assert passes_quality_gate({"urgent_recall": 0.85}, min_recall=0.80)
    assert passes_quality_gate({"urgent_recall": 0.80}, min_recall=0.80)
    assert not passes_quality_gate({"urgent_recall": 0.70}, min_recall=0.80)

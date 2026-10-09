from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / "snakemake" / "unified_real_data_distributions.smk"


def test_collision_data_rule_does_not_supply_testbeam_provenance_values():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert '"--source-domain-id", "3"' in text
    assert "--expected-source-domain-id 3" in text
    assert "--source-pdg-code" not in text
    assert "--source-energy-gev" not in text
    assert "--expected-source-pdg-code" not in text
    assert "--expected-source-energy-gev" not in text

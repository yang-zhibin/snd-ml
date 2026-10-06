#!/usr/bin/env python3

from __future__ import annotations

import copy

from abcd_ds_scifi_selection import configured_cutflow, configured_selection


def config() -> dict:
    return {
        "selection": {
            "base_expression": "count_scifi > 35",
            "scifi": {
                "active_xy_stations_equal": 5,
                "max_hits_per_station_exclusive": 20,
            },
            "ds": {
                "require_all_stations": True,
                "max_hits_per_station_inclusive": 2,
                "max_hits_total_inclusive": 5,
            },
        }
    }


def evaluate(expression: str, values: dict) -> bool:
    translated = expression.replace("&&", " and ").replace("||", " or ")
    translated = translated.replace("!", " not ")
    return bool(eval(translated, {"__builtins__": {}}, values))


def event(**updates) -> dict:
    values = {
        "scifi_n_active_stations_xy": 5,
        "count_scifi": 40,
        "count_ds": 4,
        "density_sndsw_scifi": 1000.0,
    }
    values.update({f"count_scifi{i}": 8 for i in range(1, 6)})
    values.update({f"count_ds{i}": 1 for i in range(1, 5)})
    values.update(updates)
    return values


def main() -> None:
    result = configured_selection(config())
    cutflow = configured_cutflow(config())
    assert [stage["key"] for stage in cutflow] == [
        "base", "scifi_xy", "scifi_station_hits", "ds_all_active",
        "ds_station_hits", "ds_total_hits",
    ]
    cumulative = cutflow[0]["expression"]
    for stage in cutflow[1:]:
        cumulative = f"({cumulative}) && ({stage['expression']})"
    tag = result["tag_expression"]
    assert evaluate(cumulative, event())
    assert evaluate(cumulative, event()) == evaluate(
        f"({result['base_expression']}) && ({tag})", event()
    )
    assert evaluate(tag, event())
    assert evaluate(tag, event(count_ds1=2, count_ds=5))
    assert not evaluate(tag, event(count_ds1=0, count_ds=3))
    assert not evaluate(tag, event(count_ds1=2, count_ds2=2, count_ds=6))
    assert not evaluate(tag, event(scifi_n_active_stations_xy=4))
    assert not evaluate(tag, event(count_scifi3=20))
    assert not evaluate(result["keep_expression"], event())
    assert evaluate(result["keep_expression"], event(count_ds4=0, count_ds=3))
    assert "count_veto" not in tag and "count_us" not in tag
    assert result["fingerprint"] == configured_selection(config())["fingerprint"]

    loose_config = copy.deepcopy(config())
    loose_config["selection"]["ds"] = {"require_all_stations": True}
    loose_selection = configured_selection(loose_config)
    loose_cutflow = configured_cutflow(loose_config)
    assert [stage["key"] for stage in loose_cutflow] == [
        "base", "scifi_xy", "scifi_station_hits", "ds_all_active",
    ]
    assert "count_ds <=" not in loose_selection["tag_expression"]
    assert all(
        f"count_ds{station} <=" not in loose_selection["tag_expression"]
        for station in range(1, 5)
    )
    assert evaluate(loose_selection["tag_expression"], event(count_ds1=8, count_ds=11))
    assert loose_selection["fingerprint"] != result["fingerprint"]
    print("DS--SciFi selection test: OK")


if __name__ == "__main__":
    main()

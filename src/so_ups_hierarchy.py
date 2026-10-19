#!/usr/bin/env python3
"""Hierarchy UES of Russia -> 7 IPS -> subjects of the RF (ATS codes), proposal.

Membership of subjects in IPS follows the pages of so-ups.ru (functioning/ees/oes-*), where each
IPS lists its regional power systems. Two power systems unite a federal city with a region:
Moscow city is inside "Moscow region" (code 46) and St Petersburg is inside "Leningrad region"
(code 41), as ATS publishes them. UTC offsets are the legal zone of the subject (general
knowledge, to be checked against the federal law on time; the data themselves are in Moscow time).

    python3 src/so_ups_hierarchy.py            # writes <root>/tables/hierarchy_ru.json
"""

import json
import os
from pathlib import Path

# code (ATS) -> (name, IPS, UTC offset of the main part of the subject)
R = {
    # IPS Center (18 power systems)
    14: ("Белгородская область", "CENTER", 3), 15: ("Брянская область", "CENTER", 3),
    17: ("Владимирская область", "CENTER", 3), 19: ("Вологодская область", "CENTER", 3),
    20: ("Воронежская область", "CENTER", 3), 24: ("Ивановская область", "CENTER", 3),
    29: ("Калужская область", "CENTER", 3), 34: ("Костромская область", "CENTER", 3),
    38: ("Курская область", "CENTER", 3), 42: ("Липецкая область", "CENTER", 3),
    46: ("Московская область (с г. Москвой)", "CENTER", 3), 54: ("Орловская область", "CENTER", 3),
    61: ("Рязанская область", "CENTER", 3), 66: ("Смоленская область", "CENTER", 3),
    68: ("Тамбовская область", "CENTER", 3), 28: ("Тверская область", "CENTER", 3),
    70: ("Тульская область", "CENTER", 3), 78: ("Ярославская область", "CENTER", 3),
    # IPS North-West (Kaliningrad power system is in the IPS, but ATS has no series for it)
    11: ("Архангельская область", "NORTHWEST", 3), 86: ("Республика Карелия", "NORTHWEST", 3),
    47: ("Мурманская область", "NORTHWEST", 3),
    41: ("Ленинградская область (с г. Санкт-Петербургом)", "NORTHWEST", 3),
    49: ("Новгородская область", "NORTHWEST", 3), 58: ("Псковская область", "NORTHWEST", 3),
    87: ("Республика Коми", "NORTHWEST", 3),
    # IPS Middle Volga (9)
    22: ("Нижегородская область", "VOLGA", 3), 36: ("Самарская область", "VOLGA", 4),
    63: ("Саратовская область", "VOLGA", 4), 56: ("Пензенская область", "VOLGA", 3),
    73: ("Ульяновская область", "VOLGA", 4), 89: ("Республика Мордовия", "VOLGA", 3),
    97: ("Чувашская Республика", "VOLGA", 3), 88: ("Республика Марий Эл", "VOLGA", 3),
    92: ("Республика Татарстан", "VOLGA", 3),
    # IPS South (17 power systems, 4 of them new: DPR, LPR, Zaporozhye, Kherson; no series)
    12: ("Астраханская область", "SOUTH", 4), 18: ("Волгоградская область", "SOUTH", 3),
    82: ("Республика Дагестан", "SOUTH", 3),
    3: ("Краснодарский край (с Республикой Адыгея)", "SOUTH", 3),
    85: ("Республика Калмыкия", "SOUTH", 3), 35: ("Республика Крым", "SOUTH", 3),
    67: ("Город Севастополь", "SOUTH", 3), 60: ("Ростовская область", "SOUTH", 3),
    26: ("Республика Ингушетия", "SOUTH", 3), 83: ("Кабардино-Балкарская Республика", "SOUTH", 3),
    91: ("Карачаево-Черкесская Республика", "SOUTH", 3),
    90: ("Республика Северная Осетия-Алания", "SOUTH", 3),
    96: ("Чеченская Республика", "SOUTH", 3), 7: ("Ставропольский край", "SOUTH", 3),
    # IPS Ural (9)
    80: ("Республика Башкортостан", "URAL", 5), 33: ("Кировская область", "URAL", 3),
    53: ("Оренбургская область", "URAL", 5), 57: ("Пермский край", "URAL", 5),
    94: ("Удмуртская Республика", "URAL", 4), 37: ("Курганская область", "URAL", 5),
    65: ("Свердловская область", "URAL", 5), 71: ("Тюменская область (с ХМАО и ЯНАО)", "URAL", 5),
    75: ("Челябинская область", "URAL", 5),
    # IPS Siberia (10 power systems)
    1: ("Алтайский край", "SIBERIA", 7), 84: ("Республика Алтай", "SIBERIA", 7),
    81: ("Республика Бурятия", "SIBERIA", 8), 76200: ("Забайкальский край", "SIBERIA", 9),
    25: ("Иркутская область", "SIBERIA", 8), 4: ("Красноярский край", "SIBERIA", 7),
    93: ("Республика Тыва", "SIBERIA", 7), 50: ("Новосибирская область", "SIBERIA", 7),
    52: ("Омская область", "SIBERIA", 6), 69: ("Томская область", "SIBERIA", 7),
    95: ("Республика Хакасия", "SIBERIA", 7), 32: ("Кемеровская область - Кузбасс", "SIBERIA", 7),
    # IPS East (second synchronous zone); ATS series exist only from 2025
    10: ("Амурская область", "EAST", 9), 5: ("Приморский край", "EAST", 10),
    8: ("Хабаровский край", "EAST", 10), 99: ("Еврейская автономная область", "EAST", 10),
    98: ("Республика Саха (Якутия), энергорайоны ОЭС Востока", "EAST", 9),
}
IPS = {"CENTER": "ОЭС Центра", "NORTHWEST": "ОЭС Северо-Запада", "VOLGA": "ОЭС Средней Волги",
       "SOUTH": "ОЭС Юга", "URAL": "ОЭС Урала", "SIBERIA": "ОЭС Сибири", "EAST": "ОЭС Востока"}


def build(with_residual=True):
    """Node list in the format of hierfm/hierarchy.py (top-down, level by level)."""
    nodes = [{"id": "UES", "parent": None, "attrs": {"name": "ЕЭС России"}}]
    for k, name in IPS.items():
        nodes.append({"id": f"IPS_{k}", "parent": "UES", "attrs": {"name": name}})
    order = list(IPS)
    for code, (name, ips, tz) in sorted(R.items(), key=lambda kv: (order.index(kv[1][1]), kv[0])):
        nodes.append({"id": f"R{code}", "parent": f"IPS_{ips}",
                      "attrs": {"name": name, "ats_code": code, "tz": tz}})
    if with_residual:
        for k in IPS:
            nodes.append({"id": f"RES_{k}", "parent": f"IPS_{k}",
                          "attrs": {"name": f"Остаток {IPS[k]}: ряд СО минус сумма субъектов", "tz": 3}})
    return {"levels": ["UES", "IPS", "region"], "nodes": nodes}


if __name__ == "__main__":
    root = Path(os.environ.get("SO_UPS_ROOT", Path(__file__).resolve().parents[1] / "data/raw/so_ups"))
    out = root / "tables" / "hierarchy_ru.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    h = build()
    out.write_text(json.dumps(h, ensure_ascii=False, indent=1))
    n_leaf = sum(1 for n in h["nodes"] if n["id"].startswith(("R", "RES_")))
    print(out, "nodes", len(h["nodes"]), "leaves", n_leaf)

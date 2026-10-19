"""Узлы иерархий и города для температурных рядов.

Модуль хранит только данные. Его читают build_weather_points.py и download_weather.py.

Узел это словарь: id, level, parent, name, ats (имя в списке АТС или None), cities.
Город это (запрос для геокодера, приближённая широта, приближённая долгота, код страны).
Геокодер Open-Meteo уточняет координаты и отдаёт население GeoNames.
Приближённые координаты нужны только для выбора города среди одноимённых.

Состав ОЭС взят со страницы so-ups.ru/functioning/ees/oes-center/ и из статьи
ru.wikipedia.org/wiki/Единая_энергетическая_система_России (проверено 2026-10-19).
Тыва и Республика Алтай отнесены к ОЭС Сибири по данным АТС и общим знаниям: на страницах СО их не нашли.
"""

# ---------------------------------------------------------------- Россия
# (id, имя АТС или русское имя, ats?, [города])
RU_OES = {
    "IPS_CENTER": ("ОЭС Центра", [
        ("Белгородская область", True, [("Belgorod", 50.60, 36.59), ("Stary Oskol", 51.30, 37.84)]),
        ("Брянская область", True, [("Bryansk", 53.25, 34.37)]),
        ("Владимирская область", True, [("Vladimir", 56.13, 40.41), ("Kovrov", 56.36, 41.32)]),
        ("Вологодская область", True, [("Vologda", 59.22, 39.89), ("Cherepovets", 59.13, 37.90)]),
        ("Воронежская область", True, [("Voronezh", 51.66, 39.20)]),
        ("Ивановская область", True, [("Ivanovo", 57.00, 40.97)]),
        ("Калужская область", True, [("Kaluga", 54.51, 36.26), ("Obninsk", 55.10, 36.61)]),
        ("Костромская область", True, [("Kostroma", 57.77, 40.93)]),
        ("Курская область", True, [("Kursk", 51.73, 36.19)]),
        ("Липецкая область", True, [("Lipetsk", 52.61, 39.60), ("Yelets", 52.62, 38.50)]),
        ("Московская область", True, [("Moscow", 55.75, 37.62), ("Balashikha", 55.80, 37.94),
                                      ("Podolsk", 55.43, 37.54)]),
        ("Орловская область", True, [("Oryol", 52.97, 36.07)]),
        ("Рязанская область", True, [("Ryazan", 54.63, 39.74)]),
        ("Смоленская область", True, [("Smolensk", 54.78, 32.04)]),
        ("Тамбовская область", True, [("Tambov", 52.72, 41.45)]),
        ("Тверская область", True, [("Tver", 56.86, 35.90)]),
        ("Тульская область", True, [("Tula", 54.19, 37.62)]),
        ("Ярославская область", True, [("Yaroslavl", 57.63, 39.87), ("Rybinsk", 58.05, 38.84)]),
    ]),
    "IPS_SOUTH": ("ОЭС Юга", [
        ("Астраханская область", True, [("Astrakhan", 46.35, 48.04)]),
        ("Волгоградская область", True, [("Volgograd", 48.71, 44.51), ("Volzhsky", 48.79, 44.78)]),
        ("Республика Дагестан", True, [("Makhachkala", 42.98, 47.50), ("Derbent", 42.06, 48.29)]),
        ("Республика Калмыкия", True, [("Elista", 46.31, 44.26)]),
        ("Карачаево-Черкесская Республика", True, [("Cherkessk", 44.22, 42.06)]),
        ("Кабардино-Балкарская Республика", True, [("Nalchik", 43.49, 43.62)]),
        ("Республика Крым", True, [("Simferopol", 44.95, 34.10), ("Kerch", 45.36, 36.47)]),
        ("Город Севастополь", True, [("Sevastopol", 44.62, 33.52)]),
        ("Краснодарский край", True, [("Krasnodar", 45.04, 38.98), ("Sochi", 43.60, 39.73),
                                       ("Novorossiysk", 44.72, 37.77)]),
        ("Ростовская область", True, [("Rostov-on-Don", 47.23, 39.72), ("Taganrog", 47.21, 38.94)]),
        ("Республика Северная Осетия-Алания", True, [("Vladikavkaz", 43.02, 44.68)]),
        ("Ставропольский край", True, [("Stavropol", 45.04, 41.97), ("Pyatigorsk", 44.04, 43.06)]),
        ("Чеченская Республика", True, [("Grozny", 43.32, 45.69)]),
        ("Республика Ингушетия", True, [("Nazran", 43.23, 44.77)]),
    ]),
    "IPS_NORTHWEST": ("ОЭС Северо-Запада", [
        ("Ленинградская область", True, [("Saint Petersburg", 59.94, 30.31), ("Gatchina", 59.57, 30.13)]),
        ("Мурманская область", True, [("Murmansk", 68.97, 33.09)]),
        ("Новгородская область", True, [("Velikiy Novgorod", 58.52, 31.27)]),
        ("Псковская область", True, [("Pskov", 57.82, 28.33)]),
        ("Республика Карелия", True, [("Petrozavodsk", 61.79, 34.39)]),
        # Нет в списках АТС (fact_region). Нужны для полного состава ОЭС.
        ("Архангельская область", False, [("Arkhangelsk", 64.54, 40.54), ("Severodvinsk", 64.56, 39.83)]),
        ("Республика Коми", False, [("Syktyvkar", 61.67, 50.81), ("Ukhta", 63.56, 53.68)]),
        ("Калининградская область", False, [("Kaliningrad", 54.71, 20.51)]),
    ]),
    "IPS_VOLGA": ("ОЭС Средней Волги", [
        ("Нижегородская область", True, [("Nizhniy Novgorod", 56.33, 44.00), ("Dzerzhinsk", 56.24, 43.46)]),
        ("Республика Марий Эл", True, [("Yoshkar-Ola", 56.63, 47.89)]),
        ("Республика Мордовия", True, [("Saransk", 54.18, 45.18)]),
        ("Пензенская область", True, [("Penza", 53.20, 45.00)]),
        ("Самарская область", True, [("Samara", 53.20, 50.15), ("Tolyatti", 53.51, 49.42)]),
        ("Саратовская область", True, [("Saratov", 51.54, 46.01), ("Engels", 51.48, 46.12)]),
        ("Республика Татарстан", True, [("Kazan", 55.79, 49.12), ("Naberezhnyye Chelny", 55.74, 52.41)]),
        ("Ульяновская область", True, [("Ulyanovsk", 54.32, 48.39)]),
        ("Чувашская Республика-Чувашия", True, [("Cheboksary", 56.13, 47.25)]),
    ]),
    "IPS_URAL": ("ОЭС Урала", [
        ("Республика Башкортостан", True, [("Ufa", 54.74, 55.97), ("Sterlitamak", 53.63, 55.95)]),
        ("Кировская область", True, [("Kirov", 58.60, 49.66)]),
        ("Курганская область", True, [("Kurgan", 55.44, 65.34)]),
        ("Оренбургская область", True, [("Orenburg", 51.77, 55.10), ("Orsk", 51.20, 58.57)]),
        ("Пермский край", True, [("Perm", 58.01, 56.25)]),
        ("Свердловская область", True, [("Yekaterinburg", 56.84, 60.60), ("Nizhny Tagil", 57.92, 59.96)]),
        ("Тюменская область", True, [("Tyumen", 57.15, 65.53), ("Surgut", 61.25, 73.40),
                                      ("Nizhnevartovsk", 60.93, 76.57)]),
        ("Удмуртская Республика", True, [("Izhevsk", 56.85, 53.20)]),
        ("Челябинская область", True, [("Chelyabinsk", 55.15, 61.43), ("Magnitogorsk", 53.42, 59.05)]),
    ]),
    "IPS_SIBERIA": ("ОЭС Сибири", [
        ("Алтайский край", True, [("Barnaul", 53.36, 83.76), ("Biysk", 52.54, 85.21)]),
        ("Республика Алтай", True, [("Gorno-Altaysk", 51.96, 85.96)]),
        ("Республика Бурятия", True, [("Ulan-Ude", 51.83, 107.61)]),
        ("Иркутская область", True, [("Irkutsk", 52.30, 104.30), ("Bratsk", 56.15, 101.63),
                                      ("Angarsk", 52.54, 103.89)]),
        ("Кемеровская область", True, [("Novokuznetsk", 53.76, 87.12), ("Kemerovo", 55.35, 86.09)]),
        ("Красноярский край", True, [("Krasnoyarsk", 56.01, 92.85), ("Achinsk", 56.27, 90.50)]),
        ("Новосибирская область", True, [("Novosibirsk", 55.04, 82.93)]),
        ("Омская область", True, [("Omsk", 54.99, 73.37)]),
        ("Томская область", True, [("Tomsk", 56.50, 84.97)]),
        ("Республика Тыва", True, [("Kyzyl", 51.72, 94.43)]),
        ("Республика Хакасия", True, [("Abakan", 53.72, 91.43)]),
        ("Забайкальский край", True, [("Chita", 52.03, 113.50)]),
    ]),
    "IPS_EAST": ("ОЭС Востока", [
        # Нет в списках АТС. На уровне ОЭС данные СО есть (код 540000), по субъектам нет.
        ("Амурская область", False, [("Blagoveshchensk", 50.28, 127.54)]),
        ("Приморский край", False, [("Vladivostok", 43.12, 131.89), ("Ussuriysk", 43.80, 131.95)]),
        ("Хабаровский край", False, [("Khabarovsk", 48.48, 135.08), ("Komsomolsk-on-Amur", 50.55, 137.00),
                                     ("Birobidzhan", 48.79, 132.92)]),
        ("Республика Саха (Южно-Якутский район)", False, [("Neryungri", 56.66, 124.72)]),
    ]),
}

TRANS = {
    "Белгородская": "belgorod", "Брянская": "bryansk", "Владимирская": "vladimir", "Вологодская": "vologda",
    "Воронежская": "voronezh", "Ивановская": "ivanovo", "Калужская": "kaluga", "Костромская": "kostroma",
    "Курская": "kursk", "Липецкая": "lipetsk", "Московская": "moscow", "Орловская": "oryol",
    "Рязанская": "ryazan", "Смоленская": "smolensk", "Тамбовская": "tambov", "Тверская": "tver",
    "Тульская": "tula", "Ярославская": "yaroslavl", "Астраханская": "astrakhan", "Волгоградская": "volgograd",
    "Республика Дагестан": "dagestan", "Республика Калмыкия": "kalmykia",
    "Карачаево-Черкесская Республика": "karachay", "Кабардино-Балкарская Республика": "kabardino",
    "Республика Крым": "crimea", "Город Севастополь": "sevastopol", "Краснодарский край": "krasnodar",
    "Ростовская": "rostov", "Республика Северная Осетия-Алания": "osetia", "Ставропольский край": "stavropol",
    "Чеченская Республика": "chechnya", "Республика Ингушетия": "ingushetia",
    "Ленинградская": "leningrad", "Мурманская": "murmansk", "Новгородская": "novgorod", "Псковская": "pskov",
    "Республика Карелия": "karelia", "Архангельская": "arkhangelsk", "Республика Коми": "komi",
    "Калининградская": "kaliningrad", "Нижегородская": "nizhny", "Республика Марий Эл": "mariel",
    "Республика Мордовия": "mordovia", "Пензенская": "penza", "Самарская": "samara", "Саратовская": "saratov",
    "Республика Татарстан": "tatarstan", "Ульяновская": "ulyanovsk", "Чувашская Республика-Чувашия": "chuvashia",
    "Республика Башкортостан": "bashkortostan", "Кировская": "kirov", "Курганская": "kurgan",
    "Оренбургская": "orenburg", "Пермский край": "perm", "Свердловская": "sverdlovsk", "Тюменская": "tyumen",
    "Удмуртская Республика": "udmurtia", "Челябинская": "chelyabinsk", "Алтайский край": "altaikrai",
    "Республика Алтай": "altairep", "Республика Бурятия": "buryatia", "Иркутская": "irkutsk",
    "Кемеровская": "kemerovo", "Красноярский край": "krasnoyarsk", "Новосибирская": "novosibirsk",
    "Омская": "omsk", "Томская": "tomsk", "Республика Тыва": "tyva", "Республика Хакасия": "khakassia",
    "Забайкальский край": "zabaikalsky", "Амурская": "amur", "Приморский край": "primorye",
    "Хабаровский край": "khabarovsk", "Республика Саха": "sakha_south",
}


def ru_slug(name):
    for key, val in TRANS.items():
        if name.startswith(key):
            return val
    raise KeyError(name)


def russia_nodes():
    """Список узлов России: UES -> IPS_* -> REG_*."""
    out = [dict(id="UES", level=0, parent="", name="ЕЭС России", ats=None, has_load=1, cities=[], hier="ru")]
    for oes, (oname, regs) in RU_OES.items():
        out.append(dict(id=oes, level=1, parent="UES", name=oname, ats=None, has_load=1, cities=[], hier="ru"))
        for rname, in_ats, cities in regs:
            out.append(dict(id="REG_" + ru_slug(rname), level=2, parent=oes, name=rname,
                            ats=rname if in_ats else None, has_load=int(in_ats), cities=cities, hier="ru"))
    return out


# ---------------------------------------------------------------- Европа
# Узел: (id, родитель, имя, [города]); города пусты у родителей, у которых есть дети.
# Код страны для геокодера хранится в EU_CC.
EU_CC = {
    "AT": "AT", "BE": "BE", "BG": "BG", "CH": "CH", "CZ": "CZ", "EE": "EE", "ES": "ES", "FI": "FI", "FR": "FR",
    "GR": "GR", "HR": "HR", "HU": "HU", "IE": "IE", "IT": "IT", "LT": "LT", "LU": "LU", "LV": "LV", "ME": "ME",
    "MK": "MK", "NL": "NL", "PL": "PL", "PT": "PT", "RO": "RO", "RS": "RS", "SI": "SI", "SK": "SK",
    "DE": "DE", "DK": "DK", "NO": "NO", "SE": "SE",
}

EU_LEAF = {
    # страны без зон: города
    "AT": [("Vienna", 48.21, 16.37), ("Graz", 47.07, 15.44)],
    "BE": [("Brussels", 50.85, 4.35), ("Antwerp", 51.22, 4.40)],
    "BG": [("Sofia", 42.70, 23.32), ("Plovdiv", 42.14, 24.75)],
    "CH": [("Zurich", 47.37, 8.54), ("Geneva", 46.20, 6.14)],
    "CZ": [("Prague", 50.08, 14.44), ("Brno", 49.20, 16.61)],
    "EE": [("Tallinn", 59.44, 24.75)],
    "ES": [("Madrid", 40.42, -3.70), ("Barcelona", 41.39, 2.16), ("Valencia", 39.47, -0.38)],
    "FI": [("Helsinki", 60.17, 24.94), ("Tampere", 61.50, 23.76)],
    "GR": [("Athens", 37.98, 23.73), ("Thessaloniki", 40.64, 22.94)],
    "HR": [("Zagreb", 45.81, 15.98)],
    "HU": [("Budapest", 47.50, 19.04)],
    "IE": [("Dublin", 53.35, -6.26)],
    "LT": [("Vilnius", 54.69, 25.28), ("Kaunas", 54.90, 23.89)],
    "LU": [("Luxembourg", 49.61, 6.13)],
    "LV": [("Riga", 56.95, 24.11)],
    "ME": [("Podgorica", 42.44, 19.26)],
    "MK": [("Skopje", 42.00, 21.43)],
    "NL": [("Amsterdam", 52.37, 4.90), ("Rotterdam", 51.92, 4.48)],
    "PL": [("Warsaw", 52.23, 21.01), ("Krakow", 50.06, 19.94), ("Lodz", 51.76, 19.46)],
    "PT": [("Lisbon", 38.72, -9.14), ("Porto", 41.15, -8.61)],
    "RO": [("Bucharest", 44.43, 26.10), ("Cluj-Napoca", 46.77, 23.59)],
    "RS": [("Belgrade", 44.79, 20.45), ("Novi Sad", 45.25, 19.84)],
    "SI": [("Ljubljana", 46.06, 14.51)],
    "SK": [("Bratislava", 48.15, 17.11), ("Kosice", 48.72, 21.26)],
}

# страны с зонами: (страна, [(id зоны, имя, [города])])
EU_ZONES = {
    "FR": [
        ("FR_ARA", "Auvergne-Rhône-Alpes", [("Lyon", 45.76, 4.84), ("Grenoble", 45.19, 5.72)]),
        ("FR_BFC", "Bourgogne-Franche-Comté", [("Dijon", 47.32, 5.04), ("Besancon", 47.24, 6.02)]),
        ("FR_BRE", "Bretagne", [("Rennes", 48.11, -1.68), ("Brest", 48.39, -4.49)]),
        ("FR_CVL", "Centre-Val de Loire", [("Orleans", 47.90, 1.91), ("Tours", 47.39, 0.69)]),
        ("FR_GES", "Grand Est", [("Strasbourg", 48.58, 7.75), ("Reims", 49.26, 4.03), ("Metz", 49.12, 6.18)]),
        ("FR_HDF", "Hauts-de-France", [("Lille", 50.63, 3.06), ("Amiens", 49.89, 2.30)]),
        ("FR_NOR", "Normandie", [("Rouen", 49.44, 1.10), ("Caen", 49.18, -0.37)]),
        ("FR_NAQ", "Nouvelle-Aquitaine", [("Bordeaux", 44.84, -0.58), ("Limoges", 45.83, 1.26)]),
        ("FR_OCC", "Occitanie", [("Toulouse", 43.60, 1.44), ("Montpellier", 43.61, 3.88)]),
        ("FR_PDL", "Pays de la Loire", [("Nantes", 47.22, -1.55), ("Angers", 47.47, -0.55)]),
        ("FR_PAC", "Provence-Alpes-Côte d'Azur", [("Marseille", 43.30, 5.37), ("Nice", 43.70, 7.27)]),
        ("FR_IDF", "Île-de-France", [("Paris", 48.86, 2.35)]),
    ],
    "DE": [  # четыре зоны ОПС, как в src/download_europe.py (SMARD)
        ("DE_50Hertz", "50Hertz", [("Berlin", 52.52, 13.40), ("Hamburg", 53.55, 9.99), ("Leipzig", 51.34, 12.37)]),
        ("DE_Amprion", "Amprion", [("Cologne", 50.94, 6.96), ("Dortmund", 51.51, 7.47), ("Dusseldorf", 51.23, 6.78)]),
        ("DE_TenneT", "TenneT", [("Munich", 48.14, 11.58), ("Hannover", 52.37, 9.74), ("Nuremberg", 49.45, 11.08)]),
        ("DE_TransnetBW", "TransnetBW", [("Stuttgart", 48.78, 9.18), ("Karlsruhe", 49.01, 8.40)]),
    ],
    "DK": [
        ("DK1", "DK1 (запад)", [("Aarhus", 56.16, 10.20), ("Aalborg", 57.05, 9.92), ("Odense", 55.40, 10.39)]),
        ("DK2", "DK2 (восток)", [("Copenhagen", 55.68, 12.57), ("Roskilde", 55.64, 12.08)]),
    ],
    "NO": [
        ("NO1", "NO1 Осло", [("Oslo", 59.91, 10.75), ("Drammen", 59.74, 10.20)]),
        ("NO2", "NO2 Кристиансанн", [("Kristiansand", 58.15, 8.00), ("Stavanger", 58.97, 5.73)]),
        ("NO3", "NO3 Тронхейм", [("Trondheim", 63.43, 10.39)]),
        ("NO4", "NO4 Тромсё", [("Tromso", 69.65, 18.96)]),
        ("NO5", "NO5 Берген", [("Bergen", 60.39, 5.32)]),
    ],
    "SE": [
        ("SE1", "SE1 Лулео", [("Lulea", 65.58, 22.15)]),
        ("SE2", "SE2 Сундсвалль", [("Sundsvall", 62.39, 17.31), ("Ostersund", 63.18, 14.64)]),
        ("SE3", "SE3 Стокгольм", [("Stockholm", 59.33, 18.07), ("Gothenburg", 57.71, 11.97)]),
        ("SE4", "SE4 Мальмё", [("Malmo", 55.61, 13.00)]),
    ],
    "IT": [  # зоны рынка Terna
        ("IT_NORD", "IT Север", [("Milan", 45.46, 9.19), ("Turin", 45.07, 7.69), ("Bologna", 44.49, 11.34)]),
        ("IT_CNOR", "IT Центр-Север", [("Florence", 43.77, 11.25), ("Ancona", 43.62, 13.52)]),
        ("IT_CSUD", "IT Центр-Юг", [("Rome", 41.90, 12.50), ("Naples", 40.85, 14.27)]),
        ("IT_SUD", "IT Юг", [("Bari", 41.12, 16.87)]),
        ("IT_SICI", "IT Сицилия", [("Palermo", 38.12, 13.36), ("Catania", 37.50, 15.09)]),
        ("IT_SARD", "IT Сардиния", [("Cagliari", 39.22, 9.12)]),
    ],
}


def europe_nodes():
    """Список узлов Европы: EU -> страна -> зона."""
    out = [dict(id="EU", level=0, parent="", name="Европа (сумма стран)", ats=None, has_load=1, cities=[], hier="eu")]
    for cc in sorted(set(EU_LEAF) | set(EU_ZONES)):
        out.append(dict(id=cc, level=1, parent="EU", name=cc, ats=None, has_load=1, cities=EU_LEAF.get(cc, []),
                        hier="eu", cc=EU_CC[cc]))
        for zid, zname, cities in EU_ZONES.get(cc, []):
            out.append(dict(id=zid, level=2, parent=cc, name=zname, ats=None, has_load=1, cities=cities,
                            hier="eu", cc=EU_CC[cc]))
    return out


def all_nodes():
    return russia_nodes() + europe_nodes()

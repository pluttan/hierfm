"""Work plan for the paper on probabilistic reconciliation of time series foundation model forecasts.

Built with deckkit (~/mnt/pr/pets/deckkit): nodes have names, edges join names,
every slide carries a `note` with its meaning, so this file reads without pictures.

    deck all docs/deck/slides.py      # build deck.html, check geometry, write deck.pdf
"""
from deckkit import M, W, Slide, row, tree

from explain import EXPLAIN

TITLE = "План работ по статье"
FOOTER = "papers · согласование прогнозов"


# ==========================
# ===  Goal              ===
# ==========================

def title():
    s = Slide(None, None, note="title slide; the small tree on the right hints at the grid hierarchy")
    s.eyebrow(M, 228, "статья · план работ · 19.10.2026", "accent")
    s.text(M, 300, "План работ", 76)
    s.para(M, 372, "Согласование вероятностных прогнозов предобученных моделей временных рядов "
                   "по иерархии энергосистемы при короткой истории", 720, 26, "muted")
    s.text(M, 640, "Все сроки в презентации даны оценкой.", 15, "soft")
    tree(s, "t", [880, 1000, 1120], 80, 36, [252, 324, 396, 468],
         lambda lv, i: "focal" if lv == 0 else "plain", lambda lv, i: "", "accent")
    return s


def goal():
    s = Slide("цель", "Что статья должна доказать",
              "Статья держится на двух утверждениях. Опыты проверяют их на двух энергосистемах.",
              note="two claims: accuracy with a short error history, and calibrated intervals; "
                   "both are checked on Russia (SO UPS) and Europe (ENTSO-E)")
    s.node("claim_short", M, 240, 560, 136, "focal", "Точность при короткой истории",
           "согласование работает на окне в 1–4 недели", size=24, sub_size=16)
    s.node("claim_calib", 656, 240, 560, 136, "focal", "Честные интервалы",
           "доля попаданий совпадает с заявленной", size=24, sub_size=16)
    s.node("check", M, 456, W - 2 * M, 96, "store", "Проверка: ЕЭС России и ENTSO-E",
           "одинаковый порядок опытов на обеих иерархиях", size=22, sub_size=16)
    s.edge("claim_short", "check")
    s.edge("claim_calib", "check")
    return s


def stages():
    s = Slide("этапы", "Семь этапов",
              "Этапы идут по порядку. Сиреневым выделены этапы с новым методом.",
              note="seven stages: data, base forecasts, baselines, method A (covariance), "
                   "method B (joint scenarios), experiments, paper text; about 17 weeks by estimate")
    titles = ["Данные", "Прогнозы", "Эталоны", "Ковариация", "Сценарии", "Опыты", "Статья"]
    caps = ["иерархии и погода", "модели без дообучения", "известные методы согласования",
            "метод A", "метод B", "таблицы и рисунки", "текст и подача"]
    weeks = ["2 нед.", "2 нед.", "1 нед.", "3 нед.", "3 нед.", "3 нед.", "3 нед."]
    row(s, ["data", "base", "ref", "cov", "scen", "exp", "paper"], M, 300, 144, 88, 24,
        lambda i: "focal" if i in (3, 4) else "plain", titles, size=18)
    for i, (cap, wk) in enumerate(zip(caps, weeks)):
        x = M + i * 168 + 72
        s.para(x, 420, cap, 136, 14, "muted", "middle")
        s.text(x, 268, wk, 14, "soft", "middle")
    s.text(M, 580, "Всего около 17 недель. Это оценка.", 20)
    return s


# ==========================
# ===  Stages            ===
# ==========================

def data():
    s = Slide("этап 1", "Данные",
              "Три источника дают две иерархии и температуру по каждому узлу.",
              note="sources: SO UPS hourly load (UPS, 7 IPS, regions), ENTSO-E (countries, bidding zones), "
                   "ERA5 temperature; output is two hierarchies with summing matrices")
    s.node("so", M, 232, 400, 88, "store", "Системный оператор", "ЕЭС, ОЭС, субъекты · по часам", sub_size=15)
    s.node("entsoe", M, 352, 400, 88, "store", "ENTSO-E", "страны и зоны торгов · по часам", sub_size=15)
    s.node("era5", M, 472, 400, 88, "store", "ERA5", "температура по узлам", sub_size=15)
    s.node("hier", 720, 336, 400, 120, "focal", "Две иерархии", "ряды и матрица суммирования", size=24)
    s.edge("so", "hier")
    s.edge("entsoe", "hier")
    s.edge("era5", "hier")
    s.para(M, 604, "Риск: часовых данных по субъектам может не хватить. Тогда нижний уровень займут ОЭС.",
           W - 2 * M, 16, "ink")
    return s


def forecasts():
    s = Slide("этапы 2–3", "Прогнозы и эталоны",
              "Модели дают сценарии по каждому узлу. Известные методы задают уровень для сравнения.",
              note="zero-shot TSFMs plus two classic baselines produce per-node scenarios and window errors; "
                   "five known reconciliation methods are the reference")
    s.zone("models", M, 240, 320, 344, "модели")
    for i, name in enumerate(["Chronos-2", "TimesFM-2.5", "Moirai-2", "Сезонный наивный", "Бустинг на погоде"]):
        s.node(f"model{i}", 88, 256 + i * 64, 272, 52, "plain" if i < 3 else "ghost", name, size=17)
    s.arrow([(392, 412), (448, 412)], "accent", width=2)
    s.node("scen", 456, 352, 264, 120, "store", "Сценарии по узлам", "и ошибки на окне", size=20)
    s.arrow([(728, 412), (784, 412)], "accent", width=2)
    s.zone("methods", 792, 240, 424, 344, "известные методы согласования")
    for i, name in enumerate(["Снизу вверх", "OLS и WLS", "MinT-shrink", "Робастный метод 2025", "ERM"]):
        s.node(f"method{i}", 816, 256 + i * 64, 376, 52, "plain", name, size=17)
    s.para(M, 628, "Контрольная точка: модели без дообучения не хуже бустинга на горизонте в сутки.",
           W - 2 * M, 16, "ink")
    return s


def method_cov():
    s = Slide("этап 4", "Метод A: ковариация при короткой истории",
              "Структура сети задаёт начальную ковариацию. Короткое окно ошибок её уточняет.",
              note="method A: prior covariance from grid structure (IPS membership, time zone, weather) "
                   "shrunk toward by short-window errors; shrink weight depends on window length")
    s.node("prior", M, 232, 320, 96, "store", "Структура сети", "ОЭС, часовой пояс, погода", sub_size=15)
    s.node("window", M, 368, 320, 96, "store", "Ошибки на окне", "от 1 до 4 недель", sub_size=15)
    s.node("shrink", 464, 296, 320, 104, "focal", "Сжатие к структуре", "вес зависит от длины окна", size=22)
    s.node("cov", 880, 296, 336, 104, "plain", "Ковариация ошибок", "для согласования", size=22)
    s.edge("prior", "shrink")
    s.edge("window", "shrink")
    s.edge("shrink", "cov")
    for i, task in enumerate(["1. Собрать признаки близости узлов.",
                              "2. Выбрать правило для веса сжатия.",
                              "3. Сравнить с MinT-shrink и робастным методом на окнах от недели до года."]):
        s.text(M, 524 + i * 32, task, 17)
    return s


def method_scen():
    s = Slide("этап 5", "Метод B: совместные сценарии",
              "Модель считает узлы отдельно. Метод восстанавливает связь узлов, затем согласует сценарии.",
              note="method B: per-node scenarios -> reorder to restore cross-node dependence (Schaake shuffle "
                   "by load history or temperature) -> reconcile scenarios with method A covariance -> intervals")
    row(s, ["raw", "reorder", "recon", "intervals"], M, 280, 264, 104, 32,
        lambda i: "focal" if i in (1, 2) else ("store" if i == 3 else "plain"),
        ["Сценарии по узлам", "Перестановка", "Согласование", "Интервалы"],
        ["каждый узел отдельно", "порядок из истории", "ковариация из метода A", "по всем уровням"], size=20)
    for i, task in enumerate(["1. Перестановка Шааке по истории нагрузки.",
                              "2. Перестановка по прогнозу температуры.",
                              "3. Сравнение с совместным режимом Chronos-2."]):
        s.text(M, 476 + i * 32, task, 17)
    s.text(M, 600, "Проверка: доля попаданий в интервалы 80 % и 95 % на каждом уровне.", 17, "muted")
    return s


def experiments():
    s = Slide("этап 6", "Опыты: шесть результатов",
              "Каждый узел даёт одну таблицу или один рисунок статьи.",
              note="six results: accuracy by level, window length, calibration, ablation, "
                   "transfer to ENTSO-E, significance tests (DM, MCB)")
    items = [("Точность по уровням", "таблица · MASE"), ("Длина окна", "рисунок · от недели до года"),
             ("Калибровка", "рисунок · покрытие 80 и 95 %"), ("Абляция", "таблица · без каждой части"),
             ("Перенос на ENTSO-E", "таблица · те же метрики"), ("Значимость", "тесты DM и MCB")]
    for i, (name, sub) in enumerate(items):
        s.node(f"res{i}", M + (i % 3) * 392, 240 + (i // 3) * 144, 368, 120,
               "focal" if i in (1, 2) else "plain", name, sub, size=21)
    s.para(M, 560, "Все прогоны идут на маке с nice и не больше четырёх задач сразу.", W - 2 * M, 16, "muted")
    return s


# ==========================
# ===  Control           ===
# ==========================

def gates():
    s = Slide("контроль", "Три контрольные точки",
              "Если ответ «нет», план меняется до перехода к следующему этапу.",
              note="gates: after stage 3 (TSFMs competitive?), after stage 4 (gain on short window?), "
                   "after stage 5 (coverage closer to nominal?); each 'no' has a fallback")
    gates_ = [("gate_ref", "После этапа 3", "модели не хуже бустинга?", "fix_ref", "Добавить погоду в модель"),
              ("gate_cov", "После этапа 4", "есть выигрыш на коротком окне?", "fix_cov", "Сделать упор на метод B"),
              ("gate_scen", "После этапа 5", "покрытие ближе к заявленному?", "fix_scen",
               "Ослабить заявку, журнал Q2")]
    for i, (gid, name, q, fid, fix) in enumerate(gates_):
        y = 232 + i * 128
        s.node(gid, M, y, 440, 92, "plain", name, q, size=20)
        s.node(fid, 736, y, 480, 92, "drop", fix, size=19)
        s.edge(gid, fid)
        s.text(604, y + 30, "нет", 14, "muted", "middle")
    s.edge("gate_ref", "gate_cov")
    s.edge("gate_cov", "gate_scen")
    s.text(296, 342, "да", 14, "muted", "middle")
    s.text(296, 470, "да", 14, "muted", "middle")
    return s


def human():
    s = Slide("от вас", "Что нужно от вас",
              "Четыре шага требуют вашего решения или вашей учётной записи.",
              note="human actions: ENTSO-E API token, Copernicus CDS key for ERA5, "
                   "journal choice after stage 6, agreement with the supervisor on the topic")
    items = [("Ключ ENTSO-E", "регистрация на transparency.entsoe.eu"),
             ("Ключ Copernicus CDS", "доступ к ERA5"),
             ("Выбор журнала", "IJF или Applied Energy, после этапа 6"),
             ("Согласование темы", "с научным руководителем")]
    for i, (name, sub) in enumerate(items):
        s.node(f"ask{i}", M + (i % 2) * 588, 248 + (i // 2) * 152, 564, 120, "load", name, sub, size=22)
    return s


SLIDES = [title, goal, stages, data, forecasts, method_cov, method_scen, experiments, gates, human]

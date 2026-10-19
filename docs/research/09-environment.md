# Модели и среда для кластера

Дата проверки: 2026-10-19. Ничего не устанавливалось и не запускалось. Источники читались по сети: исходники на GitHub, JSON-страницы PyPI, API Hugging Face. Все выводы о форме массивов взяты из кода. Запуск на GPU ещё не было.

## 1. Итог сверки обёрток

Файл: `hierfm/models/pretrained.py`. Копия старой версии лежала в `/tmp/t9/pretrained.orig.py` и может исчезнуть.

| Обёртка | Результат | Что сделано |
|---|---|---|
| `Chronos2` | Имена аргументов и порядок квантилей верны | Добавлены проверки формы (`assert`) и комментарий |
| `ChronosBolt` | Порядок и форма верны | Комментарий. Правок кода нет |
| `TimesFM25` | Три ошибки | Исправлены |
| `TimesFM3` | Верно | Комментарий. Правок кода нет |
| `Moirai2` | Путь через gluonts хрупкий | Заменён прямым вызовом модели |

### 1.1 Chronos-2

Источник: `src/chronos/chronos2/pipeline.py`, пакет chronos-forecasting 2.3.2.

- Подпись `predict_df(df, future_df, id_column, timestamp_column, target, prediction_length, quantile_levels, batch_size, context_length, cross_learning, validate_inputs, freq)`. Обёртка использует только имена, которые есть в подписи.
- Результат имеет длинный формат. Колонки: идентификатор, время, `target_name`, `predictions` и по одной колонке на каждый уровень. Имя колонки равно `str(уровень)`, например `"0.05"`. Обёртка читает именно такие имена.
- Список в `target` даёт один блок строк на каждую цель. Обёртка отбирает блок по `target_name`. Строки внутри блока идут по времени.
- Колонки `df`, которых нет в `future_df`, считаются прошлыми ковариатами. Колонки из `future_df` считаются известными в будущем. Температура в обёртке лежит в обоих датафреймах, значит передаётся как известная.
- Время в `future_df` должно точно совпадать с временем прогноза (проверка `validate_inputs=True`). Обёртка строит время из одного `date_range`, расхождения нет.
- Уровни `NATIVE` (21 значение от 0,01 до 0,99) равны `chronos_config.quantiles` из `config.json` модели `amazon/chronos-2`. Проверено по файлу.
- Контекст модели равен 8192 точкам. Конфигурации проекта дают 2048.

### 1.2 Chronos-Bolt

Источники: `chronos_bolt.py`, `base.py`.

- `predict_quantiles(inputs, quantile_levels, **kwargs)` возвращает пару `(quantiles, mean)`. Форма `quantiles` равна (пакет, горизонт, число уровней). Обёртка берёт первый элемент пары и не переставляет оси. Это верно.
- `prediction_length` уходит через `**kwargs` в `predict`. Это верно.
- `device_map` и `torch_dtype` проходят в transformers. Базовый класс сам заменяет `torch_dtype` на `dtype` для transformers 4.56 и выше.
- Модель предсказывает 64 шага за раз. Горизонт 168 строится в несколько шагов по собственной медиане. Хвостовые квантили такого прогноза менее надёжны. Это стоит назвать в статье, если Bolt попадёт в таблицу.

### 1.3 TimesFM 2.5 (исправлено три ошибки)

Источники: `src/timesfm/configs.py`, `timesfm_2p5_base.py`, `timesfm_2p5_torch.py`, пакет timesfm 3.0.2.

1. `ForecastConfig.per_core_batch_size` по умолчанию равен 1. Скомпилированный декодер берёт `per_core_batch_size * device_count` рядов за шаг. Старая обёртка давала по одному ряду за вызов. Теперь значение равно параметру `batch` (по умолчанию 256).
2. Пакет жёстко задаёт `cuda:0` и `device_count`, равный числу видимых карт. При запуске через `torchrun` все процессы сели бы на `cuda:0`, а пакеты дополнялись бы до числа карт. Теперь для устройств `cuda*` обёртка выставляет устройство процесса и `device_count = 1`.
3. `compile()` фиксирует `max_horizon`. Вызов с большим `h` после первого вызывал ошибку `Horizon must be less than the max horizon`. Теперь модель пересоздаётся, если `h` больше скомпилированного.

Верно и не менялось:

- `forecast(horizon, inputs)` возвращает точку формы (ряды, горизонт) и квантили формы (ряды, горизонт, 10).
- Колонка 0 квантилей равна среднему. Колонки 1 до 9 равны уровням 0,1 до 0,9. Срез `[:, :h, 1:]` верен.
- `max_context + max_horizon` не больше `context_limit` модели. Непрерывная голова квантилей работает при `max_horizon` не больше 1024.
- Параметр `torch_compile` (по умолчанию включён) вызывает `torch.compile(forward)`. На Python 3.13 это требует torch 2.6 или новее. Параметр `torch_compile: false` в конфигурации отключает компиляцию.

### 1.4 TimesFM 3.0

Источники: `src/timesfm3/torch/timesfm3_forecaster.py`, `evaluator.py`, `README.md`. Пакет timesfm 3.0.2 ставит оба верхних модуля: `timesfm` (версия 2.5) и `timesfm3`.

- `ModelConfig(checkpoint_path, per_core_batch_size, device)` и `TimesFM3Evaluator(config)` существуют. Имена в обёртке верны.
- `predict_batch(contexts, horizon, past_only_covariates, past_future_covariates, ts_ids, return_quantiles, use_symmetric_averaging, make_positive, sort_quantiles, use_znorm, padding_mode, univariate)` выдаёт `ForecastOutput(forecast, quantiles)`.
- Одномерный контекст даёт квантили (горизонт, 9). Двумерный контекст (переменные, длина) даёт (переменные, горизонт, 9). Уровни 0,1 до 0,9. Оба случая обёртка обрабатывает верно.
- Прошлая и будущая ковариата имеет длину `L + h`. Модель выводит горизонт из этой длины, поэтому `padding_mode="none"` подходит для h = 24 и h = 168.
- За один проход модель берёт не больше 32 переменных (`max_variates` в `config.json`). Если рядов и ковариат больше, оценщик режет цели на куски. Куски не видят друг друга.
- Контекст обрезается до последних 15360 точек.
- `make_positive=True` по умолчанию. Для нагрузки это безопасно.

Риск для статьи. Совместный режим `timesfm3_joint` с группой из всех узлов иерархии (около 90) превращается в куски по 31 ряду. Это не один совместный прогноз. Чтобы режим был честным, задайте `group_level`, при котором группа не больше 31 ряда (например, группа по ОЭС). Решение о схеме групп принимает автор опытов.

### 1.5 Moirai-2

Источники: `src/uni2ts/model/moirai2/forecast.py`, `module.py`, `config.json` модели. Пакет uni2ts 2.0.0 (2025-11-04) содержит `uni2ts/model/moirai2`. Проверено по списку файлов колеса.

Что изменено. Обёртка вызывала модель через `create_predictor` и `ListDataset` из gluonts. Теперь она вызывает модель напрямую, как в `src/smoke/smoke_moirai.py`. Причины:

- Предиктор тянет в цепочку версии gluonts и pandas. Для них нет проверки на Python 3.13.
- `QuantileForecastGenerator` получает уровни как числа, хотя аннотация требует строки. Код работает только за счёт приведения типов в pydantic внутри gluonts. Версия gluonts 0.17 не проверялась.

Форма вызова: `forward(past_target (b, L, 1), past_observed_target (b, L, 1) bool, past_is_pad (b, L) bool)` возвращает (b, число квантилей, h). Порядок уровней равен `module.quantile_levels`: от 0,1 до 0,9. Масштабирование внутри модуля, внешний масштаб не нужен. Обёртка переставляет оси в (b, h, уровни). Пропуски (NaN) маскируются и заменяются нулём.

Модель выдаёт 64 шага за раз (4 патча по 16). Для h = 168 код uni2ts идёт авторегрессией сам.

Лицензия весов не позволяет коммерческое использование (см. раздел 4).

### 1.6 Что не удалось проверить

- Ни одна обёртка не запускалась. Все выводы идут из чтения кода и конфигураций.
- Совместимость pydantic в gluonts 0.17 с uni2ts не нужна после замены, но импорт `uni2ts.model.moirai2` всё равно подтягивает gluonts. Цель `make install-moirai` проверяет импорт.
- Транзитивные импорты `uni2ts.module.*` на Python 3.13 не проверены. Первый запуск `make install-moirai` покажет, не хватает ли пакета.
- Действие `torch.compile` на A100 с Python 3.13 не проверено. При ошибке задайте `torch_compile: false` для `timesfm25`.

## 2. Требования к Python и torch

| Пакет | Версия на PyPI | `requires-python` | Ограничения |
|---|---|---|---|
| torch | 2.14.1 | 3.10 и новее | Для `torch.compile` на Python 3.13 нужен torch 2.6 и новее (заметки к выпуску 2.6, pytorch.org/blog/pytorch2-6) |
| chronos-forecasting | 2.3.2 | 3.10 и новее | torch от 2.2 до 3, transformers от 4.41 до 6, numpy от 1.21 до 3, pandas от 2.0 до 4, accelerate от 1.1 до 2 |
| timesfm | 3.0.2 | 3.10 и новее | numpy 1.26.4 и новее, huggingface_hub 0.28 и новее, safetensors; экстра `torch` требует torch 2.0 и новее |
| uni2ts | 2.0.0 | 3.10 и новее | Жёсткие пины: `torch<2.5`, `numpy~=1.26.0`, `gluonts~=0.14.3`, `scipy~=1.11.3`, `einops==0.7.*`, `datasets~=2.17.1`. Для Python 3.13 не подходят |
| gluonts | 0.17.0 | от 3.10 до 3.15 (не включая) | Нужен для импорта uni2ts |

Как это учтено в проекте:

- `requirements-models.txt` теперь требует `torch>=2.6` (было 2.5). Правка нужна из-за `torch.compile` на Python 3.13.
- `make install-moirai` ставит uni2ts с ключом `--no-deps` и добавляет зависимости из `requirements-moirai.txt` без пинов. Схема верна для Python 3.13, потому что numpy 1.26 не имеет сборки под 3.13.
- Версия CUDA колёс torch на PyPI не проверялась. Комментарий в `requirements-models.txt` про CUDA 12 нужно сверить. Выполните `nvidia-smi` на узле и сравните версию драйвера с версией CUDA установленного колеса.
- Python 3.13 на кластере. Если там только 3.12, схема остаётся рабочей, а пины uni2ts всё равно мешают из-за torch и numpy.

## 3. Идентификаторы на Hugging Face и размеры весов

Данные получены через API Hugging Face (`/api/models/<id>?blobs=true`) 2026-10-19. Размер равен размеру файла `model.safetensors`. Все репозитории открыты, доступ без токена.

| Модель | Идентификатор | Файл весов | Параметры (оценка) | Изменён |
|---|---|---|---|---|
| Chronos-2 | `amazon/chronos-2` | 477 930 472 байт (456 МиБ) | 119,5 млн (478 МБ при float32) | 2026-06-05 |
| Chronos-2 small | `autogluon/chronos-2-small` | 111 749 048 байт (107 МиБ) | 27,9 млн | 2025-12-03 |
| Chronos-Bolt small | `amazon/chronos-bolt-small` | 190 888 824 байт (182 МиБ) | 47,7 млн | 2025-11-21 |
| Chronos-Bolt base | `amazon/chronos-bolt-base` | 821 203 576 байт (783 МиБ) | 205,3 млн | 2025-11-21 |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | 925 181 104 байт (882 МиБ) | 231 млн по заголовку safetensors (в названии указано 200m) | 2025-10-02 |
| TimesFM 3.0 | `google/timesfm-3.0-pytorch` | 1 322 898 824 байт (1,23 ГиБ) | 330,7 млн | 2026-09-02 |
| Moirai-2.0 small | `Salesforce/moirai-2.0-R-small` | 45 557 824 байт (43 МиБ) | 11,4 млн (11 387 208) | 2026-01-29 |

Замечания:

- Столбец «Параметры» взят из заголовка safetensors (поле `safetensors.total` в API Hugging Face). Для TimesFM 2.5 число 231 млн больше номинальных 200 млн из названия.
- Все тяжёлые веса вместе занимают около 2,9 ГБ (Chronos-2, Bolt small, TimesFM 2.5, TimesFM 3.0, Moirai-2).
- Кэш лежит в `~/.cache/huggingface`. Задайте `HF_HOME` на быстрый диск кластера.
- Если вычислительные узлы без выхода в сеть, скачайте веса заранее на узле с сетью: `huggingface_hub.snapshot_download(repo_id, local_dir=...)`. В конфигурации укажите параметр `repo` с локальным путём. Все четыре пакета принимают локальный каталог. Затем задайте `HF_HUB_OFFLINE=1`.
- Для воспроизводимости зафиксируйте ревизию (`revision`) каждого репозитория. Полный хеш даёт `GET /api/models/<id>`, поле `sha`. Префиксы на 2026-10-19: chronos-2 `29ec3766`, chronos-bolt-small `772f3d25`, timesfm-2.5 `1d952420`, timesfm-3.0 `43046b85`, moirai-2.0-R-small `30f43ff0`.

## 4. Лицензии

| Компонент | Лицензия | Что это значит для статьи |
|---|---|---|
| Код chronos-forecasting, веса Chronos-2, Chronos-Bolt | Apache-2.0 | Ограничений нет |
| Код timesfm и веса TimesFM 2.5 | Apache-2.0 | Ограничений нет |
| Веса TimesFM 3.0 | `timesfm-non-commercial-license-v1.0` (карточка Hugging Face: `license: other`) | Только некоммерческое использование. Для научной статьи допустимо. Условие укажите в тексте |
| Код uni2ts | Apache-2.0 | Ограничений нет |
| Веса Moirai-2.0 | CC BY-NC 4.0 | Только некоммерческое использование, нужна атрибуция |

Источник по TimesFM 3.0: README репозитория google-research/timesfm. Текст самой лицензии не читался, условия понимайте по карточке и README.

## 5. Порядок проверки на кластере

Цель `make smoke-gpu` перебирает модели по одной. Падение одной модели не останавливает остальные.

1. Проверьте драйвер и Python: `nvidia-smi`, `python3 --version`.
2. Выполните `make install-models`, затем `make install-moirai`.
3. Скачайте веса заранее, если узлы без сети (раздел 3).
4. Запустите `make smoke-gpu GPU=0`.
5. Если сборка `timesfm25` упала на компиляции, добавьте `torch_compile: false` в параметры модели в `configs/smoke-gpu.yaml`.
6. Если `moirai2` пишет `skip`, откройте `runs-smoke-gpu-moirai2.log` и найдите недостающий пакет.
7. Отдельные дымовые скрипты лежат в `src/smoke/`. После правки `common.py` они принимают `--device cuda`.

Ожидаемая форма результата каждой обёртки: `levels` (число уровней) и `q` формы (ряды, горизонт, число уровней), уровни по возрастанию. Эту форму проверяют `assert` в `Chronos2` и функция `to_levels` в `base.py`.

## 6. Что осталось неясным

- Все модели не запускались. Реальные ошибки появятся только при первом запуске на кластере.
- Нет дымового скрипта для TimesFM 3.0. Обёртка `TimesFM3` проверена только по коду.
- Не проверена работа `Chronos2` в совместном режиме с группой около 90 рядов по памяти A100. Размер групп можно уменьшить параметром `group_level`.
- Для хвостовых уровней (0,025 и 0,975) у TimesFM и Moirai нет собственных значений. `to_levels` прижимает их к 0,1 и 0,9. Это уже отмечено в `docs/research/03-tsfm.md`.
- Версию CUDA колёс torch 2.14.1 на PyPI не проверял.

## Источники

- Исходники chronos-forecasting: https://github.com/amazon-science/chronos-forecasting
- Исходники timesfm и timesfm3: https://github.com/google-research/timesfm
- Исходники uni2ts: https://github.com/SalesforceAIResearch/uni2ts
- gluonts: https://github.com/awslabs/gluonts
- Страницы PyPI: https://pypi.org/project/chronos-forecasting/, https://pypi.org/project/timesfm/, https://pypi.org/project/uni2ts/, https://pypi.org/project/gluonts/, https://pypi.org/project/torch/
- Hugging Face: https://huggingface.co/amazon/chronos-2, https://huggingface.co/amazon/chronos-bolt-small, https://huggingface.co/google/timesfm-2.5-200m-pytorch, https://huggingface.co/google/timesfm-3.0-pytorch, https://huggingface.co/Salesforce/moirai-2.0-R-small
- Заметки к выпуску PyTorch 2.6: https://pytorch.org/blog/pytorch2-6/

## Проверка

Дата: 2026-10-19. Проверяющий читал API Hugging Face, JSON PyPI, `config.json` моделей и README google-research/timesfm.

- Ссылок на научные работы (статьи с авторами и годом) в файле нет. Ссылки ведут только на репозитории, страницы PyPI и Hugging Face. Все адреса из списка источников существуют. Выдуманных работ нет.
- Идентификаторы Hugging Face, размеры `model.safetensors`, даты изменения и префиксы ревизий совпали с API по всем семи репозиториям. Все репозитории открыты без токена.
- Исправлено число параметров: TimesFM 2.5 имеет 231 млн (было 200 млн), остальные значения уточнены по заголовку safetensors.
- Лицензии совпали: Apache-2.0 у Chronos-2, Chronos-2 small, Bolt, TimesFM 2.5; `other` (timesfm-non-commercial-license-v1.0) у TimesFM 3.0; cc-by-nc-4.0 у Moirai-2.0. README TimesFM называет вес 3.0 "non-commercial, non-production". Лицензия кода uni2ts не проверялась (не проверено).
- Версии и `requires-python` на PyPI совпали: torch 2.14.1, chronos-forecasting 2.3.2, timesfm 3.0.2, uni2ts 2.0.0, gluonts 0.17.0. Пины uni2ts подтверждены (`torch<2.5`, `numpy~=1.26.0`, `gluonts~=0.14.3`, `scipy~=1.11.3`, `einops==0.7.*`, `datasets~=2.17.1`).
- Config подтвердил: у Chronos-2 контекст 8192 и 21 уровень; у TimesFM 3.0 `max_variates` равно 32 и 9 уровней; у Moirai-2.0 `num_predict_token` равно 4, `patch_size` равно 16, 9 уровней.
- Не проверено: утверждения о коде обёрток и исходниках (формы массивов, поведение `torch.compile`), версия CUDA колёс torch, число 15360 для контекста TimesFM 3.0, число 64 шагов Chronos-Bolt. Запусков не было.
- Скачанные файлы: данных нет. `/tmp/t9` содержит исходники и два списка файлов колеса, около 400 КБ, веса отсутствуют, как заявлено. Каталог можно удалить. Файл `pretrained.orig.py` из этого каталога упомянут в разделе 1 и исчезнет вместе с ним.

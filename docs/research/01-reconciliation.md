# 01. Точечное иерархическое согласование: обзор литературы

Дата: 2026-10-19. Проект: согласование вероятностных прогнозов предобученных моделей по иерархии энергосистемы при короткой истории.

Метки проверки в тексте:

- [П] источник найден веб-поиском или в API arXiv в этой сессии, выходные данные сверены.
- [А] абстракт прочитан целиком в API arXiv или на странице статьи.
- [Т] прочитан фрагмент полного текста (HTML на arXiv).
- [Н] источник назван из памяти, выходные данные не сверены. Перед цитированием проверьте.

Файлы PDF лежат в `docs/papers/` (список в разделе 9).

---

## 1. Постановка и обозначения

Иерархия задана матрицей суммирования S размера n x m. Здесь n число всех узлов, m число нижних узлов. Вектор всех рядов равен y = S b, где b вектор нижнего уровня.

Для проекта: корень ЕЭС России, 7 ОЭС, субъекты РФ. Нижний уровень содержит примерно 80 субъектов. Число узлов n примерно 90. История 1–4 недели при часовом шаге даёт 168–672 точки, но число независимых окон ретроспективной проверки заметно меньше.

Базовые прогнозы y_hat (вектор n) несогласованы. Согласование задаётся матрицей P (m x n):

    y_tilde = S P y_hat

Условие несмещённости: P S = I. Тогда согласованный прогноз несмещён, если базовый прогноз несмещён.

Оптимальная P минимизирует след ковариации ошибок согласованного прогноза (MinT):

    P = (S' W^-1 S)^-1 S' W^-1

Здесь W ковариация ошибок базовых прогнозов на горизонте h. Вся статья упирается в оценку W.

## 2. Классические методы

### 2.1 Снизу вверх, сверху вниз, средний уровень

- Bottom-up. P = [0 | I]. Верхние уровни равны сумме нижних. Нет смещения, но шумные нижние ряды копят ошибку.
- Top-down. Прогноз корня делится по долям. Доли берут из истории (average historical proportions, proportions of historical averages) или из прогнозных долей. Согласованный прогноз смещён по построению.
- Middle-out. Комбинация: прогноз среднего уровня суммируют вверх и делят вниз.

Источники:

- Gross, Sohl. «Disaggregation methods to expedite product line forecasting». 1990, том 9, выпуск 3. Статья существует. Журнал (Journal of Forecasting) и страницы (233–254) источники дают противоречиво. (не проверено)
- Athanasopoulos, Ahmed, Hyndman. «Hierarchical forecasts for Australian domestic tourism». International Journal of Forecasting, 25(1), 146–166, 2009. [П, проверено в сессии проверки]

Для проекта: top-down по долям нагрузки субъектов близок к практике диспетчерского планирования. Он полезен как слабая база для сравнения.

### 2.2 Оптимальная комбинация, OLS и GLS

Hyndman, Ahmed, Athanasopoulos, Shang. «Optimal combination forecasts for hierarchical time series». Computational Statistics & Data Analysis, 55(9), 2579–2589, 2011. DOI 10.1016/j.csda.2011.03.006. [П]

Идея: регрессия базовых прогнозов на нижние ряды. Классический OLS эквивалентен W = I. Версия GLS требует ковариации ошибок согласования, которую нельзя оценить (Wickramasuriya и др. 2019, [П]). Поэтому практика перешла на MinT.

### 2.3 WLS: structural и variance scaling

- WLS structural (wls_struct). W = диагональ, где вес узла равен числу нижних рядов под ним. Не требует остатков. Работает при любой длине истории.
- WLS variance scaling (wls_var). W = diag(дисперсии остатков базовых прогнозов). Нужна оценка только n дисперсий.

Для проекта: WLS structural даёт априорную диагональ без данных. Он служит нижней границей по сложности для метода A. Вес субъекта в структурной схеме равен 1, вес ОЭС равен числу её субъектов.

### 2.4 MinT

Wickramasuriya, Athanasopoulos, Hyndman. «Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization». Journal of the American Statistical Association, 114(526), 804–819, 2019. DOI 10.1080/01621459.2018.1448825. [П]

Результаты: замкнутая формула для P, несмещённость при P S = I, минимум суммарной дисперсии ошибок. Авторы рассматривают пять оценок W (в таблице ols и wls_struct взяты из общего семейства, sam и shr предложены как оценки по данным):

| Оценка | W | Свойства |
|---|---|---|
| ols | I | не использует данные |
| wls_struct | diag по структуре | не использует остатки |
| wls_var | diag(s_i^2) | n параметров |
| mint_cov (sam) | выборочная ковариация остатков | n(n+1)/2 параметров, вырождена при T < n |
| mint_shrink (shr) | сжатие выборочной ковариации к диагонали | устойчива при малом T |

Авторы рекомендуют shrink как рабочий вариант [Н, по общей памяти; вывод Steinmeister и Pauly ниже [Т] его подтверждает].

Геометрия: Panagiotelis, Athanasopoulos, Gamakumara, Hyndman. «Forecast reconciliation: A geometric view with new insights on bias correction». International Journal of Forecasting, 37(1), 343–359, 2021. [П] Согласование есть проекция на согласованное подпространство. Проекция гарантированно улучшает прогноз для класса потерь на обобщённой метрике. MinT минимизирует ожидаемую потерю в этом классе. Там же показана коррекция смещения до согласования.

Свойства сравнений: Wickramasuriya. «Properties of point forecast reconciliation approaches». arXiv:2103.11129, 2021. [А] GLS и MinT совпадают. Вариант MinT-U (без условия несмещённости, близок к ERM) в среднем точнее MinT. Порядок по общему MSE: MinT-U, MinT, OLS, базовые.

## 3. Оценка ковариации W

### 3.1 Выборочная оценка и сжатие

Остатки: e_t = y_t - y_hat_t, где y_hat_t прогноз на горизонт h, сделанный в момент t (ретроспективная проверка или одношаговые остатки внутри выборки). Выборочная ковариация имеет ранг не выше T - 1. При T < n она вырождена, формула MinT не считается.

Сжатие по Шефер–Штриммер (Schäfer, Strimmer). «A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics». Statistical Applications in Genetics and Molecular Biology, 4(1), Article 32, 2005. DOI 10.2202/1544-6115.1175. [П]

    W_shr = lambda * W_target + (1 - lambda) * W_sam

Цель W_target: диагональ выборочных дисперсий (цель D в пакете corpcor). Интенсивность lambda считается по аналитической формуле Ledoit–Wolf. Оценка положительно определена при любом T [П, по описанию в поиске].

Развитие: пакет FoReco предлагает вариант `oasd` (oracle approximating shrinkage toward diagonal), ссылка в документации FoReco на Ando и Xiao (2023). Работа существует: Sakai Ando, Mingmei Xiao. «High-Dimensional Covariance Matrix Estimation: Shrinkage Toward a Diagonal Target». IMF Working Paper WP/23/257, декабрь 2023, https://www.imf.org/en/publications/wp/issues/2023/12/08/high-dimensional-covariance-matrix-estimation-shrinkage-toward-a-diagonal-target-542025. Это рабочая статья МВФ, не журнал. Метод расширяет OAS (Chen и др. 2009): сжатие идёт к диагонали выборочной матрицы, выборочные дисперсии не меняются. Результаты выведены в предположении нормальности. [П]

### 3.2 Неопределённость самой оценки W

- Møller, Nystrup, Hjorth, Madsen. «Optimal Forecast Reconciliation with Uncertainty Quantification». arXiv:2402.06480, 2024. [А] Оценка весов как параметров линейной модели. Вывод распределения Уишарта для ковариации. Сжатие через априорные распределения. Кейс: нагрузка в Швеции (прямая связь с проектом).
- Carrara, Azzimonti, Corani, Zambon. «Modeling uncertainty in the covariance matrix for probabilistic forecast reconciliation». arXiv:2506.19554, 2025. [А] Априорное распределение обратного Уишарта. Согласованное предсказательное распределение есть многомерное t, а не гауссово. Оценка W принимается неточной, интервалы MinT при этом слишком узкие. Это прямая основа для вывода: короткая история даёт недооценку дисперсии.
- Aikawa и др. (2025–2026), раздел 4.

### 3.3 Малая выборка: что известно (отдельный вопрос)

Факты:

1. Число параметров W равно n(n+1)/2. Для n = 110 это 6160 параметров при 24 наблюдениях (пример из Steinmeister, Pauly [Т]).
2. Выборочная ковариация при T < n вырождена. Рабочий выход: shrink к диагонали, wls_var, wls_struct, ols.
3. Steinmeister, Pauly. «Iterative Trace Minimization for the Reconciliation of Very Short Hierarchical Time Series». arXiv:2409.18550, 2024. [Т] Метод MinTit: иерархию делят на одноуровневые подиерархии, малые ковариации оценивают отдельно, затем итерируют до сходимости. Моделирование: T = 15, 30, 60; 5000 прогонов; иерархия из 15 рядов. MinTit лучше MinT для ARIMA и ETS, в среднем около 10% снижения RMSE против 9,1% у MinT. MinT лучше при сильной сезонности и для GPR. При разной длине рядов MinTit_l даёт больше 20% снижения. Ограничение: моделирование небольшое (15 рядов), энергетики нет.
4. Aikawa и др. (раздел 4): оценка W из конечной выборки ухудшает результат. Предложена устойчивая постановка с интервалом неопределённости для M = W^-1 [Т].
5. Møller и др. и Carrara и др.: неопределённость оценки W занижает ширину интервалов.
6. Вывод Wickramasuriya и др. [Н]: shrink стабилен, но интенсивность lambda считается по той же короткой выборке и сама шумная.

Что делают в этом случае (сводка):

| Приём | Идея | Источник |
|---|---|---|
| Сжатие к диагонали | W_shr | MinT 2019 [П] |
| Структурные веса | W = diag(структура) | Hyndman 2011 [П] |
| Разбиение на подиерархии | малые ковариации | MinTit [Т] |
| Устойчивая оптимизация | интервал для M | Aikawa [Т] |
| Байесовские априорные | Уишарт | Møller [А], Carrara [А] |
| ERM / регуляризация | нет оценки W | Ben Taieb, Koo 2019 [П] |
| Выбор метода кросс-валидацией | при достаточной длине | Steinmeister [Т] |

Пробел для статьи: нет работы, где априорная ковариация строится из физической структуры сети (ОЭС, часовой пояс, температура), а вес сжатия задан как функция длины окна. Предложенный метод A занимает эту нишу. Поиск в API arXiv по ключевым словам не нашёл такой работы, но поиск был неполным. Нужна проверка в Scopus и Google Scholar перед заявкой на новизну.

## 4. Методы без оценки W и другие подходы

### 4.1 ERM

Ben Taieb, Koo. «Regularized regression for hierarchical forecasting without unbiasedness conditions». KDD 2019, 1337–1347. DOI 10.1145/3292500.3330976. [П] Версии на arXiv не найдено; PDF лежит в репозитории Университета Монса (orbi.umons.ac.be, файл 2019_kdd_hts_reg.pdf). [П]

Идея: отказ от условия несмещённости P S = I. Матрица P ищется минимизацией эмпирического риска на ретроспективных прогнозах с регуляризацией. Метод хорош при смещённых базовых прогнозах. Это важно для предобученных моделей без дообучения: их смещение на энергосистеме России нельзя исключить.

Риск при малом T: ERM оценивает матрицу m x n, это больше параметров, чем W при малом n. Регуляризация (lasso) обязательна. В hierarchicalforecast вариант ERM имеет режимы `closed`, `reg`, `reg_bu` [П, исходный код].

### 4.2 Game-theoretic

van Erven, Cugliari. «Game-theoretically optimal reconciliation of contemporaneous hierarchical time series forecasts». Глава в книге «Modeling and Stochastic Learning for Forecasting in High Dimensions», Springer, 2015. DOI 10.1007/978-3-319-18732-7_15. [П] Метод GTOP гарантирует, что согласование не ухудшает любой набор прогнозов (по CRAN-описанию пакета gtop). Не требует несмещённости. Полного текста главы не читали. Пакет R `gtop`, версия 0.2.0. [П]

### 4.3 Устойчивое согласование (2025–2026)

- Aikawa, Suzuki, Yoshitake, Teshigawara, Iwabuchi, Kobayashi, Nakata. «Hierarchical Time Series Forecasting with Robust Reconciliation». arXiv:2510.20383 (v2, февраль 2026); Transactions on Machine Learning Research, 2026. [П] Для TMLR подтверждено поиском (страница mlanthology). [Т]
  - Допустимое множество для M = W^-1: покомпонентные границы M_низ, M_верх, M положительно полуопределена.
  - Границы получены бутстрэпом (5000 повторов), перцентили (1-a)/2 и (1+a)/2, уровень a выбирается по валидации.
  - Задача min-max переходит в полуопределённую оптимизацию (решатель MOSEK).
  - Данные: рождаемость Австралии (n=10), туризм Австралии (n=35), продажи Walmart (n=14), электропотребление Швейцарии (n=31, 353 дня наблюдений, 19 дней прогноза). Базовые прогнозы Prophet.
  - Результат по Швейцарии (относительный MAE, база = 1): на уровнях сети, агрегата и подагрегата лучший метод устойчивый (0,888; 0,808; 0,867). MinT на уровне сети 1,110. На уровне счётчиков устойчивый метод хуже базы (1,036), лучший GLS (0,987). На туризме MinT лучше на уровнях штата и зоны.
  - Важно для статьи: метод тестирован на энергетике, но на малом числе узлов (до 35), с Prophet, только точечные прогнозы. Бутстрэп T точек при T порядка недели малополезен: нужна проверка.
- Wang, Wang, Cao, Yang. «Robust optimal reconciliation for hierarchical time series forecasting with M-estimation». arXiv:2602.22694, 2026. [А] M-оценка в самой задаче согласования, защита от выбросов в базовых прогнозах. Данные: туризм Австралии. Это иная задача (выбросы), чем малая выборка.

### 4.4 Машинное обучение и новые постановки (2020–2026)

- Spiliotis, Abolghasemi, Hyndman, Petropoulos, Assimakopoulos. «Hierarchical forecast reconciliation with machine learning». arXiv:2006.02043, 2020 (Applied Soft Computing, 2021 [Н]). [А] Нелинейная комбинация базовых прогнозов.
- Nguyen, Vahid, Wickramasuriya. «Hierarchical forecasting: The role of information». arXiv:2605.29611, 2026. [А] Метод IComb: улучшение связано не только с ограничениями, но и с объединением информации разных прогнозов.
- Hollyman и др. «Scaleable Dynamic Forecast Reconciliation». arXiv:2409.12856, 2024. [П, только название и автор по API]
- Wang, Johnson, Klee, Malloy. «Billions-Scale Forecast Reconciliation». arXiv:2602.05030, 2026. [А] Оптимизационное согласование на масштабе более 4 млрд значений. К нашей задаче (n около 90) отношения не имеет.
- Girolimetto и др. «Forecast reconciliation with non-linear constraints». arXiv:2510.21249, 2025. [П, только название по API]
- Zhang и др. «Discrete forecast reconciliation». arXiv:2305.18809, 2023. [П, только название по API]

Обзор: Athanasopoulos, Hyndman, Kourentzes, Panagiotelis. «Forecast reconciliation: A review». International Journal of Forecasting, 40(2), 430–456, 2024. DOI 10.1016/j.ijforecast.2023.10.010. [П] Открытая версия: рабочая статья Монаш (bridges.monash.edu, отчёт 28756238). Полный текст в этой сессии не читали, поэтому тематические выводы обзора здесь не пересказываются. Прочитайте его целиком перед написанием раздела «Обзор литературы».

## 5. Cross-temporal согласование

- Athanasopoulos, Hyndman, Kourentzes, Petropoulos. «Forecasting with temporal hierarchies». European Journal of Operational Research, 262(1), 60–74, 2017. DOI 10.1016/j.ejor.2017.02.046. [П] Агрегация ряда по времени, пакет R `thief`.
- Di Fonzo, Girolimetto. «Cross-temporal forecast reconciliation: Optimal combination method and heuristic alternatives». arXiv:2006.08570, 2020 (International Journal of Forecasting, 2023 [Н]). [П]
- Girolimetto, Athanasopoulos, Di Fonzo, Hyndman. «Cross-temporal probabilistic forecast reconciliation: Methodological and practical issues». arXiv:2303.17277; International Journal of Forecasting, 40(3), 1134–1151, 2024. [П] Содержит четыре варианта ковариации для высокой размерности и многошаговые остатки. Это прямой источник приёмов для малой выборки: многошаговые остатки и перекрывающиеся остатки увеличивают число точек для оценки W.
- Girolimetto, Di Fonzo. «Insights into regression-based cross-temporal forecast reconciliation». arXiv:2410.19407, 2024. [А] Кейс: почасовая фотоэлектрическая генерация.
- Girolimetto, Di Fonzo. «Point and probabilistic forecast reconciliation for general linearly constrained multiple time series». arXiv:2305.05330, 2023. [А]

Для проекта: временная иерархия (час, 3 часа, сутки) умножает число остатков для оценки W в окне 1–4 недели. Это дешёвый способ увеличить T. Идея требует проверки в опытах: остатки разных масштабов агрегации зависимы.

## 6. Вероятностное согласование (граница с методами A и B)

Тема другого агента в этом проекте. Здесь перечислены только работы, которые нужны для связки с точечным согласованием.

- Jeon, Panagiotelis, Petropoulos. «Reconciliation of probabilistic forecasts with an application to wind power». arXiv:1808.02635; European Journal of Operational Research, 279, 364–379, 2019. [А] Данные: ветроэнергетика, температурная иерархия.
- Panagiotelis, Gamakumara, Athanasopoulos, Hyndman. «Probabilistic forecast reconciliation: Properties, evaluation and score optimisation». European Journal of Operational Research, 306(2), 693–706, 2023. DOI 10.1016/j.ejor.2022.07.040. [П]
- Wickramasuriya. «Probabilistic forecast reconciliation under the Gaussian framework». arXiv:2103.11128, 2021. [А] Если несогласованное распределение гауссово, MinT минимизирует логарифмическую оценку; лог-оценка MinT по каждому маргиналу лучше, чем у OLS. Основа для метода «согласование выборок ковариацией A».
- Corani, Azzimonti, Augusto, Zaffalon. «Probabilistic reconciliation of hierarchical forecast via Bayes' rule». ECML PKDD 2020, 211–226. Версия на arXiv:1906.03105 названа «Reconciling Hierarchical Forecasts via Bayes' Rule». [П]
- Ben Taieb, Taylor, Hyndman. «Coherent Probabilistic Forecasts for Hierarchical Time Series». ICML 2017, PMLR 70, 3348–3357. https://proceedings.mlr.press/v70/taieb17a.html [П]
- Rangapuram и др. «End-to-End Learning of Coherent Probabilistic Forecasts for Hierarchical Time Series». ICML 2021, PMLR 139, 8832–8843. [П] На arXiv версии не найдено.
- Olivares и др. «Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures». arXiv:2110.13179; International Journal of Forecasting, 2023. [А]

## 7. Энергетика и электропотребление

| Работа | Что сделано | Метка |
|---|---|---|
| Ben Taieb, Taylor, Hyndman. «Hierarchical Probabilistic Forecasting of Electricity Demand With Smart Meter Data». JASA, 116(533), 27–43, 2021. DOI 10.1080/01621459.2020.1736081 | Иерархия счётчиков Великобритании; когерентные вероятностные прогнозы; достаточно моделировать малый набор зависимостей | [П] |
| Ben Taieb, Yu, Barreto, Rajagopal. «Regularization in Hierarchical Time Series Forecasting with Application to Electricity Smart Meter Data». AAAI 2017 (том 31), https://ojs.aaai.org/index.php/AAAI/article/view/11167 | Регуляризация согласования на данных счётчиков | [П] |
| Jeon, Panagiotelis, Petropoulos 2019 | Ветроэнергетика, Крит | [А] |
| Møller и др. 2024 | Нагрузка Швеции, стандартные ошибки весов | [А] |
| Girolimetto, Di Fonzo 2024 | Почасовая фотоэлектрическая генерация | [А] |
| Leprince, Madsen, Møller, Zeiler. «Hierarchical learning, forecasting coherent spatio-temporal individual and aggregated building loads». arXiv:2301.12967, 2023 | Здания, пространственные и временные иерархии, функция потерь со структурой согласования | [А] |
| Zheng и др. «Coherent Hierarchical Probabilistic Forecasting of Electric Vehicle Charging Demand». arXiv:2411.00337, 2024 | Станции зарядки; сценарии согласуются слоем выпуклой оптимизации | [А] |
| Aikawa и др. 2025 | Электропотребление Швейцарии | [Т] |
| Ghelasi и др. «Hierarchical forecasting for aggregated curves with an application to day-ahead electricity price auctions». arXiv:2305.16255, 2023 | Кривые заявок на рынке на сутки вперёд | [П; авторы Ghelasi, Ziel] |

Пробелы:

- Нет найденной работы по иерархии ЕЭС России с делением по ОЭС. Работы про Россию в списке не искали отдельно. Нужен поиск по eLibrary и Google Scholar.
- Нет найденной работы по предобученным моделям (Chronos, TimesFM, Moirai) с согласованием. Запросы про «foundation models» и «reconciliation» в API arXiv пустые. Это возможная ниша, но поиск был узким. Проверьте.
- Нет найденной работы по сетевым иерархиям с физическим (электрическим) приором для W. Это ниша метода A.

## 8. Библиотеки

### 8.1 Nixtla hierarchicalforecast

- Версия в PyPI на 2026-10-19: 1.5.3, загружена 2026-09-24, Python от 3.10, лицензия Apache 2.0. [П, JSON PyPI]
- Установка: `pip install hierarchicalforecast`. На vpnvm не ставить, ставить на маке.
- Классы согласования: `BottomUp`, `TopDown`, `MiddleOut`, `MinTrace`, `ERM`, `OptimalCombination` (устарел, заменён MinTrace), а также разреженные `BottomUpSparse`, `TopDownSparse`, `MiddleOutSparse`, `MinTraceSparse`. [П, исходный код methods.py]
- MinTrace: методы `ols`, `wls_struct`, `wls_var`, `mint_cov`, `mint_shrink`, `emint`. Для shrink используется оценка Шефер–Штриммер, есть параметр `mint_shr_ridge` (по умолчанию 2e-8). Есть опция `nonnegative` (QP через Clarabel). Минимального числа остатков в коде не требуется. [П]
- ERM: `closed`, `reg`, `reg_bu`, `lambda_reg` = 1e-2 по умолчанию. [П]
- Вероятностные выборки: `Normality`, `Bootstrap`, `PERMBU`, `Conformal`. [П] PERMBU это перестановочный подход к выборкам снизу вверх, идея близка к методу B, но не Шааке по температуре. Нужно читать код отдельно.
- Временная иерархия поддержана. Кросс-временное согласование в README не найдено. [П]
- Оценка: `hierarchicalforecast.evaluation.evaluate`, метрики из `utilsforecast.losses`.
- Для статьи библиотека подходит как воспроизводимая базовая линия. Матрицу W с априорным сжатием по структуре сети придётся писать самим: `MinTrace` принимает только готовые методы.

### 8.2 FoReco (R)

- Версия на CRAN: 1.3.1, опубликована 2026-07-09. [П]
- Функции: cross-sectional (`csbu`, `csmo`, `cstd`, `cslcc`, `csrec`), temporal (`te*`), cross-temporal (`ct*`, `tcsrec`, `cstrec`, `iterec`), вероятностные (`cssmp`, `csmvn`, `tesmp`, `temvn`, `ctsmp`, `ctmvn`), ковариации (`cscov`, `tecov`, `ctcov`, `shrink_estim`, `shrink_oasd`). [П]
- Варианты `comb` в `cscov`: `ols`, `str` (структурные), `wls`, `shr` (сжатие, Wickramasuriya 2019), `oasd` (Ando, Xiao 2023), `sam`. [П]
- Самая полная реализация кросс-временного и вероятностного согласования. R нужен на маке (в этой сессии наличие R на маке не проверяли).

### 8.3 scikit-hts

- Версия 0.5.12 в PyPI, загружена 2021-09-02, это последняя версия. Статус в описании: alpha. Поле requires_python в PyPI пустое; поддержка Python 3.6–3.8 из описания не подтверждена. [П]
- Проект не обновляется более пяти лет. Состав методов в этой сессии не удалось проверить (страница PyPI и README на GitHub недоступны). По памяти: bottom-up, средние исторические доли, OLS, WLS. [Н]
- Рекомендация: не использовать как основу, только упомянуть в обзоре.

### 8.4 Прочее

- Пакет R `gtop` для game-theoretic метода, версия 0.2.0. [П]
- Пакет R `hts`: MinT, пакет `thief` для временных иерархий. [П]
- Пакет R `ProbReco` для оптимизации оценок вероятностного согласования. [П, упомянут в выдаче поиска]
- Для устойчивого метода Aikawa и др. нужен MOSEK. Лицензия платная, есть академическая. Кода авторов не искали.

## 9. Скачанные PDF

Каталог: `/Volumes/pr/bmstu/papers/soglasovanie-prognozov/docs/papers/`. Загрузка выполнена на маке, во всех файлах первые байты `%PDF`. Мак недоступен, поэтому эти файлы при проверке не открыты. Каталог /data/hierfm/docs/papers на pcomp пуст. (не проверено)

| Файл | arXiv | Тема |
|---|---|---|
| 2025_aikawa_robust.pdf | 2510.20383 | устойчивое согласование (TMLR 2026) |
| 2026_wang_mestimation.pdf | 2602.22694 | M-оценка в согласовании |
| 2024_steinmeister_mintit.pdf | 2409.18550 | MinT для очень коротких рядов |
| 2025_carrara_covunc.pdf | 2506.19554 | неопределённость ковариации, t-распределение |
| 2024_moller_uncertainty.pdf | 2402.06480 | неопределённость весов, нагрузка Швеции |
| 2021_wickramasuriya_properties.pdf | 2103.11129 | свойства MinT, OLS, ERM |
| 2021_wickramasuriya_gaussian.pdf | 2103.11128 | вероятностное согласование, гауссов случай |
| 2019_jeon_probwind.pdf | 1808.02635 | вероятностное согласование, ветер |
| 2024_girolimetto_crosstemporal.pdf | 2303.17277 | кросс-временное вероятностное |
| 2023_girolimetto_linear.pdf | 2305.05330 | общие линейные ограничения |
| 2021_difonzo_crosstemporal.pdf | 2006.08570 | кросс-временная оптимальная комбинация |
| 2019_corani_bayes.pdf | 1906.03105 | байесовское согласование |
| 2026_nguyen_information.pdf | 2605.29611 | роль информации |
| 2023_leprince_buildingloads.pdf | 2301.12967 | нагрузки зданий |
| 2024_zheng_evcharging.pdf | 2411.00337 | зарядка электромобилей |

Итого 15 файлов. Не скачано, потому что нет открытой версии на arXiv: MinT 2019 (JASA), обзор 2024 (IJF), Ben Taieb и Koo 2019 (есть в репозитории Монса, не скачивали по лимиту), Ben Taieb и др. 2021 (JASA, есть в репозитории Монса), Hyndman 2011, Panagiotelis 2021 (IJF). Рабочие версии Монаш (bridges.monash.edu) можно скачать вручную.

## 10. Выводы для статьи

1. Базовый набор сравнения для опытов: bottom-up, top-down (доли), OLS, WLS struct, WLS var, MinT shrink, MinT sample (там, где T хватает), ERM reg, MinTit (по описанию, реализацию писать самим), устойчивый метод Aikawa (если код недоступен, воспроизводить через SDP).
2. Метод A (априорная ковариация из структуры сети и сжатие к ней по короткому окну) отличается от Schäfer–Strimmer целью сжатия: диагональ заменена структурной матрицей. Близкие работы: Møller и др. (априор Уишарта), Carrara и др. (то же для вероятностного случая), Steinmeister и Pauly (разбиение на подиерархии). Отличие нужно сформулировать явно.
3. Вес сжатия как функция длины окна: Schäfer–Strimmer уже даёт lambda по данным. Нужно показать, что явный закон lambda(T) лучше аналитического lambda на коротких окнах. Это проверяемая гипотеза.
4. Метод B и температурные копулы: стыкуется с PERMBU, Jeon и др. и Wickramasuriya 2021. Детали в файлах других агентов.
5. Данные Aikawa и др. по Швейцарии показывают: на уровне отдельных счётчиков согласование может вредить. В нашей иерархии нижний уровень это субъекты РФ с гладкой нагрузкой, риск меньше, но проверить нужно.

Риски новизны:

- Работы Møller и др. 2024 и Carrara и др. 2025 близки к идее «учёт неопределённости W». Нужна чёткая граница: наша априорная ковариация строится из физической структуры, а не из распределения Уишарта по данным.
- Устойчивое согласование уже опубликовано в TMLR 2026. Требуется сравнение с ним.
- Новизна «предобученные модели + согласование» не подтверждена и не опровергнута. Нужен более глубокий поиск.

## 11. Что проверить до опытов

- Прочитать обзор Athanasopoulos и др. 2024 целиком. Выписать раздел про малую выборку и ковариацию.
- Прочитать Ben Taieb и Koo 2019 (PDF в репозитории Монса) и Ben Taieb и др. 2021. Проверить, как они оценивают W на данных счётчиков.
- Проверить журнал и страницы Gross и Sohl 1990 по странице издателя.
- Проверить, есть ли код у Aikawa и др. и Steinmeister и Pauly (в этой сессии репозитории не искали).
- Определить, как получать остатки при длине окна 1–4 недели: число скользящих начал прогноза, горизонт, шаг.
- Оценить на реальных данных ЕЭС, насколько структурная ковариация (ОЭС, часовой пояс) объясняет выборочную корреляцию ошибок.

## 12. Проверка

Дата проверки: 2026-10-19. Проверяющий сверил ссылки веб-поиском и API arXiv.

Подтверждены: название, авторы, год, место публикации.

- arXiv (все 23 идентификатора из текста): 2103.11129, 2402.06480, 2506.19554, 2409.18550, 2510.20383, 2602.22694, 2006.02043, 2605.29611, 2409.12856, 2602.05030, 2510.21249, 2305.18809, 2006.08570, 2303.17277, 2410.19407, 2305.05330, 1808.02635, 2103.11128, 1906.03105, 2110.13179, 2301.12967, 2411.00337, 2305.16255.
- Журналы и труды: Hyndman 2011 (CSDA 55(9)), Wickramasuriya 2019 (JASA 114), Panagiotelis 2021 (IJF 37(1)), Athanasopoulos 2024 (IJF 40(2)), Schäfer и Strimmer 2005, Ben Taieb и Koo 2019 (KDD, 1337–1347), van Erven и Cugliari 2015 (Springer), Athanasopoulos 2017 (EJOR 262(1)), Panagiotelis 2023 (EJOR 306(2)), Ben Taieb, Taylor, Hyndman 2021 (JASA 116(533)), Rangapuram 2021 (ICML, PMLR 139, 8832–8843), Athanasopoulos, Ahmed, Hyndman 2009 (IJF 25(1), 146–166), Jeon 2019 (EJOR 279), Girolimetto 2024 (IJF 40(3)), Aikawa 2026 (TMLR, по mlanthology).
- Содержание сверено с абстрактом: Wickramasuriya 2021 (порядок MinT-U, MinT, OLS, базовые), Hyndman 2011, Wickramasuriya 2019, Panagiotelis 2021, Aikawa (постановка и SDP), Ando и Xiao.

Исправлено:

- Ando и Xiao 2023 существуют: это рабочая статья МВФ WP/23/257, не журнал. Ссылка добавлена в раздел 3.1.
- AAAI-статья Ben Taieb: авторы Ben Taieb, Yu, Barreto, Rajagopal, год 2017. Раньше список авторов был неполным.
- Corani и др.: добавлены четыре автора и страницы; название в ECML отличается от названия на arXiv.
- Ghelasi, Ziel: указаны оба автора.
- Ben Taieb, Taylor, Hyndman 2017: добавлены страницы 3348–3357.
- Athanasopoulos, Ahmed, Hyndman 2009: добавлены страницы, метка [Н] заменена на [П].
- MinT 2019: «четыре оценки» заменено на «пять», так как в таблице пять строк.
- scikit-hts: поддержка Python 3.6–3.8 не подтверждена (requires_python пуст).

Не проверено:

- Gross, Sohl 1990: статья существует, но источники называют разные журналы и страницы. Не цитируйте журнал, пока не откроете страницу издателя.
- Spiliotis 2021 (Applied Soft Computing) и Di Fonzo, Girolimetto (IJF 2023): годы и журналы остались из памяти (метка [Н]).
- Полные тексты Aikawa, Steinmeister и Møller читал автор обзора, не проверяющий. Числа результатов (0,888; 0,808; 0,867; 1,110; около 10% и 9,1%) не пересчитаны.
- Утверждение о ERM «lasso» и режимах hierarchicalforecast взято из кода, при проверке не открывалось.

Числа и библиотеки:

- hierarchicalforecast: версия 1.5.3, загрузка 2026-09-24, Python от 3.10, Apache 2.0. Совпадает с PyPI.
- scikit-hts: версия 0.5.12, 2021-09-02, статус Alpha. Совпадает с PyPI.
- FoReco: версия 1.3.1, опубликована 2026-07-09. Совпадает с CRAN.
- gtop 0.2.0 (CRAN, 2015-03-05): совпадает.

Скачанные данные: файл заявляет пустой список, проверять нечего. Каталог /data/hierfm/docs/papers на pcomp пуст. Пятнадцать PDF из раздела 9 лежат только на недоступном маке и не проверены.

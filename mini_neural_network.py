# -*- coding: utf-8 -*-
"""
Учебная программа: мини-нейросеть 2 -> 3 -> 1 и её прямой проход (forward pass).

z = x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b
y = activation(z)

Ничему не обучает: веса задаёт человек, и сами они не меняются.
Только стандартная библиотека Python (tkinter, math, textwrap): без интернета, без
PyTorch/TensorFlow и без сторонних зависимостей.
Язык интерфейса переключается на ходу: русский (по умолчанию) или английский.

Где что искать:
  0.  Все надписи (ru/en) ......... LANG / TEXTS / T(), справки GUIDE_RU и GUIDE_EN
  1.  Числа сети .................. DEFAULT_VALUES (входы, веса, смещения)
  2.  Функции активации ........... ACTIVATIONS и apply_activation()
  3.  Расчёт одного нейрона ....... class NeuroCalc: произведения -> Σ -> z -> y
  4.  Прямой проход .............. forward_pass(): скрытый слой, затем выход
  5.  Порядок шагов ............... TOTAL_STEPS / STEP_ORDER / REVEAL_STEP
  6.  Текст расчёта ............... neuron_block(), step_blocks(), explain()
  7.  Схема сети .................. class NetworkView (tkinter Canvas)
  8.  Поля ввода .................. ValueControl (ползунок) и WeightEntry (таблица)
  9.  Примеры ..................... EXAMPLES (четыре, каждый с объяснением)
 10.  Окно ........................ class MiniNetworkApp
"""

import math
import textwrap
import tkinter as tk
from tkinter import ttk, font as tkfont

# ---------------------------------------------------------------
# Устройство сети: 2 входа -> 3 нейрона скрытого слоя -> 1 выходной нейрон
# ---------------------------------------------------------------
TOTAL_STEPS = 7          # шагов в пошаговом режиме
SLIDER_MIN, SLIDER_MAX = -5.0, 5.0

# Имена — математические обозначения, они одинаковы на обоих языках.
HIDDEN_NAMES = ("N₁", "N₂", "N₃")
OUTPUT_NAME = "OUT"
SUB = ("₁", "₂", "₃")     # нижние индексы для w₁, w₂, w₃

# Ключи всех чисел сети: h{номер нейрона}w{номер входа} — вес, h{номер}b — смещение.
DEFAULT_VALUES = {
    "x1": 2.0, "x2": 3.0,
    "h0w0": 0.5, "h0w1": 0.2, "h0b": 0.1,
    "h1w0": -0.3, "h1w1": 0.4, "h1b": 0.2,
    "h2w0": 0.1, "h2w1": -0.5, "h2b": 0.3,
    "ow0": 0.8, "ow1": 0.6, "ow2": 0.4, "ob": 0.1,
}

# Строки таблицы весов: (подпись строки, [w₁, w₂, w₃, b]). None — ячейки нет.
TABLE_ROWS = (
    (HIDDEN_NAMES[0], ("h0w0", "h0w1", None, "h0b")),
    (HIDDEN_NAMES[1], ("h1w0", "h1w1", None, "h1b")),
    (HIDDEN_NAMES[2], ("h2w0", "h2w1", None, "h2b")),
    (OUTPUT_NAME, ("ow0", "ow1", "ow2", "ob")),
)

# Цвета: мягкая палитра, смысл у каждого цвета свой.
COL_TEXT = "#242424"
COL_BG = "#fbfbf7"                 # фон окна
COL_PAPER = "#fdfdf9"              # фон «бумажных» панелей
COL_INPUT = "#2f6f9f"              # синий — вход
COL_INPUT_FILL = "#e0eef7"
COL_HIDDEN = "#5b3a86"             # фиолетовый — скрытый слой
COL_HIDDEN_FILL = "#ece5f5"
COL_OUTPUT = "#c8651b"             # оранжевый — выход
COL_OUTPUT_FILL = "#fbeada"
COL_ACTIVE = "#b07d10"             # янтарный — нейрон считается прямо сейчас
COL_DIM = "#91918a"                # серый — значение ещё не вычислено
COL_DIM_FILL = "#efefe9"
COL_FRAME = "#c8c8c0"
COL_LINE = "#8d949b"               # подпись слоя
COL_POS = "#2f7d4f"                # положительный вес
COL_NEG = "#b3261e"                # отрицательный вес
COL_ZERO = "#a8a89f"               # нулевой вес

# ---------------------------------------------------------------
# Локализация: весь текст интерфейса живёт в TEXTS (русский по умолчанию)
# ---------------------------------------------------------------
LANG = "ru"

GUIDE_RU = """1. ЧТО ПОКАЗЫВАЕТ ЭТА ПРОГРАММА
Один проход данных через нейросеть: как числа идут слева направо. Это называется
прямым проходом (forward pass). Ничему не обучается: веса задаёте вы сами.

2. НЕЙРОН, СЛОЙ, СЕТЬ
Нейрон — одно вычисление: взвешенная сумма входов плюс смещение, затем функция
активации.
Слой — несколько нейронов, которые считают параллельно и друг друга не видят.
Сеть — слои, соединённые так, что выходы одного слоя становятся входами следующего.
Здесь: входной слой (2 числа) -> скрытый слой (3 нейрона) -> выходной слой (1 нейрон).

3. ФОРМУЛА НЕЙРОНА
z = x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b
y = activation(z)
w — веса («насколько важен этот вход»), b — смещение (сдвиг, который прибавляется всегда).

4. ПРОИЗВЕДЕНИЯ И СУММА
Каждый вход умножается на свой вес. Все произведения складываются — это Σ.
К Σ прибавляется b. Результат z — сигнал до активации.

5. АКТИВАЦИЯ
Активация превращает z в ответ нейрона y. По умолчанию ReLU:
ReLU(z) = max(0, z) — отрицательное обнуляет, положительное пропускает как есть.
Можно переключить: σ(z) = 1/(1+e⁻ᶻ) даёт число от 0 до 1; tanh(z) — от -1 до 1;
бинарный шаг выдаёт 1 при z >= 0 и 0 иначе.

6. КАК ЧИТАТЬ СХЕМУ
Слева — входы x₁ и x₂ (синие рамки), в середине — три нейрона скрытого слоя
(фиолетовые круги), справа — выходной нейрон (оранжевый круг) и ответ y.
Числа на линиях — веса: зелёный положительный, красный отрицательный, серый нулевой.
Чем толще линия, тем больше модуль веса.
Внутри каждого нейрона — его ответ; прочерк означает «ещё не вычислено».
Клик по нейрону открывает его полный расчёт на вкладке «Расчёт нейрона».

7. ПОЧЕМУ ПОРЯДОК ИМЕННО ТАКОЙ
Ответ нейрона скрытого слоя — это ровно то, что получит выходной нейрон на свой
вход. Поэтому сначала считаются N₁, N₂ и N₃, и только потом OUT.

8. ПОШАГОВЫЙ РЕЖИМ
Кнопка «Шаг вперёд» вычисляет ровно один шаг (всего их семь): входы, затем N₁,
N₂, N₃, передача слоёв, OUT, итог y. Янтарная подсветка показывает, какой нейрон
считается сейчас и какие линии в него входят. «Шаг назад» откатывает один шаг.
Если посреди шагов поменять число — сеть пересчитается целиком.

9. ЧТО МОЖНО ПОПРОБОВАТЬ
· Поменяйте w₁ у N₁ с 0.5 на 2 и посмотрите, как изменится y.
· Сделайте какой-нибудь вес отрицательным и следите за знаком z.
· На ReLU добейтесь, чтобы все три нейрона скрытого слоя выдали 0 (пример 2).
· Переключите на сигмоиду: обнулённый ReLU ответ перестанет быть нулём.

10. ГРАНИЦЫ ПРОГРАММЫ
Нет обучения, backpropagation, градиентного спуска, датасетов, эпох,
автоматического подбора весов и внешних библиотек. Сеть всегда одна и та же:
2 -> 3 -> 1.

Автор: SteveWorks001
"""

GUIDE_EN = """1. WHAT THIS PROGRAM SHOWS
One pass of data through a neural network: how numbers travel from left to right.
That is the forward pass. Nothing is trained — you set the weights yourself.

2. NEURON, LAYER, NETWORK
A neuron performs one calculation: a weighted sum of its inputs plus a bias,
then an activation function.
A layer is several neurons computing in parallel, unaware of each other.
A network is layers connected so that one layer's outputs become the next layer's inputs.
Here: input layer (2 numbers) -> hidden layer (3 neurons) -> output layer (1 neuron).

3. THE NEURON FORMULA
z = x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b
y = activation(z)
w are weights ("how much this input matters"), b is the bias (an offset always added).

4. PRODUCTS AND THE SUM
Each input is multiplied by its own weight. The products are added — that is Σ.
Then b is added to Σ. The result z is the signal before activation.

5. ACTIVATION
Activation turns z into the neuron's answer y. The default is ReLU:
ReLU(z) = max(0, z) — it drops negatives to zero and passes positives unchanged.
You can switch: σ(z) = 1/(1+e⁻ᶻ) returns a number between 0 and 1; tanh(z) between
-1 and 1; the binary step returns 1 when z >= 0 and 0 otherwise.

6. HOW TO READ THE DIAGRAM
Left: the inputs x₁ and x₂ (blue boxes). Middle: the three hidden-layer neurons
(purple circles). Right: the output neuron (orange circle) and the answer y.
The numbers on the lines are weights: green positive, red negative, grey zero.
The thicker the line, the larger the absolute weight.
Inside each neuron is its own output value; a dash means "not computed yet".
Clicking a neuron opens its full calculation on the "Neuron calculation" tab.

7. WHY THIS ORDER
A hidden neuron's output is exactly what the output neuron receives on its input.
So N₁, N₂ and N₃ are computed first, and only then OUT.

8. STEP-BY-STEP MODE
"Step forward" computes exactly one step (seven in total): the inputs, then N₁,
N₂, N₃, the layer hand-over, OUT, and the final y. The amber highlight shows which
neuron is being calculated right now and which lines feed into it. "Step back"
undoes one step. If you change a number in the middle of the walk-through, the
network recalculates fully.

9. THINGS TO TRY
· Change w₁ of N₁ from 0.5 to 2 and watch what happens to y.
· Make some weight negative and follow the sign of z.
· With ReLU, try to make all three hidden neurons output 0 (example 2).
· Switch to sigmoid: an answer that ReLU zeroed stops being zero.

10. LIMITS OF THE PROGRAM
No training, no backpropagation, no gradient descent, no datasets, no epochs, no
automatic weight search, no external libraries. The network is always the same:
2 -> 3 -> 1.

Author: SteveWorks001
"""

TEXTS = {
    "ru": {
        "app_title": "Мини-нейросеть",
        "subtitle": "учебный симулятор прямого прохода · сеть 2 → 3 → 1",
        "lang_label": "Язык",
        "lang_ru": "RU",
        "lang_en": "EN",

        "box_inputs": "Входы сети",
        "box_activation": "Функция активации",
        "box_weights": "Веса и смещения",
        "box_examples": "Примеры",
        "box_result": "Ответ всей сети",

        "lbl_x1": "x₁",
        "lbl_x2": "x₂",
        "col_neuron": "Нейрон",
        "col_w1": "w₁",
        "col_w2": "w₂",
        "col_w3": "w₃",
        "col_b": "b",

        "btn_calc": "Рассчитать весь проход",
        "btn_step": "Шаг вперёд",
        "btn_back": "Шаг назад",
        "btn_reset": "Сбросить всё",
        "btn_ex1": "1. Простой проход",
        "btn_ex2": "2. Отрицательный z",
        "btn_ex3": "3. Разные ответы",
        "btn_ex4": "4. Влияние веса",

        "tab_calc": "Расчёт нейрона",
        "tab_steps": "По шагам",
        "tab_guide": "Справка",

        "sel_neuron": "Выбранный нейрон: {name}",
        "mode_full": "Полный режим: рассчитаны все нейроны.",
        "mode_step": "Пошаговый режим: открыто {n} из {total} шагов.",
        "not_yet": "{name} ещё не вычислен.\nНажмите «Шаг вперёд» ещё {n} раз.",

        "caption_inputs": "ВХОДЫ",
        "caption_hidden": "СКРЫТЫЙ СЛОЙ",
        "caption_output": "ВЫХОД",
        "layer_note": "Нейрон — одно вычисление.  Слой — несколько нейронов, считающих "
                      "параллельно.  Сеть — слои, соединённые вместе: выходы одного слоя "
                      "становятся входами следующего.",
        "what_title": "Что произошло?",

        "status_ready": "Готово. Меняйте числа — прямой проход пересчитывается сам.",
        "status_lang": "Язык интерфейса переключён.",
        "status_calc": "Прямой проход рассчитан целиком: y = {y}.",
        "status_reset": "Сеть возвращена к значениям по умолчанию.",
        "status_err": "Поле «{field}»: «{raw}» — не число. Расчёт остался по последним правильным значениям.",
        "status_example": "Пример «{name}»: {desc}",
        "status_step": "Шаг {n} из {total}: {title}",

        "act_relu": "ReLU — max(0, z)",
        "act_sigmoid": "Сигмоида",
        "act_tanh": "Тангенс (tanh)",
        "act_step": "Бинарный шаг",
        "sym_relu": "ReLU",
        "sym_sigmoid": "σ",
        "sym_tanh": "tanh",
        "sym_step": "шаг",
        "formula_relu": "ReLU(z) = max(0, z)",
        "formula_sigmoid": "σ(z) = 1 / (1 + e⁻ᶻ)",
        "formula_tanh": "tanh(z) — всегда между -1 и 1",
        "formula_step": "шаг(z) = 1 при z ≥ 0, иначе 0",

        "ln_inputs": "Входы",
        "ln_weights": "Веса",
        "ln_products": "Взвешенные входы",
        "ln_sigma": "Взвешенная сумма",
        "ln_bias": "Смещение",
        "ln_z": "Значение до активации",
        "ln_act": "Активация",
        "ln_y": "Ответ нейрона",
        "net_summary": "Ответ всей сети",
        "tpl_named": "{name} = {v}",
        "tpl_product": "  {src}·{wn} = {x} · {w} = {p}",
        "tpl_sigma": "Σ = {terms} = {s}",
        "tpl_z": "z = Σ + b = {s} {op} = {z}",
        "tpl_act": "{sym}({z}) = {y}",
        "tpl_from": "Входы пришли с прошлого слоя:\n  {lst}",

        "step1_title": "Входы сети",
        "step1_desc": "Сеть получает два числа: x₁ = {x1} и x₂ = {x2}. Дальше каждый "
                      "нейрон скрытого слоя умножит их на свои веса.",
        "step2_title": "Нейрон N₁",
        "step2_desc": "Считаем первый нейрон скрытого слоя.",
        "step3_title": "Нейрон N₂",
        "step3_desc": "Считаем второй нейрон скрытого слоя.",
        "step4_title": "Нейрон N₃",
        "step4_desc": "Считаем третий нейрон скрытого слоя.",
        "step5_title": "Передача на следующий слой",
        "step5_desc": "Ответы скрытого слоя становятся входами выходного нейрона: "
                      "x₁ = {o1}, x₂ = {o2}, x₃ = {o3}.",
        "step6_title": "Выходной нейрон",
        "step6_desc": "Один нейрон собирает ответ всей сети.",
        "step7_title": "Итог",
        "step7_desc": "Прямой проход завершён: y = {y}.",

        "why_in_hidden": "{name} получил входы {ins}.",
        "why_in_output": "Выходной нейрон получил ответы скрытого слоя: {ins}.",
        "why_mul": "Он умножил каждый вход на его вес, сложил произведения и прибавил "
                   "смещение: z = {z}.",
        "why_relu_pos": "Так как z = {z} больше 0, ReLU вернула его без изменения: y = {y}.",
        "why_relu_neg": "Так как z = {z} меньше 0, ReLU вернула 0 — нейрон молчит.",
        "why_relu_zero": "z равен ровно 0, и ReLU возвращает 0: y = 0.",
        "why_sigmoid": "Сигмоида сжала z = {z} в {y}: её ответ всегда между 0 и 1.",
        "why_tanh": "Тангенс сжал z = {z} в {y}: его ответ всегда между -1 и 1.",
        "why_step": "Бинарный шаг: z = {z} {cmp} 0, поэтому ответ {y}.",
        "why_next": "Это значение передано следующему слою.",
        "why_final": "Это ответ всей сети: y = {y}.",
        "cmp_ge": "≥",
        "cmp_lt": "<",

        "ex1_name": "Простой проход",
        "ex1_desc": "легко проверить руками: N₁ = 1.7, N₂ = 0.8, N₃ = 0, y = 1.94",
        "ex2_name": "Отрицательный z",
        "ex2_desc": "все три нейрона скрытого слоя молчат (z < 0 → ReLU = 0), а сеть "
                    "всё равно отвечает за счёт смещения",
        "ex3_name": "Разные ответы",
        "ex3_desc": "N₁ = 3.0, N₂ = 2.5, N₃ = 0 — один нейрон обнулён ReLU",
        "ex4_name": "Влияние веса",
        "ex4_desc": "изменили только один вес выходного нейрона (w₁: 0.8 → 0.2): тот же "
                    "скрытый слой, а ответ сети другой",

        "guide": GUIDE_RU,
    },

    "en": {
        "app_title": "Mini Neural Network",
        "subtitle": "an educational forward-pass simulator · network 2 → 3 → 1",
        "lang_label": "Language",
        "lang_ru": "RU",
        "lang_en": "EN",

        "box_inputs": "Network inputs",
        "box_activation": "Activation function",
        "box_weights": "Weights and biases",
        "box_examples": "Examples",
        "box_result": "Whole-network answer",

        "lbl_x1": "x₁",
        "lbl_x2": "x₂",
        "col_neuron": "Neuron",
        "col_w1": "w₁",
        "col_w2": "w₂",
        "col_w3": "w₃",
        "col_b": "b",

        "btn_calc": "Calculate full pass",
        "btn_step": "Step forward",
        "btn_back": "Step back",
        "btn_reset": "Reset all",
        "btn_ex1": "1. Basic pass",
        "btn_ex2": "2. Negative z",
        "btn_ex3": "3. Different outputs",
        "btn_ex4": "4. One weight changed",

        "tab_calc": "Neuron calculation",
        "tab_steps": "Step by step",
        "tab_guide": "Guide",

        "sel_neuron": "Selected neuron: {name}",
        "mode_full": "Full mode: every neuron is calculated.",
        "mode_step": "Step-by-step: {n} of {total} steps revealed.",
        "not_yet": "{name} is not calculated yet.\nPress \"Step forward\" {n} more time(s).",

        "caption_inputs": "INPUTS",
        "caption_hidden": "HIDDEN LAYER",
        "caption_output": "OUTPUT",
        "layer_note": "A neuron is one calculation.  A layer is several neurons computing in "
                      "parallel.  A network is layers connected together: one layer's outputs "
                      "become the next layer's inputs.",
        "what_title": "What happened?",

        "status_ready": "Ready. Change any number — the forward pass recalculates itself.",
        "status_lang": "Interface language switched.",
        "status_calc": "Forward pass fully calculated: y = {y}.",
        "status_reset": "Network restored to its default values.",
        "status_err": "Field \"{field}\": \"{raw}\" is not a number. The calculation kept the last valid values.",
        "status_example": "Example \"{name}\": {desc}",
        "status_step": "Step {n} of {total}: {title}",

        "act_relu": "ReLU — max(0, z)",
        "act_sigmoid": "Sigmoid",
        "act_tanh": "Tanh",
        "act_step": "Binary step",
        "sym_relu": "ReLU",
        "sym_sigmoid": "σ",
        "sym_tanh": "tanh",
        "sym_step": "step",
        "formula_relu": "ReLU(z) = max(0, z)",
        "formula_sigmoid": "σ(z) = 1 / (1 + e⁻ᶻ)",
        "formula_tanh": "tanh(z) — always between -1 and 1",
        "formula_step": "step(z) = 1 when z ≥ 0, otherwise 0",

        "ln_inputs": "Inputs",
        "ln_weights": "Weights",
        "ln_products": "Weighted inputs",
        "ln_sigma": "Weighted sum",
        "ln_bias": "Bias",
        "ln_z": "Value before activation",
        "ln_act": "Activation",
        "ln_y": "Neuron output",
        "net_summary": "Whole-network answer",
        "tpl_named": "{name} = {v}",
        "tpl_product": "  {src}·{wn} = {x} · {w} = {p}",
        "tpl_sigma": "Σ = {terms} = {s}",
        "tpl_z": "z = Σ + b = {s} {op} = {z}",
        "tpl_act": "{sym}({z}) = {y}",
        "tpl_from": "Inputs come from the previous layer:\n  {lst}",

        "step1_title": "Network inputs",
        "step1_desc": "The network receives two numbers: x₁ = {x1} and x₂ = {x2}. Every "
                      "hidden neuron will now multiply them by its own weights.",
        "step2_title": "Neuron N₁",
        "step2_desc": "Calculating the first hidden-layer neuron.",
        "step3_title": "Neuron N₂",
        "step3_desc": "Calculating the second hidden-layer neuron.",
        "step4_title": "Neuron N₃",
        "step4_desc": "Calculating the third hidden-layer neuron.",
        "step5_title": "Handing over to the next layer",
        "step5_desc": "The hidden layer's outputs become the output neuron's inputs: "
                      "x₁ = {o1}, x₂ = {o2}, x₃ = {o3}.",
        "step6_title": "Output neuron",
        "step6_desc": "One neuron collects the answer of the whole network.",
        "step7_title": "Result",
        "step7_desc": "Forward pass finished: y = {y}.",

        "why_in_hidden": "{name} received the inputs {ins}.",
        "why_in_output": "The output neuron received the hidden layer's answers: {ins}.",
        "why_mul": "It multiplied each input by its weight, added the products and the "
                   "bias: z = {z}.",
        "why_relu_pos": "Since z = {z} is greater than 0, ReLU returned it unchanged: y = {y}.",
        "why_relu_neg": "Since z = {z} is less than 0, ReLU returned 0 — the neuron stays silent.",
        "why_relu_zero": "z is exactly 0, and ReLU returns 0: y = 0.",
        "why_sigmoid": "Sigmoid squeezed z = {z} into {y}: its answer is always between 0 and 1.",
        "why_tanh": "Tanh squeezed z = {z} into {y}: its answer is always between -1 and 1.",
        "why_step": "Binary step: z = {z} {cmp} 0, so the answer is {y}.",
        "why_next": "This value is now passed to the next layer.",
        "why_final": "This is the answer of the whole network: y = {y}.",
        "cmp_ge": "≥",
        "cmp_lt": "<",

        "ex1_name": "Basic forward pass",
        "ex1_desc": "easy to check by hand: N₁ = 1.7, N₂ = 0.8, N₃ = 0, y = 1.94",
        "ex2_name": "Negative z with ReLU",
        "ex2_desc": "all three hidden neurons stay silent (z < 0 → ReLU = 0), yet the network "
                    "still answers because of the bias",
        "ex3_name": "Different neuron outputs",
        "ex3_desc": "N₁ = 3.0, N₂ = 2.5, N₃ = 0 — one neuron is zeroed by ReLU",
        "ex4_name": "Effect of changing a weight",
        "ex4_desc": "only one weight of the output neuron changed (w₁: 0.8 → 0.2): same hidden "
                    "layer, different network answer",

        "guide": GUIDE_EN,
    },
}

ACT_KEYS = ("relu", "sigmoid", "tanh", "step")


def T(key, **kw):
    """Текст текущего языка по ключу; подстановки {name} — только если они переданы."""
    text = TEXTS[LANG][key]
    return text.format(**kw) if kw else text


def set_language(code):
    """Сменить язык интерфейса (обновление виджетов делает MiniNetworkApp.retranslate)."""
    global LANG
    LANG = code


def guide_text():
    """Справка на текущем языке (10 разделов)."""
    return TEXTS[LANG]["guide"]


def step_word(k, title):
    """Заголовок шага: «Шаг 3 из 7 — Нейрон N₂» / «Step 3 of 7 — Neuron N₂»."""
    if LANG == "ru":
        return f"Шаг {k} из {TOTAL_STEPS} — {title}"
    return f"Step {k} of {TOTAL_STEPS} — {title}"


# ---------------------------------------------------------------
# Функции активации: четыре штуки, без каталога и лишних абстракций
# ---------------------------------------------------------------
def _relu(z):
    return max(0.0, z)


def _sigmoid(z):
    # math.exp перекошен на больших |z| — считаем устойчивой формой.
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _tanh(z):
    return math.tanh(z)


def _step(z):
    return 1.0 if z >= 0 else 0.0


# ключ: (функция, ключ сокращения, ключ формулы, ключ названия)
ACTIVATIONS = {
    "relu": (_relu, "sym_relu", "formula_relu", "act_relu"),
    "sigmoid": (_sigmoid, "sym_sigmoid", "formula_sigmoid", "act_sigmoid"),
    "tanh": (_tanh, "sym_tanh", "formula_tanh", "act_tanh"),
    "step": (_step, "sym_step", "formula_step", "act_step"),
}


def apply_activation(key, z):
    """y = activation(z) для выбранной функции."""
    return ACTIVATIONS[key][0](z)


def act_sym(key):
    """Сокращение функции активации для формул: ReLU, σ, tanh, шаг/step."""
    return T(ACTIVATIONS[key][1])


# ---------------------------------------------------------------
# Форматирование чисел: в формулах две десятичные цифры, как в учебнике
# ---------------------------------------------------------------
def f2(v):
    """Число для формул: 2 -> «2.00», -0.5 -> «-0.50», мелкий мусор -> «0.00»."""
    if abs(v) < 5e-9:
        v = 0.0
    return f"{v:.2f}"


def fmt(v):
    """Число для полей ввода и подписей на схеме: 2.0 -> «2», 0.25 -> «0.25»."""
    if abs(v - round(v)) < 1e-10 and abs(v) < 1e15:
        return str(int(round(v)))
    return f"{v:.4g}"


def signed(v):
    """Отрицательное число в множителе берём в скобки: 2 · (-0.5)."""
    return f"({f2(v)})" if v < 0 else f2(v)


def sum_terms(values):
    """Список слагаемых строкой: «1.00 + 0.60» или «1.00 - 0.60 - 0.40»."""
    parts = [f2(values[0])]
    for v in values[1:]:
        parts.append(" - " if v < 0 else " + ")
        parts.append(f2(abs(v)))
    return "".join(parts)


def bias_term(b):
    """Прибавление смещения: «+ 0.10» или «- 0.10»."""
    return f"- {f2(abs(b))}" if b < 0 else f"+ {f2(b)}"


# ---------------------------------------------------------------
# Расчёт одного нейрона: все промежуточные числа остаются видимыми
# ---------------------------------------------------------------
class NeuroCalc:
    """Один нейрон и вся его арифметика: входы, веса, произведения, Σ, z, y."""

    def __init__(self, name, inputs, weights, bias, sources, act_key):
        self.name = name                     # «N₁», «N₂», «N₃» или «OUT»
        self.inputs = list(inputs)
        self.weights = list(weights)
        self.bias = bias
        self.sources = list(sources)         # имена того, что пришло по проводам
        self.products = [i * w for i, w in zip(self.inputs, self.weights)]
        self.sigma = sum(self.products)
        self.z = self.sigma + bias
        self.act_key = act_key
        self.y = apply_activation(act_key, self.z)


def forward_pass(values, act_key):
    """Прямой проход: сначала три нейрона скрытого слоя, затем выходной нейрон.

    Возвращает (hidden, out): hidden — список из трёх NeuroCalc, out — NeuroCalc.
    Выходы скрытого слоя буквально подставляются входами выходного нейрона,
    поэтому «выход одного слоя — вход следующего» видно прямо в коде.
    """
    hidden = [
        NeuroCalc(HIDDEN_NAMES[i],
                  [values["x1"], values["x2"]],
                  [values[f"h{i}w{j}"] for j in range(2)],
                  values[f"h{i}b"],
                  ["x₁", "x₂"],
                  act_key)
        for i in range(len(HIDDEN_NAMES))
    ]
    out = NeuroCalc(OUTPUT_NAME,
                    [n.y for n in hidden],
                    [values[f"ow{j}"] for j in range(len(HIDDEN_NAMES))],
                    values["ob"],
                    list(HIDDEN_NAMES),
                    act_key)
    return hidden, out


# ---------------------------------------------------------------
# Порядок вычислений: что уже посчитано на шаге с номером revealed
# ---------------------------------------------------------------
# Шаг 1 — входы, 2..4 — N₁..N₃, 5 — передача слоёв, 6 — OUT, 7 — итог y.
STEP_ORDER = (None, HIDDEN_NAMES[0], HIDDEN_NAMES[1], HIDDEN_NAMES[2],
              "handover", OUTPUT_NAME, "final")

# На каком шаге появляется ответ нейрона (и становится доступен его расчёт).
REVEAL_STEP = {HIDDEN_NAMES[0]: 2, HIDDEN_NAMES[1]: 3, HIDDEN_NAMES[2]: 4,
               OUTPUT_NAME: 6}


def computed_names(revealed):
    """Имена нейронов, ответ которых уже открыт (в полном режиме открыто всё)."""
    return {name for name, step in REVEAL_STEP.items() if revealed >= step}


# ---------------------------------------------------------------
# Текст: полный расчёт нейрона — образец явной подстановки всех чисел
# ---------------------------------------------------------------
def neuron_block(calc):
    """Арифметика нейрона построчно: имя, входы, веса, произведения, Σ, b, z, y."""
    L = [calc.name, ""]
    L.append(T("ln_inputs") + ":")
    for src, x in zip(calc.sources, calc.inputs):
        L.append("  " + T("tpl_named", name=src, v=f2(x)))
    L.append("")
    L.append(T("ln_weights") + ":")
    for i, w in enumerate(calc.weights):
        L.append("  " + T("tpl_named", name="w" + SUB[i], v=f2(w)))
    L.append("")
    L.append(T("ln_products") + ":")
    for i, (src, x, w, p) in enumerate(zip(calc.sources, calc.inputs,
                                           calc.weights, calc.products)):
        L.append(T("tpl_product", src=src, wn="w" + SUB[i], x=f2(x), w=signed(w), p=f2(p)))
    L.append("")
    L.append(T("ln_sigma") + ":")
    L.append(T("tpl_sigma", terms=sum_terms(calc.products), s=f2(calc.sigma)))
    L.append("")
    L.append(T("ln_bias") + f":\n  b = {f2(calc.bias)}")
    L.append("")
    L.append(T("ln_z") + ":")
    L.append(T("tpl_z", s=f2(calc.sigma), op=bias_term(calc.bias), z=f2(calc.z)))
    L.append("")
    L.append(T("ln_act") + ":")
    L.append(T("tpl_act", sym=act_sym(calc.act_key), z=f2(calc.z), y=f2(calc.y)))
    L.append("")
    L.append(T("ln_y") + f":\n  y = {f2(calc.y)}")
    return "\n".join(L)


def from_previous_layer(calc):
    """Для выходного нейрона: явный список того, что пришло со скрытого слоя."""
    return T("tpl_from", lst=", ".join(T("tpl_named", name=s, v=f2(x))
                                       for s, x in zip(calc.sources, calc.inputs)))


def explain(calc, is_output):
    """«Что произошло?» — 3–4 предложения про конкретный нейрон."""
    ins = ", ".join(T("tpl_named", name=s, v=f2(x)) for s, x in zip(calc.sources, calc.inputs))
    out = [T("why_in_output" if is_output else "why_in_hidden",
             name=calc.name, ins=ins),
           T("why_mul", z=f2(calc.z))]
    key = calc.act_key
    if key == "relu":
        if calc.z > 1e-9:
            out.append(T("why_relu_pos", z=f2(calc.z), y=f2(calc.y)))
        elif calc.z < -1e-9:
            out.append(T("why_relu_neg", z=f2(calc.z)))
        else:
            out.append(T("why_relu_zero"))
    elif key == "sigmoid":
        out.append(T("why_sigmoid", z=f2(calc.z), y=f2(calc.y)))
    elif key == "tanh":
        out.append(T("why_tanh", z=f2(calc.z), y=f2(calc.y)))
    elif calc.z >= -1e-9:
        out.append(T("why_step", z=f2(calc.z), cmp=T("cmp_ge"), y=f2(calc.y)))
    else:
        out.append(T("why_step", z=f2(calc.z), cmp=T("cmp_lt"), y=f2(calc.y)))
    out.append(T("why_final" if is_output else "why_next", y=f2(calc.y)))
    return " ".join(out)


def step_blocks(revealed, hidden, out, values):
    """Накопительный текст пошагового режима: шаг 1, шаг 2, ... до открытого."""
    blocks = []
    for k in range(1, revealed + 1):
        title = step_word(k, T(f"step{k}_title"))
        if k == 1:
            head = T("step1_desc", x1=f2(values["x1"]), x2=f2(values["x2"]))
            body = f"  x₁ = {f2(values['x1'])}    x₂ = {f2(values['x2'])}"
        elif k in (2, 3, 4):
            head = T(f"step{k}_desc")
            body = neuron_block(hidden[k - 2])
        elif k == 5:
            head = T("step5_desc", o1=f2(hidden[0].y), o2=f2(hidden[1].y), o3=f2(hidden[2].y))
            body = "\n".join("  " + T("tpl_named", name=n, v=f2(c.y))
                             for n, c in zip(HIDDEN_NAMES, hidden))
        elif k == 6:
            head = T("step6_desc")
            body = neuron_block(out)
        else:
            head = T("step7_desc", y=f2(out.y))
            body = f"  y = {f2(out.y)}"
        blocks.append(f"{title}\n{head}\n{body}")
    return "\n\n".join(blocks)


def reflow(text, width=44):
    """Перенос длинных подписей по ширине панели. Строки с отступом — это
    формулы, их не трогаем: перенос посередине выкладки запутал бы ученика."""
    lines = []
    for line in text.split("\n"):
        if len(line) <= width or line.startswith("  "):
            lines.append(line)
        else:
            lines.extend(textwrap.wrap(line, width=width))
    return "\n".join(lines)


# ---------------------------------------------------------------
# Примеры: четыре готовых набора чисел, каждый со своим уроком
# ---------------------------------------------------------------
EXAMPLES = {
    # 1. Простой проход: N₁ = 1.7, N₂ = 0.8, N₃ = 0, y = 1.94
    "ex1": {"act": "relu", "values": dict(DEFAULT_VALUES)},
    # 2. Отрицательный z: все три нейрона скрытого слоя обнулены ReLU,
    #    ответ сети даёт только смещение выходного нейрона (y = 0.3).
    "ex2": {
        "act": "relu",
        "values": {
            "x1": 1.0, "x2": 2.0,
            "h0w0": -2.0, "h0w1": -1.0, "h0b": 0.5,
            "h1w0": 1.0, "h1w1": -1.0, "h1b": 0.2,
            "h2w0": 2.0, "h2w1": -1.0, "h2b": -0.1,
            "ow0": 1.0, "ow1": 1.0, "ow2": 1.0, "ob": 0.3,
        },
    },
    # 3. Заведомо разные ответы скрытого слоя: 3.0, 2.5 и 0 → y = 0.45
    "ex3": {
        "act": "relu",
        "values": {
            "x1": 2.0, "x2": 1.0,
            "h0w0": 1.5, "h0w1": 0.0, "h0b": 0.0,
            "h1w0": 0.0, "h1w1": 2.0, "h1b": 0.5,
            "h2w0": -1.0, "h2w1": 1.0, "h2b": 0.0,
            "ow0": 0.5, "ow1": -0.5, "ow2": 1.0, "ob": 0.2,
        },
    },
    # 4. Тот же скрытый слой, что и в примере 1, но ow0: 0.8 → 0.2,
    #    и ответ сети меняется с 1.94 на 0.92.
    "ex4": {"act": "relu", "values": dict(DEFAULT_VALUES, ow0=0.2)},
}
EXAMPLE_KEYS = ["ex1", "ex2", "ex3", "ex4"]


# ---------------------------------------------------------------
# Виджеты ввода
# ---------------------------------------------------------------
def _auto_wrap(label, parent, min_width=280, inset=16):
    """Переносить текст надписи по фактической шире родителя.

    Фиксированный wraplength заставлял бы grid отдавать колонке лишнее, и схема
    сети не растягивалась бы в широком окне. inset — это отступы родителя,
    которые нельзя отдать тексту: у LabelFrame они больше, чем у Frame.
    """
    def on_configure(event):
        label.configure(wraplength=max(min_width, event.width - inset))
    parent.bind("<Configure>", on_configure, add="+")


class ValueControl(ttk.Frame):
    """Одно число: подпись + поле ввода + ползунок. Поле и ползунок связаны.

    Печатаете в поле — ползунок едет следом; двигаете ползунок — число появляется
    в поле. Нечисловое поле ползунок не сдвигает: значение останется последним
    правильным.
    """

    def __init__(self, parent, label, lo=SLIDER_MIN, hi=SLIDER_MAX, on_change=None):
        super().__init__(parent)
        self.lo, self.hi = lo, hi
        self.on_change = on_change
        self.var_text = tk.StringVar(value="0")
        self.var_scale = tk.DoubleVar(value=0.0)
        self._lock = False

        self.lbl = ttk.Label(self, text=label)
        self.lbl.pack(side="left")
        self.entry = ttk.Entry(self, width=7, textvariable=self.var_text, justify="right")
        self.entry.pack(side="left", padx=(6, 6))
        self.scale = ttk.Scale(self, from_=lo, to=hi, variable=self.var_scale,
                               command=self._from_scale)
        self.scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.var_text.trace_add("write", self._from_entry)

    def set_label(self, text):
        self.lbl.configure(text=text)

    def set_value(self, v):
        """Программная установка числа (пример, сброс) без вызова on_change."""
        self._lock = True
        self.var_text.set(fmt(v))
        self.var_scale.set(min(self.hi, max(self.lo, v)))
        self._lock = False

    def _from_entry(self, *_):
        if self._lock:
            return
        raw = self.var_text.get().strip().replace(",", ".")
        try:
            v = float(raw)
        except ValueError:
            return                       # дождёмся, пока поле станет числом
        self._lock = True
        self.var_scale.set(min(self.hi, max(self.lo, v)))
        self._lock = False
        self._notify(v)

    def _from_scale(self, *_):
        if self._lock:
            return
        self._lock = True
        self.var_text.set(fmt(round(self.var_scale.get(), 2)))
        self._lock = False
        self._notify(self.var_scale.get())

    def _notify(self, value):
        if self.on_change:
            self.on_change(value)


class WeightEntry(tk.Entry):
    """Поле таблицы весов: число + красная рамка, если ввели не число.

    Обычный tk.Entry, а не ttk: только у него можно менять фон, показывая ошибку.
    """

    def __init__(self, parent, value, on_change):
        super().__init__(parent, width=7, justify="right", relief="solid", bd=1,
                         highlightthickness=1, highlightbackground=COL_FRAME,
                         bg="white", fg=COL_TEXT)
        self.on_change = on_change
        self.insert(0, fmt(value))
        self.bind("<KeyRelease>", self._changed)
        self.bind("<FocusOut>", self._changed)
        self.bind("<Return>", self._changed)

    def set_value(self, v):
        self.delete(0, "end")
        self.insert(0, fmt(v))
        self.ok()

    def ok(self):
        self.configure(bg="white", fg=COL_TEXT)

    def bad(self):
        self.configure(bg="#fde8e6", fg="#7a1a13")

    def _changed(self, *_):
        self.on_change(self.get().strip().replace(",", "."), self.get().strip())


# ---------------------------------------------------------------
# Схема сети на Canvas
# ---------------------------------------------------------------
class NetworkView(ttk.Frame):
    """Рисунок сети: входы, три нейрона, выходной нейрон, все веса и подсветка шага."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.canvas = tk.Canvas(self, bg=COL_BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())

    @staticmethod
    def _point_on_line(p, q, t):
        """Точка отрезка p→q на доле t."""
        return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)

    def _weight_color(self, w):
        if w > 1e-12:
            return COL_POS
        if w < -1e-12:
            return COL_NEG
        return COL_ZERO

    def redraw(self):
        c = self.canvas
        c.delete("all")
        width = max(c.winfo_width(), 380)
        height = max(c.winfo_height(), 300)
        # Отступы в пикселях, а не в долях: слева — половина рамки входа, справа —
        # место под «y = ...», чтобы вывод сети никогда не обрезался.
        c_left, c_right = 58, max(178, width * 0.17)
        cols = (c_left, (c_left + (width - c_right)) / 2, width - c_right)
        mid_y = height * 0.52
        gap_hidden = min(height * 0.24, 112)
        gap_input = min(height * 0.19, 84)
        pos = {
            "x1": (cols[0], mid_y - gap_input),
            "x2": (cols[0], mid_y + gap_input),
            "h0": (cols[1], mid_y - gap_hidden),
            "h1": (cols[1], mid_y),
            "h2": (cols[1], mid_y + gap_hidden),
            "out": (cols[2], mid_y),
        }
        radius = {"h0": 30, "h1": 30, "h2": 30, "out": 37}
        hidden, out = self.app.calculations
        values = self.app.values

        for cx, key in ((cols[0], "caption_inputs"), (cols[1], "caption_hidden"),
                        (cols[2], "caption_output")):
            c.create_text(cx, 22, text=T(key), font=self.app.f_smallb, fill=COL_LINE)

        # провода: вход -> скрытый слой. Подписи двух входов одного нейрона
        # ставим в разных концах провода: у точки пересечения линии сливаются.
        for i in range(3):
            for j in range(2):
                self._edge(pos[f"x{j + 1}"], pos[f"h{i}"], radius[f"h{i}"],
                           values[f"h{i}w{j}"], f"h{i}",
                           (0.34 if j == 0 else 0.70) + 0.02 * i)
        # провода: скрытый слой -> выход
        for i in range(3):
            self._edge(pos[f"h{i}"], pos["out"], radius["out"],
                       values[f"ow{i}"], "out", 0.45 + 0.1 * i)

        # узлы поверх проводов
        self._input_box(pos["x1"], "x₁", values["x1"])
        self._input_box(pos["x2"], "x₂", values["x2"])
        for i, calc in enumerate(hidden):
            self._neuron_node(pos[f"h{i}"], radius[f"h{i}"], f"h{i}", HIDDEN_NAMES[i],
                              calc.y, COL_HIDDEN, COL_HIDDEN_FILL)
        self._neuron_node(pos["out"], radius["out"], "out", OUTPUT_NAME,
                          out.y, COL_OUTPUT, COL_OUTPUT_FILL)

        # стрелка к ответу сети
        ox, oy = pos["out"]
        r = radius["out"]
        c.create_line(ox + r + 5, oy, ox + r + 52, oy, fill=COL_OUTPUT, width=2,
                      arrow="last", arrowshape=(10, 12, 6))
        done = self.app.is_computed("out")
        c.create_text(ox + r + 62, oy - 8, text="y", font=self.app.f_mid,
                      fill=COL_OUTPUT, anchor="w")
        c.create_text(ox + r + 62, oy + 14, text=f2(out.y) if done else "—",
                      font=self.app.f_big, fill=COL_TEXT if done else COL_DIM, anchor="w")

        for key, name in (("h0", HIDDEN_NAMES[0]), ("h1", HIDDEN_NAMES[1]),
                          ("h2", HIDDEN_NAMES[2]), ("out", OUTPUT_NAME)):
            c.tag_unbind(key, "<Button-1>")     # redraw не должен копить обработчики
            c.tag_bind(key, "<Button-1>", lambda e, n=name: self.app.pick(n))

    def _edge(self, source, target, r_target, weight, target_key, t_label):
        """Провод: цвет и толщина по весу, подпись веса, подстветка входящих."""
        c = self.canvas
        x1, y1 = source
        x2, y2 = target
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        ex, ey = x2 - dx / length * (r_target + 3), y2 - dy / length * (r_target + 3)
        highlight = self.app.is_incoming(target_key)
        color = COL_ACTIVE if highlight else self._weight_color(weight)
        width = 1.0 + min(abs(weight), 5.0) * 0.75 + (1.6 if highlight else 0.0)
        c.create_line(x1, y1, ex, ey, fill=color, width=width, arrow="last",
                      arrowshape=(9, 11, 6), capstyle="round", tags=("deco",))
        lx, ly = self._point_on_line(source, (ex, ey), t_label)
        text = fmt(weight)
        half = len(text) * 4 + 5
        c.create_rectangle(lx - half, ly - 9, lx + half, ly + 9,
                           fill=COL_PAPER, outline="", tags=("deco",))
        c.create_text(lx, ly, text=text, font=self.app.f_small, fill=color, tags=("deco",))

    def _halo(self, cx, cy, r):
        self.canvas.create_oval(cx - r - 7, cy - r - 7, cx + r + 7, cy + r + 7,
                                outline=COL_ACTIVE, width=3, tags=("deco",))

    def _neuron_node(self, center, r, key, name, value, edge_color, fill_color):
        """Круг нейрона: имя сверху, ответ внутри, янтарная обводка у считаемого."""
        c = self.canvas
        cx, cy = center
        done = self.app.is_computed(key)
        active = self.app.active == key
        if active:
            self._halo(cx, cy, r)
        if not done:
            edge_color, fill_color = COL_FRAME, COL_DIM_FILL
        outline = COL_ACTIVE if active else edge_color
        weight = 3 if (active or self.app.selected_node_key == key) else 2
        c.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fill_color,
                      outline=outline, width=weight, tags=("node", key))
        c.create_text(cx, cy - r - 15, text=name, font=self.app.f_mid,
                      fill=edge_color, tags=("node", key))
        c.create_text(cx, cy + 1, text=f2(value) if done else "—", font=self.app.f_big,
                      fill=COL_TEXT if done else COL_DIM, tags=("node", key))

    def _input_box(self, center, name, value):
        """Прямоугольник входа — визуально отличается от круга нейрона."""
        c = self.canvas
        cx, cy = center
        w, h = 88, 46
        active = self.app.revealed == 1 and self.app.mode == "step"
        edge = COL_ACTIVE if active else COL_INPUT
        c.create_rectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2,
                           fill=COL_INPUT_FILL, outline=edge, width=3 if active else 2,
                           tags=("deco",))
        c.create_text(cx, cy - 12, text=name, font=self.app.f_mid, fill=COL_INPUT)
        c.create_text(cx, cy + 11, text=fmt(value), font=self.app.f_big, fill=COL_TEXT)


# ---------------------------------------------------------------
# Окно приложения
# ---------------------------------------------------------------
class MiniNetworkApp(tk.Tk):
    """Слева числа, в центре схема сети, справа полный расчёт, шаги и справка."""

    def __init__(self):
        super().__init__()
        self.title(T("app_title"))
        screen_w, screen_h = self.winfo_screenwidth(), self.winfo_screenheight()

        # ---- состояние ----
        self.values = dict(DEFAULT_VALUES)
        self.act_key = "relu"
        self.selected = HIDDEN_NAMES[0]        # нейрон, чей расчёт показываем
        self.mode = "full"                     # "full" — посчитано всё, "step" — по шагам
        self.revealed = TOTAL_STEPS            # сколько шагов открыто
        self.active = None                     # ключ узла, который считается сейчас
        self._silent = False
        self._text_refs = []
        self._tabs = []
        self.calculations = forward_pass(self.values, self.act_key)

        self.var_lang = tk.StringVar(value=LANG)
        self.var_act = tk.StringVar(value="relu")
        self.var_status = tk.StringVar()
        self.var_selected = tk.StringVar()
        self.var_formula = tk.StringVar()
        self.var_mode = tk.StringVar()
        self.var_what = tk.StringVar()
        self.var_result = tk.StringVar()

        # ---- шрифты: 14–20 px для основного текста, крупнее для важных чисел ----
        base = tkfont.nametofont("TkDefaultFont").actual("family")
        self.f_title = tkfont.Font(family=base, size=17, weight="bold")
        self.f_sub = tkfont.Font(family=base, size=14)
        self.f_big = tkfont.Font(family=base, size=19, weight="bold")
        self.f_mid = tkfont.Font(family=base, size=14, weight="bold")
        self.f_small = tkfont.Font(family=base, size=13)
        self.f_smallb = tkfont.Font(family=base, size=13, weight="bold")
        self.f_mono = tkfont.nametofont("TkFixedFont").copy()
        self.f_mono.configure(size=14)
        self.option_add("*Font", self.f_sub)

        self._build_header()
        self._build_left()
        self._build_center()
        self._build_right()
        self._build_status()

        self._fill_guide()
        self.render()
        self.set_status("status_ready")

        # Размер окна: три панели рядом просят около 1640 px по ширине и 880 px
        # по высоте — это и есть minsize. Просим чуть больше, но не выходим за
        # экран. Замерять winfo_reqwidth() здесь нельзя: ещё не развернувшиеся
        # подписи дают завышенную цифру.
        self.geometry(f"{min(1680, screen_w - 60)}x{min(920, screen_h - 60)}")
        self.minsize(1640, 880)

    # ---------- локализация ----------
    def _reg(self, widget, key, attr="text", **kw):
        """Надпись виджета берётся из TEXTS и переведётся при смене языка."""
        widget.configure(**{attr: T(key, **kw)})
        self._text_refs.append((widget, key, attr, kw))
        return widget

    def switch_language(self):
        code = self.var_lang.get()
        if code == LANG:
            return
        set_language(code)
        self.retranslate()
        self.set_status("status_lang")

    def retranslate(self):
        """Перевести статические надписи и заново нарисовать динамические части."""
        self.title(T("app_title"))
        for widget, key, attr, kw in self._text_refs:
            widget.configure(**{attr: T(key, **kw)})
        for tab, key in self._tabs:
            self.nb.tab(tab, text=T(key))
        self.ctrl_x1.set_label(T("lbl_x1"))
        self.ctrl_x2.set_label(T("lbl_x2"))
        for key, button in self.act_buttons.items():
            button.configure(text=T(ACTIVATIONS[key][3]))
        self.var_formula.set(T(ACTIVATIONS[self.act_key][2]))
        self._fill_guide()
        self.render()

    def set_status(self, key, **kw):
        self.var_status.set(T(key, **kw))

    # ---------- построение интерфейса ----------
    def _build_header(self):
        head = ttk.Frame(self)
        head.grid(row=0, column=0, columnspan=3, sticky="ew", padx=12, pady=(12, 2))
        self._reg(ttk.Label(head, font=self.f_title), "app_title").pack(side="left")
        self._reg(ttk.Label(head, font=self.f_sub, foreground="#4a4a4a"),
                  "subtitle").pack(side="left", padx=(18, 0))
        box = ttk.Frame(head)
        box.pack(side="right")
        self._reg(ttk.Label(box, font=self.f_sub), "lang_label").pack(side="left")
        for code, key in (("ru", "lang_ru"), ("en", "lang_en")):
            ttk.Radiobutton(box, text=T(key), value=code, variable=self.var_lang,
                            command=self.switch_language).pack(side="left", padx=(8, 0))
        # Колонки: панель чисел | схема (тянется) | вкладки с расчётом.
        # minsize — это пол колонки: содержимое шире не сожмётся, но и не обрежется.
        self.columnconfigure(0, minsize=378, weight=0)
        self.columnconfigure(1, minsize=580, weight=1)
        self.columnconfigure(2, minsize=450, weight=0)
        self.rowconfigure(1, weight=1)

    def _build_left(self):
        left = ttk.Frame(self, padding=(12, 4))
        left.grid(row=1, column=0, sticky="ns")

        in_box = ttk.LabelFrame(left, padding=10)
        self._reg(in_box, "box_inputs")
        in_box.pack(fill="x", pady=(0, 10))
        self.ctrl_x1 = ValueControl(in_box, T("lbl_x1"),
                                    on_change=lambda v: self.set_input("x1", v))
        self.ctrl_x2 = ValueControl(in_box, T("lbl_x2"),
                                    on_change=lambda v: self.set_input("x2", v))
        self.ctrl_x1.pack(fill="x", pady=3)
        self.ctrl_x2.pack(fill="x", pady=3)
        self.ctrl_x1.set_value(self.values["x1"])
        self.ctrl_x2.set_value(self.values["x2"])

        act_box = ttk.LabelFrame(left, padding=10)
        self._reg(act_box, "box_activation")
        act_box.pack(fill="x", pady=(0, 10))
        self.act_buttons = {}
        for key in ACT_KEYS:
            button = ttk.Radiobutton(act_box, text=T(ACTIVATIONS[key][3]), value=key,
                                     variable=self.var_act, command=self.change_activation)
            button.pack(anchor="w", pady=1)
            self.act_buttons[key] = button
        ttk.Label(act_box, textvariable=self.var_formula, font=self.f_small,
                  foreground="#4a4a4a").pack(anchor="w", pady=(6, 0))
        self.var_formula.set(T(ACTIVATIONS[self.act_key][2]))

        w_box = ttk.LabelFrame(left, padding=10)
        self._reg(w_box, "box_weights")
        w_box.pack(fill="x", pady=(0, 10))
        self.entries = {}
        for col, key in enumerate(("col_neuron", "col_w1", "col_w2", "col_w3", "col_b")):
            self._reg(ttk.Label(w_box, font=self.f_smallb), key).grid(
                row=0, column=col, sticky="ew", padx=3, pady=(0, 4))
        for row, (name, keys) in enumerate(TABLE_ROWS, start=1):
            ttk.Label(w_box, text=name, font=self.f_smallb).grid(row=row, column=0, sticky="w")
            for col, key in enumerate(keys, start=1):
                if key is None:
                    ttk.Label(w_box, text="·").grid(row=row, column=col)
                    continue
                entry = WeightEntry(w_box, self.values[key],
                                    on_change=lambda raw, shown, k=key:
                                    self.set_value(k, raw, shown))
                entry.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
                entry.configure(font=self.f_sub)
                self.entries[key] = entry
        for col in range(1, 5):
            w_box.columnconfigure(col, weight=1)

        res_box = ttk.LabelFrame(left, padding=10)
        self._reg(res_box, "box_result")
        res_box.pack(fill="x", pady=(0, 10))
        ttk.Label(res_box, textvariable=self.var_result, font=self.f_big,
                  foreground=COL_OUTPUT).pack(anchor="w")

    def _build_center(self):
        center = ttk.Frame(self)
        center.grid(row=1, column=1, sticky="nsew", padx=(0, 6), pady=4)
        center.rowconfigure(1, weight=1)
        center.columnconfigure(0, weight=1)

        head = ttk.Frame(center)
        head.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        ttk.Label(head, textvariable=self.var_selected, font=self.f_mid).pack(side="left")
        ttk.Label(head, textvariable=self.var_mode, font=self.f_small,
                  foreground="#4a4a4a").pack(side="right")

        self.view = NetworkView(center, self)
        self.view.grid(row=1, column=0, sticky="nsew")

        note = self._reg(ttk.Label(center, font=self.f_small, foreground="#3d3d3d",
                                   justify="left", anchor="w"), "layer_note")
        note.grid(row=2, column=0, sticky="ew", pady=(4, 6))
        _auto_wrap(note, center, min_width=360)

        buttons = ttk.Frame(center)
        buttons.grid(row=3, column=0, sticky="ew")
        self._reg(ttk.Button(buttons, command=self.calculate_all),
                  "btn_calc").pack(side="left")
        self._reg(ttk.Button(buttons, command=self.step_forward),
                  "btn_step").pack(side="left", padx=(8, 0))
        self._reg(ttk.Button(buttons, command=self.step_back),
                  "btn_back").pack(side="left", padx=(6, 0))
        self._reg(ttk.Button(buttons, command=self.reset_all),
                  "btn_reset").pack(side="left", padx=(6, 0))

        ex_box = ttk.LabelFrame(center, padding=8)
        self._reg(ex_box, "box_examples")
        ex_box.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        for i, key in enumerate(EXAMPLE_KEYS):
            self._reg(ttk.Button(ex_box, command=lambda k=key: self.load_example(k)),
                      f"btn_{key}").pack(side="left", padx=(0 if i == 0 else 6, 0))

    def _build_right(self):
        self.nb = ttk.Notebook(self)
        self.nb.grid(row=1, column=2, sticky="nsew", padx=(0, 12), pady=4)
        for key, name, chars in (("tab_calc", "txt_calc", 46), ("tab_steps", "txt_steps", 46),
                                 ("tab_guide", "txt_guide", 46)):
            frame = ttk.Frame(self.nb, padding=6)
            setattr(self, name,
                    self._text_panel(frame, wrap="word" if key == "tab_guide" else "none",
                                     width=chars))
            self.nb.add(frame, text=T(key))
            self._tabs.append((frame, key))

        what = ttk.LabelFrame(self, padding=10)
        self._reg(what, "what_title")
        what.grid(row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=(0, 4))
        what_label = ttk.Label(what, textvariable=self.var_what, font=self.f_sub,
                               justify="left")
        what_label.pack(anchor="w")
        _auto_wrap(what_label, what, inset=32)

    def _text_panel(self, parent, wrap, width):
        widget = tk.Text(parent, font=self.f_mono, bg=COL_PAPER, fg=COL_TEXT, wrap=wrap,
                         width=width, relief="flat", padx=12, pady=10,
                         highlightthickness=1, highlightbackground=COL_FRAME, state="disabled")
        bar = ttk.Scrollbar(parent, command=widget.yview)
        widget.configure(yscrollcommand=bar.set)
        widget.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        return widget

    def _build_status(self):
        bar = ttk.Frame(self)
        bar.grid(row=3, column=0, columnspan=3, sticky="ew", padx=12, pady=(0, 10))
        ttk.Label(bar, textvariable=self.var_status, font=self.f_small).pack(side="left")
        ttk.Label(bar, text="SteveWorks001", font=self.f_small,
                  foreground="#8a8a82").pack(side="right")

    # ---------- числа ----------
    def set_input(self, key, value):
        """Вход x₁ или x₂: ValueControl отдаёт только разобранные числа."""
        if self._silent:
            return
        self.values[key] = value
        self.recompute()

    def set_value(self, key, raw, shown):
        """Поле таблицы весов изменили: проверяем число, затем пересчитываем сеть."""
        if self._silent:
            return
        entry = self.entries[key]
        try:
            value = float(raw)
        except ValueError:
            entry.bad()
            self.set_status("status_err", field=self.field_name(key), raw=shown)
            return
        entry.ok()
        self.values[key] = value
        self.recompute()

    def field_name(self, key):
        """Понятное имя поля для сообщения об ошибке: «N₂: w₁» или «OUT: b»."""
        row = HIDDEN_NAMES[int(key[1])] if key[0] == "h" else OUTPUT_NAME
        if key.endswith("b"):
            return f"{row}: b"
        return f"{row}: w{SUB[int(key[-1])]}"

    def change_activation(self):
        self.act_key = self.var_act.get()
        self.var_formula.set(T(ACTIVATIONS[self.act_key][2]))
        self.recompute()

    def recompute(self):
        """Прямой проход из текущих чисел. Любая правка возвращает полный режим."""
        self.calculations = forward_pass(self.values, self.act_key)
        self.mode = "full"
        self.revealed = TOTAL_STEPS
        self.active = None
        self.render()

    # ---------- состояние схемы ----------
    @property
    def hidden(self):
        return self.calculations[0]

    @property
    def out(self):
        return self.calculations[1]

    @property
    def selected_node_key(self):
        if self.selected == OUTPUT_NAME:
            return "out"
        return "h%d" % HIDDEN_NAMES.index(self.selected)

    @property
    def selected_calc(self):
        if self.selected == OUTPUT_NAME:
            return self.out
        return self.hidden[HIDDEN_NAMES.index(self.selected)]

    @property
    def revealed_here(self):
        """До какого шага считать открытым (в полном режиме открыто всё)."""
        return TOTAL_STEPS if self.mode == "full" else self.revealed

    def is_computed(self, node_key):
        if node_key in ("x1", "x2"):
            return True
        name = OUTPUT_NAME if node_key == "out" else HIDDEN_NAMES[int(node_key[1])]
        return name in computed_names(self.revealed_here)

    def is_incoming(self, node_key):
        """Подсвечивать ли провода, входящие в узел: он считается прямо сейчас."""
        return self.mode == "step" and self.active == node_key

    def pick(self, name):
        self.selected = name
        self.render()

    # ---------- режимы ----------
    def calculate_all(self):
        self.mode = "full"
        self.revealed = TOTAL_STEPS
        self.active = None
        self.render()
        self.set_status("status_calc", y=f2(self.out.y))

    def step_forward(self):
        if self.mode != "step":
            self.mode, self.revealed = "step", 0
        if self.revealed < TOTAL_STEPS:
            self.revealed += 1
        else:
            self.mode = "full"           # седьмой шаг нажат ещё раз — полный режим
        self.sync_step()
        self.render()
        self.set_status("status_step", n=self.revealed_here, total=TOTAL_STEPS,
                        title=T(f"step{self.revealed_here}_title"))

    def step_back(self):
        if self.mode != "step":
            self.mode, self.revealed = "step", TOTAL_STEPS
        self.revealed = max(1, self.revealed - 1)
        self.sync_step()
        self.render()
        self.set_status("status_step", n=self.revealed, total=TOTAL_STEPS,
                        title=T(f"step{self.revealed}_title"))

    def sync_step(self):
        """По номеру шага: какой узел подсветить и чей расчёт показать."""
        if self.mode != "step":
            self.active = None
            return
        key = STEP_ORDER[self.revealed - 1]
        if key in HIDDEN_NAMES:
            self.active = "h%d" % HIDDEN_NAMES.index(key)
            self.selected = key
        elif key in (OUTPUT_NAME, "handover", "final"):
            self.active = "out"
            self.selected = OUTPUT_NAME
        else:
            self.active = None

    def reset_all(self):
        self.apply_numbers(DEFAULT_VALUES, "relu")
        self.set_status("status_reset")

    def load_example(self, key):
        self.apply_numbers(EXAMPLES[key]["values"], EXAMPLES[key]["act"])
        self.set_status("status_example", name=T(f"{key}_name"), desc=T(f"{key}_desc"))

    def apply_numbers(self, values, act_key):
        """Программно выставить все числа (сброс, пример) и вернуться к полному режиму."""
        self._silent = True
        self.values = dict(values)
        self.act_key = act_key
        self.var_act.set(act_key)
        self.var_formula.set(T(ACTIVATIONS[act_key][2]))
        self.ctrl_x1.set_value(self.values["x1"])
        self.ctrl_x2.set_value(self.values["x2"])
        for k, entry in self.entries.items():
            entry.set_value(self.values[k])
        self._silent = False
        self.selected = HIDDEN_NAMES[0]
        self.mode = "full"
        self.revealed = TOTAL_STEPS
        self.active = None
        self.calculations = forward_pass(self.values, self.act_key)
        self.render()

    # ---------- вывод панелей ----------
    def render(self):
        self.view.redraw()
        self.var_selected.set(T("sel_neuron", name=self.selected))
        self.var_mode.set(T("mode_step", n=self.revealed, total=TOTAL_STEPS)
                          if self.mode == "step" else T("mode_full"))
        self.var_result.set(f"y = {f2(self.out.y)}" if self.is_computed("out") else "y = —")
        self._render_calc()
        self._render_steps()
        self._render_what()

    def _render_calc(self):
        calc = self.selected_calc
        step = REVEAL_STEP[calc.name]
        if self.revealed_here < step:
            text = T("not_yet", name=calc.name, n=step - self.revealed_here)
        else:
            lines = []
            if calc.name == OUTPUT_NAME:
                lines = [from_previous_layer(calc), ""]
            lines.append(neuron_block(calc))
            if self.mode == "full":
                lines += ["", T("net_summary") + ":", f"  y = {f2(self.out.y)}"]
            text = "\n".join(lines)
        self._set_text(self.txt_calc, text)

    def _render_steps(self):
        self._set_text(self.txt_steps,
                       reflow(step_blocks(self.revealed_here, self.hidden,
                                          self.out, self.values)))

    def _render_what(self):
        calc = self.selected_calc
        if self.revealed_here < REVEAL_STEP[calc.name]:
            text = T("not_yet", name=calc.name, n=REVEAL_STEP[calc.name] - self.revealed_here)
        else:
            text = explain(calc, calc.name == OUTPUT_NAME)
        self.var_what.set(text)

    def _set_text(self, widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    def _fill_guide(self):
        self._set_text(self.txt_guide, guide_text())


def main():
    MiniNetworkApp().mainloop()


if __name__ == "__main__":
    main()

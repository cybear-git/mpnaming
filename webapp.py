#!/usr/bin/env python3
"""Веб-приложение: загрузить файл — получить названия — скачать результат,
плюс редактор конфигов (словари/веса/правила/запреты) в разделе «Настройки».

Запуск:  python webapp.py     →  http://127.0.0.1:5000
"""
from __future__ import annotations

import html
import io
import json
import uuid
from pathlib import Path

import pandas as pd
from flask import Flask, redirect, render_template_string, request, send_file, url_for

from namer.bans import DEFAULT_BANS_PATH, Bans
from namer.dictionaries import DEFAULT_DICT_PATH, Dictionaries
from namer.model import DEFAULT_WEIGHTS_PATH, Model
from namer.pipeline import (
    OUTPUT_COLUMN,
    find_reference_column,
    process_dataframe,
    read_table,
    review_dataframe,
    train_model,
)
from namer.rules import DEFAULT_RULES_PATH, Rules

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024

STORAGE: dict[str, pd.DataFrame] = {}

# ----------------------------------------------------------------------
# Общие стили и навигация для всех страниц
# ----------------------------------------------------------------------

STYLE = """
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f6f4;color:#1a1a1a}
 .wrap{max-width:1100px;margin:0 auto;padding:32px 24px}
 h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:0 0 10px}
 .sub{color:#666;font-size:14px;margin-bottom:24px}
 .card{background:#fff;border:1px solid #e3e3df;border-radius:10px;padding:20px;margin-bottom:18px}
 label{font-size:14px;font-weight:600;display:block;margin-bottom:8px}
 input[type=file]{font-size:14px}
 input[type=text],input[type=number],select,textarea{font-size:13px;padding:6px 8px;border:1px solid #ccc;
  border-radius:6px;font-family:inherit}
 textarea{width:100%;box-sizing:border-box;font-family:ui-monospace,Consolas,monospace}
 button{background:#1a1a1a;color:#fff;border:0;border-radius:6px;
  padding:9px 18px;font-size:14px;cursor:pointer} button.ghost{background:#fff;color:#1a1a1a;border:1px solid #ccc}
 button.small{padding:4px 10px;font-size:12px}
 table{border-collapse:collapse;width:100%;font-size:13px} th,td{border-bottom:1px solid #eee;
  padding:7px 8px;text-align:left;vertical-align:top} th{background:#fafaf8;font-weight:600}
 .ok{color:#137333}.bad{color:#b3261e} .stats{display:flex;gap:26px;font-size:14px;flex-wrap:wrap}
 .stats b{display:block;font-size:20px}
 .err{background:#fdecea;border:1px solid #f5c2bd;color:#8c1d18;padding:12px;border-radius:8px;font-size:14px;
  margin-bottom:18px}
 .msg-ok{background:#e6f4ea;border:1px solid #b7ddc0;color:#0d5c26;padding:12px;border-radius:8px;font-size:14px;
  margin-bottom:18px}
 .nav{display:flex;gap:14px;align-items:center;font-size:13px;margin-bottom:20px;flex-wrap:wrap}
 .nav a{color:#1a1a1a;text-decoration:none;padding:5px 10px;border-radius:6px} .nav a:hover{background:#eee}
 .nav .cur{background:#1a1a1a;color:#fff}
 .nav .sep{color:#bbb}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
 .field-row{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px}
 .field-row .f{min-width:160px}
 .hint{color:#888;font-size:12px;margin-top:-4px;margin-bottom:10px}
 .fileops{display:flex;gap:10px;align-items:center;margin-top:10px}
 code{background:#f1f1ee;padding:1px 5px;border-radius:4px;font-size:12px}
"""


def _nav(active: str) -> str:
    items = [
        ("index", "Названия", None),
        ("config_index", "Настройки", "config_index"),
        ("config_dictionaries", "Словари", "config_dictionaries"),
        ("config_weights", "Веса", "config_weights"),
        ("config_rules", "Правила", "config_rules"),
        ("config_bans", "Запреты", "config_bans"),
    ]
    parts = ['<div class="nav">']
    for i, (endpoint, label, _) in enumerate(items):
        cls = "cur" if endpoint == active else ""
        if i == 1:
            parts.append('<span class="sep">|</span>')
        parts.append(f'<a class="{cls}" href="{url_for(endpoint)}">{label}</a>')
    parts.append("</div>")
    return "".join(parts)


PAGE = (
    """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8"><title>Названия карточек товара</title>
<style>"""
    + STYLE
    + """</style>
</head>
<body><div class="wrap">
{{ nav|safe }}
<h1>Названия карточек товара для маркетплейсов</h1>
<div class="sub">Правила ПРОМПТ v5 + обучаемые веса признаков. Модель: {{ meta }}</div>

{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if ok %}<div class="msg-ok">{{ ok }}</div>{% endif %}

<div class="card">
  <form method="post" action="{{ url_for('process') }}" enctype="multipart/form-data">
    <label>Файл с товарами (.xlsx или .csv)</label>
    <input type="file" name="file" accept=".xlsx,.xls,.csv" required>
    <p><button type="submit">Сформировать названия</button></p>
  </form>
</div>

<div class="card">
  <form method="post" action="{{ url_for('train') }}" enctype="multipart/form-data">
    <label>Дообучение: файл с колонкой «Эталонный ответ»</label>
    <input type="file" name="file" accept=".xlsx,.xls,.csv" required>
    <p><button class="ghost" type="submit">Дообучить модель</button></p>
  </form>
</div>

{% if table is not none %}
<div class="card">
  <div class="stats">
    <div>Строк<b>{{ stats.rows }}</b></div>
    <div>Средняя длина<b>{{ stats.avg }}</b></div>
    <div>Вне 30–60<b>{{ stats.out }}</b></div>
    <div>Пустых<b>{{ stats.empty }}</b></div>
    {% if stats.exact is not none %}<div>Совпало по существу<b>{{ stats.exact }}</b></div>{% endif %}
  </div>
  <p><a href="{{ url_for('download', token=token) }}"><button>Скачать xlsx</button></a></p>
  {{ table|safe }}
</div>
{% endif %}
</div></body></html>
"""
)


def _read_upload(upload) -> pd.DataFrame:
    """Читает загруженный файл в DataFrame (xlsx или csv)."""
    data = io.BytesIO(upload.read())
    if upload.filename.lower().endswith((".csv", ".tsv")):
        sep = "\t" if upload.filename.lower().endswith(".tsv") else ","
        return pd.read_csv(data, sep=sep)
    return read_table(data)


def _meta_line() -> str:
    model = Model.load()
    meta = model.meta or {}
    metrics = meta.get("metrics", {})
    if not metrics:
        return "веса не обучены"
    return (
        f"обучена на {meta.get('rows', '?')} строках, "
        f"точное совпадение {metrics.get('exact', 0):.1%}, F1 {metrics.get('f1', 0):.3f}"
    )


@app.get("/")
def index():
    return render_template_string(PAGE, table=None, meta=_meta_line(), token=None,
                                  stats=None, error=request.args.get("error"),
                                  ok=request.args.get("ok"), nav=_nav("index"))


@app.post("/process")
def process():
    upload = request.files["file"]
    dicts, model = Dictionaries(DEFAULT_DICT_PATH), Model.load(DEFAULT_WEIGHTS_PATH)
    df = _read_upload(upload)

    result = process_dataframe(df, dicts, model)
    ref_col = find_reference_column(df)
    review = review_dataframe(df, dicts, model, ref_col)

    token = uuid.uuid4().hex
    STORAGE[token] = result

    lengths = result[OUTPUT_COLUMN].str.len()
    graded = review[review["Совпало"] != ""]
    stats = {
        "rows": len(result),
        "avg": round(float(lengths.mean()), 1),
        "out": int(((lengths < 30) | (lengths > 60)).sum()),
        "empty": int((result[OUTPUT_COLUMN].str.strip() == "").sum()),
        "exact": f"{graded['Совпало'].isin(['да', 'опечатки', 'порядок']).mean():.1%}" if len(graded) else None,
    }

    columns = ["№", "Артикул", "Группа", "Эталонное название", "Название скрипта",
               "Длина", "Совпало", "Не хватает слов", "Лишние слова"]
    html = review[columns].head(200).to_html(index=False, escape=True)
    return render_template_string(PAGE, table=html, meta=_meta_line(), token=token,
                                  stats=stats, error=None, ok=None, nav=_nav("index"))


@app.post("/train")
def train():
    upload = request.files["file"]
    dicts = Dictionaries(DEFAULT_DICT_PATH)
    model = Model.load(DEFAULT_WEIGHTS_PATH)
    df = _read_upload(upload)
    ref_col = find_reference_column(df)
    if not ref_col:
        return redirect(url_for("index", error="В файле нет колонки «Эталонный ответ»"))
    train_model(df, dicts, model, ref_col, epochs=10, verbose=False)
    model.save()
    return redirect(url_for("index"))


@app.get("/download/<token>")
def download(token):
    df = STORAGE.get(token)
    if df is None:
        return redirect(url_for("index", error="Результат устарел, загрузите файл заново"))
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="результат.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ----------------------------------------------------------------------
# Настройки: редактирование словарей / весов / правил / запретов
# ----------------------------------------------------------------------

# Реестр конфиг-файлов: путь, заголовок и «загрузчик» — та же самая функция,
# которую использует движок генерации. Перед сохранением новый файл сначала
# прогоняется через неё же (во временном файле): если конструктор не упал,
# файл структурно годный и можно сохранять поверх настоящего.
CONFIG_FILES = {
    "dictionaries": {
        "path": Path(DEFAULT_DICT_PATH), "title": "Словари",
        "loader": lambda p: Dictionaries(p, DEFAULT_BANS_PATH, DEFAULT_RULES_PATH),
    },
    "bans": {
        "path": Path(DEFAULT_BANS_PATH), "title": "Запреты",
        "loader": lambda p: Bans(p),
    },
    "rules": {
        "path": Path(DEFAULT_RULES_PATH), "title": "Правила",
        "loader": lambda p: Rules(p),
    },
    "weights": {
        "path": Path(DEFAULT_WEIGHTS_PATH), "title": "Веса",
        "loader": lambda p: Model.load(p),
    },
}


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _pretty(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


def _esc(value) -> str:
    """Экранирует значение перед вставкой в HTML, собранный руками (не через Jinja)."""
    return html.escape(str(value), quote=True)


def _save_config_json(kind: str, data: dict) -> str | None:
    """Валидирует и сохраняет data в файл конфига kind.

    Возвращает текст ошибки, если данные не прошли проверку, иначе None.
    """
    info = CONFIG_FILES[kind]
    path: Path = info["path"]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        info["loader"](tmp_path)  # смоук-тест: тот же код, что использует генератор
    except Exception as exc:  # noqa: BLE001 — любая ошибка загрузки конфига
        tmp_path.unlink(missing_ok=True)
        return f"Файл не сохранён, он не проходит проверку: {exc}"
    tmp_path.replace(path)
    return None


def _file_actions_html(kind: str) -> str:
    return (
        '<div class="fileops">'
        f'<a href="{url_for("config_download", kind=kind)}"><button class="ghost" type="button">'
        "Скачать файл</button></a>"
        f'<form method="post" action="{url_for("config_upload", kind=kind)}" '
        'enctype="multipart/form-data" style="display:inline;margin:0">'
        '<input type="file" name="file" accept=".json" required> '
        '<button class="ghost" type="submit">Загрузить файл</button>'
        "</form></div>"
    )


@app.get("/config/<kind>/download")
def config_download(kind):
    if kind not in CONFIG_FILES:
        return redirect(url_for("config_index"))
    path = CONFIG_FILES[kind]["path"]
    return send_file(path, as_attachment=True, download_name=path.name, mimetype="application/json")


@app.post("/config/<kind>/upload")
def config_upload(kind):
    if kind not in CONFIG_FILES:
        return redirect(url_for("config_index"))
    upload = request.files.get("file")
    endpoint = f"config_{kind}"
    if not upload or not upload.filename:
        return redirect(url_for(endpoint, error="Файл не выбран"))
    try:
        data = json.loads(upload.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return redirect(url_for(endpoint, error=f"Это не корректный JSON: {exc}"))
    if not isinstance(data, dict):
        return redirect(url_for(endpoint, error="Ожидается JSON-объект верхнего уровня"))
    err = _save_config_json(kind, data)
    if err:
        return redirect(url_for(endpoint, error=err))
    return redirect(url_for(endpoint, ok=f"Файл {CONFIG_FILES[kind]['path'].name} загружен и сохранён"))


@app.get("/config")
def config_index():
    cards = []
    for kind, info in CONFIG_FILES.items():
        path: Path = info["path"]
        size = path.stat().st_size if path.exists() else 0
        cards.append({"kind": kind, "title": info["title"], "path": str(path), "size": size})
    return render_template_string(
        """
<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Настройки</title>
<style>""" + STYLE + """</style></head>
<body><div class="wrap">
{{ nav|safe }}
<h1>Настройки генератора</h1>
<div class="sub">Словари, веса, правила и запреты — отдельные JSON-файлы. Их можно править прямо здесь,
скачать и отправить заказчику или нейросети, либо загрузить обратно готовый файл.</div>
<div class="grid">
{% for c in cards %}
<div class="card">
  <h2>{{ c.title }}</h2>
  <div class="hint">{{ c.path }} · {{ c.size }} байт</div>
  <a href="{{ url_for('config_' + c.kind) }}"><button>Открыть</button></a>
</div>
{% endfor %}
</div>
</div></body></html>
""",
        cards=cards, nav=_nav("config_index"),
    )


# ---- Запреты -----------------------------------------------------------

BAN_LISTS = ["stop_values", "head_forbidden", "banned_phrases", "discouraged_words"]
BAN_LABELS = {
    "stop_values": "Стоп-значения полей (не становятся признаками совсем)",
    "head_forbidden": "Запрещено как «Вид изделия» (слишком общие значения)",
    "banned_phrases": "Вырезаются из готового названия целиком",
    "discouraged_words": "Не предлагаются специально, но не вырезаются (штраф в «Правилах»)",
}


@app.route("/config/bans", methods=["GET", "POST"])
def config_bans():
    error = request.args.get("error")
    ok = request.args.get("ok")
    if request.method == "POST":
        raw = _read_json(CONFIG_FILES["bans"]["path"])
        action = request.form.get("action")
        if action == "save_lists":
            for key in BAN_LISTS:
                lines = request.form.get(key, "")
                raw[key] = [line.strip() for line in lines.splitlines() if line.strip()]
            err = _save_config_json("bans", raw)
        elif action == "save_raw":
            try:
                data = json.loads(request.form.get("raw_json", ""))
                err = _save_config_json("bans", data) if isinstance(data, dict) else "Ожидается JSON-объект"
            except Exception as exc:  # noqa: BLE001
                err = f"Это не корректный JSON: {exc}"
        else:
            err = "Неизвестное действие"
        if err:
            return redirect(url_for("config_bans", error=err))
        return redirect(url_for("config_bans", ok="Сохранено"))

    raw = _read_json(CONFIG_FILES["bans"]["path"])
    lists_html = []
    for key in BAN_LISTS:
        values = _esc("\n".join(raw.get(key, [])))
        lists_html.append(
            f'<div class="f"><label>{BAN_LABELS[key]} <code>{key}</code></label>'
            f'<textarea name="{key}" rows="8" style="width:100%">{values}</textarea></div>'
        )
    return render_template_string(
        """
<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Запреты</title>
<style>""" + STYLE + """</style></head>
<body><div class="wrap">
{{ nav|safe }}
<h1>Запреты</h1>
<div class="sub">Файл <code>data/bans.json</code>. По одному значению на строку.</div>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if ok %}<div class="msg-ok">{{ ok }}</div>{% endif %}

<div class="card">
  <form method="post">
    <input type="hidden" name="action" value="save_lists">
    <div class="grid">{{ lists_html|safe }}</div>
    <p><button type="submit">Сохранить списки</button></p>
  </form>
  {{ file_actions|safe }}
</div>

<div class="card">
  <h2>Расширенная правка (весь файл)</h2>
  <form method="post">
    <input type="hidden" name="action" value="save_raw">
    <textarea name="raw_json" rows="18">{{ raw_json }}</textarea>
    <p><button class="ghost" type="submit">Сохранить как JSON</button></p>
  </form>
</div>
</div></body></html>
""",
        nav=_nav("config_bans"), error=error, ok=ok,
        lists_html="".join(lists_html), raw_json=_pretty(raw),
        file_actions=_file_actions_html("bans"),
    )


# ---- Правила ------------------------------------------------------------

RULES_SCALAR_FIELDS = [
    ("length", "min", "Целевая нижняя граница длины, символов"),
    ("length", "soft_min", "Реальная нижняя граница (короче не опускаемся никогда)"),
    ("length", "max", "Жёсткий потолок длины, символов"),
    ("length", "target_low", "Ориентир «нормальной» длины, от"),
    ("length", "target_high", "Ориентир «нормальной» длины, до"),
    ("length", "fill_floor", "Минимальный вес признака, добавляемого ради длины"),
    ("composition", "main_threshold", "Порог «основного» волокна в составе, %"),
    ("composition", "fabric_priority_penalty", "Штраф материалу из «Состав», если ткань уже определена по названию"),
    ("composition", "elastane_min_share", "Порог доли эластана для «с эластаном», %"),
    ("discouraged_penalty", None, "Штраф словам из «Запреты → discouraged_words»"),
]


@app.route("/config/rules", methods=["GET", "POST"])
def config_rules():
    error = request.args.get("error")
    ok = request.args.get("ok")
    if request.method == "POST":
        raw = _read_json(CONFIG_FILES["rules"]["path"])
        action = request.form.get("action")
        if action == "save_form":
            for section, key, _label in RULES_SCALAR_FIELDS:
                field_name = f"{section}.{key}" if key else section
                value = request.form.get(field_name)
                if value is None or value == "":
                    continue
                value = float(value) if "." in value or "e" in value.lower() else int(value)
                if key:
                    raw.setdefault(section, {})[key] = value
                else:
                    raw[section] = value
            err = _save_config_json("rules", raw)
        elif action == "save_raw":
            try:
                data = json.loads(request.form.get("raw_json", ""))
                err = _save_config_json("rules", data) if isinstance(data, dict) else "Ожидается JSON-объект"
            except Exception as exc:  # noqa: BLE001
                err = f"Это не корректный JSON: {exc}"
        else:
            err = "Неизвестное действие"
        if err:
            return redirect(url_for("config_rules", error=err))
        return redirect(url_for("config_rules", ok="Сохранено"))

    raw = _read_json(CONFIG_FILES["rules"]["path"])
    fields_html = []
    for section, key, label in RULES_SCALAR_FIELDS:
        field_name = f"{section}.{key}" if key else section
        value = raw.get(section, {}).get(key) if key else raw.get(section)
        fields_html.append(
            f'<div class="f"><label>{label} <code>{field_name}</code></label>'
            f'<input type="text" name="{field_name}" value="{_esc(value)}"></div>'
        )
    return render_template_string(
        """
<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Правила</title>
<style>""" + STYLE + """</style></head>
<body><div class="wrap">
{{ nav|safe }}
<h1>Правила</h1>
<div class="sub">Файл <code>data/rules.json</code> — параметры логики генерации
(раньше были константами в коде).</div>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if ok %}<div class="msg-ok">{{ ok }}</div>{% endif %}

<div class="card">
  <form method="post">
    <input type="hidden" name="action" value="save_form">
    <div class="field-row">{{ fields_html|safe }}</div>
    <p><button type="submit">Сохранить параметры</button></p>
  </form>
  {{ file_actions|safe }}
</div>

<div class="card">
  <h2>Расширенная правка (весь файл)</h2>
  <div class="hint">Здесь же — списки/структуры, для которых нет отдельной формы: fill_order,
  slot_capacity, stop_stems, состав по волокнам (poliviscosa/*, trace_fibers), head_special,
  fallback_head_by_division, head_lexicon_learning.</div>
  <form method="post">
    <input type="hidden" name="action" value="save_raw">
    <textarea name="raw_json" rows="24">{{ raw_json }}</textarea>
    <p><button class="ghost" type="submit">Сохранить как JSON</button></p>
  </form>
</div>
</div></body></html>
""",
        nav=_nav("config_rules"), error=error, ok=ok,
        fields_html="".join(fields_html), raw_json=_pretty(raw),
        file_actions=_file_actions_html("rules"),
    )


# ---- Словари -------------------------------------------------------------

@app.route("/config/dictionaries", methods=["GET", "POST"])
def config_dictionaries():
    error = request.args.get("error")
    ok = request.args.get("ok")
    if request.method == "POST":
        raw = _read_json(CONFIG_FILES["dictionaries"]["path"])
        action = request.form.get("action")
        column = request.form.get("column", "").strip()
        if action == "add_value":
            value = request.form.get("value", "").strip()
            slot = request.form.get("slot", "").strip()
            if not column or not value or not slot:
                return redirect(url_for("config_dictionaries", error="Заполните колонку, значение и слот"))
            if request.form.get("gendered") == "on":
                phrase = {
                    g: request.form.get(f"phrase_{g}", "").strip() or request.form.get("phrase_m", "").strip()
                    for g in ("m", "f", "n", "p")
                }
            else:
                phrase = request.form.get("phrase", "").strip()
            spec = {"slot": int(slot), "phrase": phrase}
            if request.form.get("strong") == "on":
                spec["strong"] = True
            if request.form.get("filler") == "on":
                spec["filler"] = True
            prior = request.form.get("prior", "").strip()
            if prior:
                spec["prior"] = float(prior)
            raw.setdefault("features", {}).setdefault(column, {})[value] = spec
            err = _save_config_json("dictionaries", raw)
        elif action == "delete_value":
            value = request.form.get("value", "")
            raw.get("features", {}).get(column, {}).pop(value, None)
            err = _save_config_json("dictionaries", raw)
        elif action == "save_raw":
            try:
                data = json.loads(request.form.get("raw_json", ""))
                err = _save_config_json("dictionaries", data) if isinstance(data, dict) else "Ожидается JSON-объект"
            except Exception as exc:  # noqa: BLE001
                err = f"Это не корректный JSON: {exc}"
        else:
            err = "Неизвестное действие"
        if err:
            return redirect(url_for("config_dictionaries", error=err, column=column))
        return redirect(url_for("config_dictionaries", ok="Сохранено", column=column))

    raw = _read_json(CONFIG_FILES["dictionaries"]["path"])
    columns = sorted(raw.get("features", {}).keys())
    selected = request.args.get("column") or (columns[0] if columns else "")
    entries = raw.get("features", {}).get(selected, {})

    rows_html = []
    for value, spec in sorted(entries.items()):
        phrase = spec.get("phrase")
        phrase_text = phrase if isinstance(phrase, str) else " / ".join(
            f"{g}:{phrase.get(g, '')}" for g in ("m", "f", "n", "p") if phrase.get(g)
        )
        flags = ", ".join(f for f in ("strong", "filler") if spec.get(f))
        rows_html.append(
            "<tr><td>{value}</td><td>{slot}</td><td>{phrase}</td><td>{flags}</td><td>{prior}</td>"
            '<td><form method="post" style="margin:0">'
            '<input type="hidden" name="action" value="delete_value">'
            '<input type="hidden" name="column" value="{column}">'
            '<input type="hidden" name="value" value="{value}">'
            '<button class="ghost small" type="submit" '
            'onclick="return confirm(\'Удалить это значение?\')">удалить</button></form></td></tr>'.format(
                value=_esc(value), slot=_esc(spec.get("slot", "")), phrase=_esc(phrase_text), flags=_esc(flags),
                prior=_esc(spec.get("prior", 0)), column=_esc(selected),
            )
        )
    options_html = "".join(
        f'<option value="{_esc(c)}"{" selected" if c == selected else ""}>{_esc(c)}</option>' for c in columns
    )

    return render_template_string(
        """
<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Словари</title>
<style>""" + STYLE + """</style></head>
<body><div class="wrap">
{{ nav|safe }}
<h1>Словари</h1>
<div class="sub">Файл <code>data/dictionaries.json</code>. Раздел <code>features</code>:
колонка выгрузки → значение → слот и формулировка.</div>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if ok %}<div class="msg-ok">{{ ok }}</div>{% endif %}

<div class="card">
  <form method="get">
    <label>Колонка</label>
    <select name="column" onchange="this.form.submit()">{{ options_html|safe }}</select>
  </form>
  <table>
    <tr><th>Значение</th><th>Слот</th><th>Формулировка</th><th>Флаги</th><th>Prior</th><th></th></tr>
    {{ rows_html|safe }}
  </table>
</div>

<div class="card">
  <h2>Добавить / изменить значение</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_value">
    <div class="field-row">
      <div class="f"><label>Колонка</label>
        <input type="text" name="column" value="{{ selected }}" list="cols" required>
        <datalist id="cols">{{ options_html|safe }}</datalist>
      </div>
      <div class="f"><label>Значение поля</label><input type="text" name="value" required></div>
      <div class="f"><label>Слот (1–13)</label><input type="number" name="slot" min="1" max="13" required></div>
      <div class="f"><label>Prior</label><input type="number" step="0.1" name="prior" value="0"></div>
    </div>
    <div class="field-row" id="plain-phrase">
      <div class="f"><label>Формулировка (общая форма)</label><input type="text" name="phrase"></div>
    </div>
    <div class="field-row">
      <label><input type="checkbox" name="gendered" onchange="
        document.getElementById('gf').style.display=this.checked?'flex':'none';
        document.getElementById('plain-phrase').style.display=this.checked?'none':'flex';">
      Разные формы по роду/числу</label>
    </div>
    <div class="field-row" id="gf" style="display:none">
      <div class="f"><label>м.р.</label><input type="text" name="phrase_m"></div>
      <div class="f"><label>ж.р.</label><input type="text" name="phrase_f"></div>
      <div class="f"><label>ср.р.</label><input type="text" name="phrase_n"></div>
      <div class="f"><label>мн.ч.</label><input type="text" name="phrase_p"></div>
    </div>
    <div class="field-row">
      <label><input type="checkbox" name="strong"> strong — сильный признак (п. 1.1)</label>
      <label><input type="checkbox" name="filler"> filler — «наполнитель», теряет приоритет при сильном признаке</label>
    </div>
    <p><button type="submit">Сохранить значение</button></p>
  </form>
  {{ file_actions|safe }}
</div>

<div class="card">
  <h2>Расширенная правка (весь файл)</h2>
  <div class="hint">Здесь же — fabric, fiber_translate/main/genitive, valuable/service_fibers,
  head_translate, group_head, gender, head_implies, extras, head_rules, slot_names.</div>
  <form method="post">
    <input type="hidden" name="action" value="save_raw">
    <textarea name="raw_json" rows="24">{{ raw_json }}</textarea>
    <p><button class="ghost" type="submit">Сохранить как JSON</button></p>
  </form>
</div>
</div></body></html>
""",
        nav=_nav("config_dictionaries"), error=error, ok=ok, selected=selected,
        options_html=options_html, rows_html="".join(rows_html) or "<tr><td colspan=6>пусто</td></tr>",
        raw_json=_pretty(raw), file_actions=_file_actions_html("dictionaries"),
    )


# ---- Веса -----------------------------------------------------------------

@app.route("/config/weights", methods=["GET", "POST"])
def config_weights():
    error = request.args.get("error")
    ok = request.args.get("ok")
    if request.method == "POST":
        raw = _read_json(CONFIG_FILES["weights"]["path"])
        action = request.form.get("action")
        weights = raw.setdefault("weights", {})
        if action == "save_table":
            for form_key, form_value in request.form.items():
                if form_key.startswith("value__"):
                    key = form_key[len("value__"):]
                    if request.form.get(f"delete__{key}") == "on":
                        weights.pop(key, None)
                    elif form_value.strip():
                        weights[key] = round(float(form_value), 4)
            err = _save_config_json("weights", raw)
        elif action == "add_weight":
            key = request.form.get("new_key", "").strip()
            value = request.form.get("new_value", "").strip()
            if not key or not value:
                return redirect(url_for("config_weights", error="Укажите ключ и значение"))
            weights[key] = round(float(value), 4)
            err = _save_config_json("weights", raw)
        elif action == "save_raw":
            try:
                data = json.loads(request.form.get("raw_json", ""))
                err = _save_config_json("weights", data) if isinstance(data, dict) else "Ожидается JSON-объект"
            except Exception as exc:  # noqa: BLE001
                err = f"Это не корректный JSON: {exc}"
        else:
            err = "Неизвестное действие"
        q = request.form.get("q", "")
        if err:
            return redirect(url_for("config_weights", error=err, q=q))
        return redirect(url_for("config_weights", ok="Сохранено", q=q))

    raw = _read_json(CONFIG_FILES["weights"]["path"])
    weights = raw.get("weights", {})
    q = request.args.get("q", "").strip().lower()
    matches = [(k, v) for k, v in weights.items() if q in k.lower()] if q else list(weights.items())
    matches.sort(key=lambda kv: -abs(kv[1]))
    total = len(matches)
    matches = matches[:200]

    rows_html = "".join(
        f'<tr><td><code>{_esc(k)}</code></td>'
        f'<td><input type="number" step="0.0001" name="value__{_esc(k)}" value="{_esc(v)}" style="width:100px"></td>'
        f'<td><label><input type="checkbox" name="delete__{_esc(k)}"> удалить</label></td></tr>'
        for k, v in matches
    ) or "<tr><td colspan=3>ничего не найдено</td></tr>"

    return render_template_string(
        """
<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Веса</title>
<style>""" + STYLE + """</style></head>
<body><div class="wrap">
{{ nav|safe }}
<h1>Веса</h1>
<div class="sub">Файл <code>data/weights.json</code> — обучаемые веса признаков
(<code>g::слот|признак</code> — глобальный, <code>ГРУППА::слот|признак</code> — по товарной группе).
Обычно правятся переобучением («Дообучить модель» на главной), эта страница — для точечных правок.</div>
{% if error %}<div class="err">{{ error }}</div>{% endif %}
{% if ok %}<div class="msg-ok">{{ ok }}</div>{% endif %}

<div class="card">
  <form method="get">
    <label>Поиск по ключу (показаны первые 200 совпадений из {{ total }})</label>
    <input type="text" name="q" value="{{ q }}" placeholder="например: ткань|твил или 11::3">
    <button class="ghost" type="submit">Найти</button>
  </form>
</div>

<div class="card">
  <form method="post">
    <input type="hidden" name="action" value="save_table">
    <input type="hidden" name="q" value="{{ q }}">
    <table><tr><th>Ключ</th><th>Значение</th><th></th></tr>{{ rows_html|safe }}</table>
    <p><button type="submit">Сохранить изменения</button></p>
  </form>
</div>

<div class="card">
  <h2>Добавить / изменить один вес</h2>
  <form method="post">
    <input type="hidden" name="action" value="add_weight">
    <div class="field-row">
      <div class="f"><label>Ключ (g::слот|признак или ГРУППА::слот|признак)</label>
        <input type="text" name="new_key" style="width:320px"></div>
      <div class="f"><label>Значение</label><input type="number" step="0.0001" name="new_value"></div>
    </div>
    <p><button type="submit">Сохранить</button></p>
  </form>
  {{ file_actions|safe }}
</div>

<div class="card">
  <h2>Расширенная правка (весь файл)</h2>
  <div class="hint">meta и head_lexicon (выученные виды изделия) правятся только здесь.</div>
  <form method="post">
    <input type="hidden" name="action" value="save_raw">
    <textarea name="raw_json" rows="18">{{ raw_json }}</textarea>
    <p><button class="ghost" type="submit">Сохранить как JSON</button></p>
  </form>
</div>
</div></body></html>
""",
        nav=_nav("config_weights"), error=error, ok=ok, q=q, total=total,
        rows_html=rows_html, raw_json=_pretty(raw), file_actions=_file_actions_html("weights"),
    )


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    app.run(debug=False, port=5000)

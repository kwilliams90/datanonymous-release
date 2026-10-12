"""Round-3 adversarial review, names cluster (R3-04): person names beyond English headers.

R3-04: ``Nombre`` / ``Apellido`` were no name headers, so a small roster's real names became factor levels,
were written verbatim and the report said PASS; the functional dependency ``Nombre -> Apellido`` even kept
the real pairs. These tests pin the whole bug class:

* header vocabulary in the other languages of school exports (Spanish, Portuguese, French, Italian, German,
  Dutch, Tagalog, Vietnamese with and without diacritics, Chinese, Japanese, Korean, Russian, Ukrainian,
  Arabic, Hindi/Urdu): name parts, whole names and person roles, read in Romance word order
  (``Nombre del Padre`` is a person, ``Nombre_Escuela`` is a school, ``Sexo_Alumno`` is an attribute);
  accents never split a header word (``Prénom``); the same languages' phone/e-mail/address/city/ZIP/birth
  date/national-ID headers;
* names under a header that says nothing (``Homeroom`` holding teachers' names): since D35 no value is judged
  by its content; such a value is written verbatim only when it is held by >= ``--min-cell`` entities (the
  accepted residual risk), else as a same-shape pseudonym; the name-list detector (``detect.names``) stays
  for header screens and its false-positive controls (months, colours, school namesakes, cities,
  occupations, things named after people) are still pinned here;
* no functional dependency, run-wide pairing or accent variant brings a real first/last pair back;
* no false refusals: localized header rows are header rows (sniffing), legitimate factors keep their labels.

Assertion messages never print a real value beyond the invented fixture names.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import pytest

from conftest import read_dicts, run_cli
from datanonymous import hints
from datanonymous.detect import names
from datanonymous.generate import pii
from datanonymous.io import sniff
from datanonymous.privacy import leaks

REAL_FIRST = ("Zelmira", "Althea", "Floriana")
REAL_LAST = ("Quenby", "Voss", "Drexel")
NAME_KEYS = ("full_name", "first_name", "last_name", "middle_name")


def _write(path: Path, header: list[str], rows: list[list[object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _texts(*dirs: Path) -> dict[str, str]:
    return {str(p): p.read_text(encoding="utf-8", errors="replace")
            for d in dirs if d.exists() for p in d.rglob("*") if p.is_file()}


def _assert_absent(words: tuple[str, ...] | list[str], *dirs: Path, console: str = "") -> None:
    texts = _texts(*dirs)
    texts["console"] = console
    found = {name: [w for w in words if w in text] for name, text in texts.items()}
    found = {k: v for k, v in found.items() if v}
    assert not found, found


def _run(src: Path, tmp: Path, *extra: str, name: str = "run"):
    out, rep = tmp / f"{name}_out", tmp / f"{name}_rep"
    res = run_cli(src, "--out-dir", out, "--report-dir", rep, "--seed", "123", *extra)
    return res, out, rep


def _hashes(d: Path) -> dict[str, str]:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


def _report(rep: Path, name: str) -> dict:
    return json.loads((rep / "files" / f"{name}.json").read_text(encoding="utf-8"))


def _columns(rep: Path, name: str) -> dict[str, dict]:
    return {c["name"]: c for c in _report(rep, name)["tables"][0]["columns"]}


def _roster(n: int = 120) -> list[list[object]]:
    """The R3-04 reproduction: three real people repeated (Nombre -> Apellido is a 1:1 dependency)."""
    return [[f"S{i:06d}", REAL_FIRST[i % 3], REAL_LAST[i % 3], f"{(i * 37) % 100 + 0.25:.2f}"] for i in range(n)]


# =========================================================================== header vocabulary
@pytest.mark.parametrize("header,key", [
    # Spanish
    ("Nombre", "first_name"), ("Nombres", "first_name"), ("NOMBRE", "first_name"), ("Apellido", "last_name"),
    ("Apellidos", "last_name"), ("APELLIDO_PATERNO", "last_name"), ("Apellido Materno", "last_name"),
    ("ApellidoPaterno", "last_name"), ("AP_PATERNO", "last_name"), ("Primer Apellido", "last_name"),
    ("Segundo Apellido", "last_name"), ("Primer Nombre", "first_name"), ("Segundo Nombre", "middle_name"),
    ("Nombre de pila", "first_name"), ("Nombre completo", "full_name"), ("Nombre y Apellidos", "full_name"),
    ("Apellidos y Nombres", "full_name"), ("Nombre del alumno", "first_name"), ("Nombre del Tutor", "first_name"), ("Alumno_Nombre", "first_name"),
    ("Alumno_Apellido", "last_name"), ("Nombre1", "first_name"), ("Apellido 2", "last_name"),
    ("Nombre(s)", "first_name"), ("Apellido(s)", "last_name"),
    # run-together database column names
    ("apellidopaterno", "last_name"), ("nombrealumno", "first_name"), ("nombrecompleto", "full_name"),
    ("primerapellido", "last_name"), ("nomprenom", "full_name"), ("nomecompleto", "full_name"),
    # Portuguese, French, Italian, German, Dutch, Tagalog
    ("Nome", "first_name"), ("Nome Completo", "full_name"), ("Sobrenome", "last_name"), ("Apelido", "last_name"),
    ("Nome do Aluno", "first_name"), ("Nome da Mãe", "first_name"), ("Prénom", "first_name"),
    ("PRENOM", "first_name"), ("Nom", "last_name"), ("Nom de famille", "last_name"), ("Nom complet", "full_name"),
    ("Nom_Prenom", "full_name"), ("Prénom et nom", "full_name"), ("Nom de l'élève", "last_name"),
    ("Cognome", "last_name"), ("Nome e Cognome", "full_name"), ("Vorname", "first_name"),
    ("Nachname", "last_name"), ("Voornaam", "first_name"), ("Achternaam", "last_name"),
    ("Pangalan", "first_name"), ("Apelyido", "last_name"),
    # Vietnamese (diacritics, and the ASCII phrases), CJK, Korean, Russian, Ukrainian, Arabic, Hindi, Urdu
    ("Họ", "last_name"), ("Tên", "first_name"), ("Họ và tên", "full_name"), ("Ho_Ten", "full_name"),
    ("HoTen", "full_name"), ("Ho va Ten", "full_name"), ("Tên đệm", "middle_name"), ("Ten_Dem", "middle_name"),
    ("姓名", "full_name"), ("学生姓名", "full_name"), ("姓", "last_name"), ("名", "first_name"),
    ("氏名", "full_name"), ("이름", "full_name"), ("성명", "full_name"), ("Имя", "first_name"),
    ("Фамилия", "last_name"), ("Фамилия ученика", "last_name"), ("Отчество", "middle_name"),
    ("ФИО", "full_name"), ("Прізвище", "last_name"), ("الاسم الأول", "first_name"),
    ("اسم العائلة", "last_name"), ("नाम", "full_name"), ("نام", "full_name"),
])
def test_localized_name_headers(header: str, key: str) -> None:
    assert hints.header_hints(header).pii == key


@pytest.mark.parametrize("header", [
    "Padre", "Madre", "Padre o Tutor", "Tutor Legal", "Encargado", "Acudiente", "Responsable", "Docente",
    "Maestra", "Profesor", "Director_Escuela", "Maestro de Grupo", "Contacto de Emergencia", "Estudiante",
    "Alumno", "Responsável", "Aluno", "Professora", "Enseignant", "Élève", "Responsable légal", "Docentes",
    "Mãe", "家长",
])
def test_localized_person_role_headers_are_full_names(header: str) -> None:
    assert hints.header_hints(header).pii == "full_name"


@pytest.mark.parametrize("header", [
    # things named like people (Romance order: the name OF a thing)
    "Nombre_Escuela", "Nombre de la escuela", "Nome_Escola", "Nome da Turma", "Nom de l'école", "Nombre_Curso",
    "Nombre_Materia", "NOMBRE_MUNICIPIO", "Nombre_CCT", "Nombre_Turno", "Nombre_Programa", "Nome_Curso",
    "Nom_Classe", "Escuela_Director",
    # attributes of a person, counts and flags
    "Sexo_Alumno", "Grado_Alumno", "Edad_Alumno", "Numero_Alumno", "Codigo_Alumno", "ID_Alumno",
    "Cantidad_Alumnos", "Total_Docentes", "Tiene_Tutor", "Nombre d'élèves", "nombreescuela", "numeroalumno",
    "sexoalumno", "idalumno", "Renaissance", "Renaissance_Score", "nominal", "lessons", "Delete",
    # plain localized headers and bare words too ambiguous without their diacritics
    "Escuela", "Curso", "Grado", "Grupo", "Turno", "Municipio", "Estado", "Fecha", "Sexo", "Edad", "Promedio",
    "Calificacion", "Primaria", "Legal", "Complete", "Premier", "Usage", "Ten", "Ho", "HO", "MAE",
    # English headers keep their English reading
    "Contact_Method", "Teacher_Type", "Sort_By", "Primary", "Of_Record", "Student_New", "School_Name", "Course",
])
def test_other_headers_are_not_localized_name_hints(header: str) -> None:
    assert hints.header_hints(header).pii not in NAME_KEYS


@pytest.mark.parametrize("header,key", [
    ("Teléfono", "phone"), ("Telefono_Padre", "phone"), ("Padre_Telefono", "phone"), ("Celular", "phone"),
    ("Telefone", "phone"), ("Correo", "email"), ("Correo Electrónico", "email"), ("Courriel", "email"),
    ("Dirección", "address"), ("Domicilio", "address"), ("Endereço", "address"), ("Adresse", "address"),
    ("Ciudad", "city"), ("Nombre_Ciudad", "city"), ("Cidade", "city"), ("Ville", "city"),
    ("Código Postal", "zip"), ("CEP", "zip"), ("Fecha de Nacimiento", "dob"), ("Date de naissance", "dob"),
    ("Data de Nascimento", "dob"), ("CURP", "ssn"), ("CPF", "ssn"), ("Numero_Seguro_Social", "ssn"),
    ("fechanacimiento", "dob"), ("telefonopadre", "phone"), ("correoalumno", "email"),
    ("codigopostal", "zip"),
])
def test_localized_pii_headers(header: str, key: str) -> None:
    assert hints.header_hints(header).pii == key


@pytest.mark.parametrize("header", ["Movilidad", "Municipio", "Estado", "Fecha", "Renaissance_Score",
                                    "Renaissance", "Exception", "Februar", "Callejon_Count"])
def test_localized_pii_words_are_whole_words(header: str) -> None:
    assert hints.header_hints(header).pii not in ("phone", "city", "dob", "zip", "address", "ssn")


def test_accents_never_split_a_header_word() -> None:
    assert hints.normalize("Prénom") == "prenom"
    assert hints.tokens("Responsável") == ("responsavel",)
    assert hints.tokens("NomDeLÉlève") == ("nom", "de", "l", "eleve")
    assert hints.fold_accents("Đặng Øster Straße") == "Dang Oster Strasse"
    assert hints.normalize("Student_ID") == "studentid"  # ASCII headers are unchanged


# =========================================================================== R3-04 end to end
@pytest.mark.parametrize("first,last", [
    ("Nombre", "Apellido"), ("Prénom", "Nom"), ("Nome", "Sobrenome"), ("Vorname", "Nachname"),
    ("Tên", "Họ"), ("Имя", "Фамилия"), ("名", "姓"), ("Nombre del Alumno", "Apellido Paterno"),
])
def test_localized_roster_never_writes_real_names(tmp_path: Path, first: str, last: str) -> None:
    src = _write(tmp_path / "in" / "roster.csv", ["student_id", first, last, "amount"], _roster())
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "roster.csv")
    assert len(syn) == 120  # N in = N out
    cols = _columns(rep, "roster.csv")
    assert cols[first]["type"] == "PERSON_NAME" and cols[last]["type"] == "PERSON_NAME"
    assert len({r[first] for r in syn}) == 3 and len({r[last] for r in syn}) == 3  # distinct counts kept
    _assert_absent(list(REAL_FIRST + REAL_LAST), out, rep, console=res.output)
    # no functional dependency between the two name columns rebuilds a pairing
    structure = json.dumps(_report(rep, "roster.csv")["tables"][0].get("structure", {}))
    assert f"{first} -> {last}" not in structure and f"{last} -> {first}" not in structure
    real_pairs = {(f.casefold(), l.casefold()) for f, l in zip(REAL_FIRST, REAL_LAST)}
    assert not {(r[first].casefold(), r[last].casefold()) for r in syn} & real_pairs


def test_r3_04_reproduction_is_deterministic(tmp_path: Path) -> None:
    src = _write(tmp_path / "in" / "roster.csv", ["student_id", "Nombre", "Apellido", "amount"], _roster())
    res1, out1, _ = _run(src, tmp_path, name="a")
    res2, out2, _ = _run(src, tmp_path, name="b")
    assert res1.exit_code == res2.exit_code == 0
    assert _hashes(out1) == _hashes(out2)


@pytest.mark.parametrize("header", ["Nombre del Padre", "Responsável", "Docente", "Padre o Tutor", "Họ và tên"])
def test_localized_role_columns_never_write_real_names(tmp_path: Path, header: str) -> None:
    rows = [[f"{REAL_FIRST[i % 3]} {REAL_LAST[i % 3]}", i % 7] for i in range(30)]
    src = _write(tmp_path / "in" / "data.csv", [header, "score"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert len(read_dicts(out / "data.csv")) == 30
    assert _columns(rep, "data.csv")[header]["type"] == "PERSON_NAME"
    _assert_absent(list(REAL_FIRST + REAL_LAST), out, rep, console=res.output)


def test_a_factor_determinant_maps_names_onto_fake_names(tmp_path: Path) -> None:
    """Grupo -> Nombre holds in every row: the dependency is kept, but onto the column's own fake names."""
    rows = [[["A", "B", "C"][i % 3], REAL_FIRST[i % 3], REAL_LAST[i % 3], i % 5] for i in range(60)]
    src = _write(tmp_path / "in" / "g.csv", ["Grupo", "Nombre", "Apellido", "n"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "g.csv")
    by_group: dict[str, set[str]] = {}
    for r in syn:
        by_group.setdefault(r["Grupo"], set()).add(r["Nombre"])
    assert all(len(v) == 1 for v in by_group.values())  # the dependency still holds
    _assert_absent(list(REAL_FIRST + REAL_LAST), out, rep, console=res.output)


def test_localized_header_rows_are_header_rows(tmp_path: Path) -> None:
    """Sniffing knows the localized name/role/thing words: these header rows are no headerless records."""
    files = {
        "es.csv": (["Nombre", "Apellido", "Escuela"], [["Maria", "Garcia", "Benito Juarez"],
                                                       ["Jose", "Lopez", "Miguel Hidalgo"]]),
        "fr.csv": (["Prénom", "Nom"], [["Marie", "Dubois"], ["Jean", "Martin"]]),
        "pt.csv": (["Nome do Aluno", "Nome da Mãe", "Escola", "Turma"],
                   [["Joao Silva", "Rita Silva", "EE Tiradentes", "6A"], ["Ana Souza", "Ana Lima", "EM Lobato", "7B"]]),
    }
    for name, (header, rows) in files.items():
        cols = [list(c) for c in zip(*rows)]
        assert sniff.header_problems(header, cols) is None, name
    d = tmp_path / "in"
    for name, (header, rows) in files.items():
        _write(d / name, header, rows * 10)
    res, out, rep = _run(d, tmp_path)
    assert res.exit_code == 0, res.output
    for name, (header, _) in files.items():
        assert list(read_dicts(out / name)[0]) == header


def test_headerless_records_of_localized_header_word_people_are_still_refused() -> None:
    """A word that is a localized header word AND a person's name (``Campos``, ``Calle``) never vouches."""
    taken = names.known_names()[0] | names.known_names()[1]
    words = hints.localized_header_words()
    assert words and not {w for w in words if w in taken or w + "s" in taken}
    assert {"nombre", "apellido", "prenom", "nom", "sobrenome", "escuela", "responsavel"} <= words
    header = ["Campos", "Maria"]
    cols = [["Garcia", "Lopez", "Torres"], ["Ana", "Jose", "Luis"]]
    assert sniff.header_problems(header, cols) is not None


# =========================================================================== names by their values
TEACHERS = ["Mr. Smith", "Mrs. Garcia", "Ms. Nguyen", "Mr. Johnson", "Dr. Patel"]


@pytest.mark.parametrize("header", ["Homeroom", "Assigned", "HR", "Col7", "Responsible_Party", "Advisory"])
def test_unhinted_name_column_is_never_copied(tmp_path: Path, header: str) -> None:
    # D35: each teacher is held by 8 students (< --min-cell 10): every value gets a same-shape pseudonym; the column
    # stays a factor (the R3-04 value-name detection is retired)
    rows = [[f"S{i:04d}", TEACHERS[i % 5], i % 9] for i in range(40)]
    src = _write(tmp_path / "in" / "c.csv", ["student_id", header, "score"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    syn = read_dicts(out / "c.csv")
    assert len(syn) == 40 and len({r[header] for r in syn}) == 5
    col = _columns(rep, "c.csv")[header]
    assert col["type"] == "FACTOR" and col["params"]["pseudonymized_levels"] == 5
    surnames = [t.split()[-1] for t in TEACHERS]
    assert not {r[header].split()[-1] for r in syn} & set(surnames)
    _assert_absent(TEACHERS, out, rep, console=res.output)
    per_file = json.dumps(_report(rep, "c.csv"))
    assert "pseudonymized_levels" in per_file and f"--force-factor {header}" in per_file


def test_force_factor_keeps_value_detected_labels(tmp_path: Path) -> None:
    rows = [[TEACHERS[i % 5], i % 9] for i in range(40)]
    src = _write(tmp_path / "in" / "c.csv", ["Homeroom", "score"], rows)
    res, out, _ = _run(src, tmp_path, "--force-factor", "Homeroom")
    assert res.exit_code == 0, res.output
    assert {r["Homeroom"] for r in read_dicts(out / "c.csv")} == set(TEACHERS)


def test_constant_full_name_and_many_first_names_are_names(tmp_path: Path) -> None:
    firsts = ["Liam", "Olivia", "Noah", "Emma", "Mateo", "Sofia", "Aiden", "Isabella", "Jayden", "Mia", "Elijah",
              "Charlotte", "Lucas", "Amelia", "Mason", "Harper", "Ethan", "Evelyn", "Logan", "Abigail", "Jacob",
              "Emily", "Michael", "Ella", "Daniel"]
    # D35: a constant held by 50 students under a header that hints no personal data is kept (the accepted
    # residual risk; R3-04's value-name detection is retired); with 9 students it is rare, so pseudonymised.
    # 25 first names (> factor-max) are invented labels: never copied
    rows = [[f"S{i:04d}", "Maria Garcia", firsts[i % 25], i] for i in range(50)]
    src = _write(tmp_path / "in" / "k.csv", ["student_id", "Assigned", "Kid", "n"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    cols = _columns(rep, "k.csv")
    assert cols["Assigned"]["type"] == "CONSTANT" and cols["Kid"]["type"] == "HIGH_CARD_CATEGORICAL"
    syn = read_dicts(out / "k.csv")
    assert {r["Assigned"] for r in syn} == {"Maria Garcia"}
    assert not {r["Kid"] for r in syn} & set(firsts)
    small = _write(tmp_path / "in2" / "k.csv", ["student_id", "Assigned", "n"], [r[:2] + [r[3]] for r in rows[:9]])
    res, out, _ = _run(small.parent, tmp_path, name="small")
    assert res.exit_code == 0, res.output
    assert {r["Assigned"] for r in read_dicts(out / "k.csv")} != {"Maria Garcia"}


@pytest.mark.parametrize("values", [
    ["Mr. Smith", "Mrs. Garcia", "Ms. Nguyen"], ["Liam", "Olivia", "Noah", "Emma", "Mateo", "Sofia"],
    ["Smith", "Johnson", "Williams", "Garcia", "Nguyen", "Patel", "Okafor", "Kowalski"],
    ["María José Hernández", "Juan Pérez López", "Ana Gómez", "Luis Martínez"], ["Garcia, Maria", "Smith, John"],
    ["Nguyen", "Tran", "Le", "Pham", "Vu", "Hoang", "Phan"], ["João", "Ana", "Pedro", "Mariana"],
    ["Abdi", "Hassan", "Mohamed", "Fatima", "Farah"], ["Maria Garcia"],
])
def test_value_verdict_recognises_names(values: list[str]) -> None:
    assert names.name_value_verdict({v: 3 for v in values}) is not None


@pytest.mark.parametrize("values", [
    ["White", "Black", "Asian", "Hispanic", "American Indian", "Pacific Islander", "Two or More"],
    ["Male", "Female", "Nonbinary"], ["January", "February", "March", "April", "May", "June"],
    ["Red", "Blue", "Green", "Violet", "Rose"], ["Lincoln", "Washington", "Jefferson", "Roosevelt", "Kennedy"],
    ["Lincoln Elementary", "Washington High", "Jefferson Middle"], ["Austin", "Dallas", "Houston", "Jordan", "Tyler"],
    ["Georgia", "Virginia", "Carolina"], ["North", "South", "East", "West"], ["Eagle", "Falcon", "Hawk", "Owl"],
    ["Lawyer", "Doctor", "Government Officer", "Artist", "Unknown", "Software Engineer", "Teacher", "Banker"],
    ["Algebra I", "Geometry", "Biology", "English", "French"], ["Art", "Band", "Choir", "Drama"],
    ["Jalisco", "Sonora", "Chihuahua", "Guerrero", "Hidalgo", "Colima"], ["Blanco", "Rojo", "Negro", "Verde"],
    ["San José", "Santa María", "San Juan"], ["Pearson", "McGraw Hill", "Houghton Mifflin", "Scholastic"],
    ["Hope", "Faith", "Joy", "Grace"], ["Fall", "Winter", "Spring", "Summer"], ["Mother", "Father", "Other"],
    ["Present", "Absent", "Tardy"], ["Free", "Reduced", "Paid"], ["Washington"], ["Jordan Washington"],
    ["Zelmira", "Althea", "Floriana"],  # rare names alone are no evidence (headers carry them)
])
def test_value_verdict_keeps_labels(values: list[str]) -> None:
    assert names.name_value_verdict({v: 3 for v in values}) is None


@pytest.mark.parametrize("header,values", [
    ("School", ["Benito Juarez", "Miguel Hidalgo", "Emiliano Zapata"]),  # org code (D10): real values kept
    ("Dorm", ["Austen", "Darwin", "Newton", "Curie"]),
    ("Team", ["Smith", "Garcia", "Nguyen"]),
    ("Nombre_Escuela", ["Benito Juarez", "Miguel Hidalgo", "Emiliano Zapata"]),
    ("Month", ["April", "May", "June"]),
    ("Elective", ["Art", "Band", "Choir", "Drama"]),
    ("Career", ["Lawyer", "Doctor", "Banker", "Writer"]),
])
def test_things_named_after_people_keep_their_labels(tmp_path: Path, header: str, values: list[str]) -> None:
    rows = [[values[i % len(values)], i % 4] for i in range(40)]
    src = _write(tmp_path / "in" / "t.csv", [header, "n"], rows)
    res, out, rep = _run(src, tmp_path)
    assert res.exit_code == 0, res.output
    assert _columns(rep, "t.csv")[header]["type"] in ("FACTOR", "ORG_CODE")
    assert {r[header] for r in read_dicts(out / "t.csv")} == set(values)


# =========================================================================== run-wide pairs, accents
def _two_runs(tmp_path: Path, make_other) -> tuple[list[dict], list[dict]]:
    d = tmp_path / "in"
    rows = [["Zelmira Quenby", i] for i in range(20)] + [["Floriana Drexel", i] for i in range(20)]
    _write(d / "people.csv", ["Nombre", "score"], rows)
    res, out1, _ = _run(d, tmp_path, name="alone")
    assert res.exit_code == 0, res.output
    first = read_dicts(out1 / "people.csv")
    _write(d / "staff.csv", ["Nombre", "score"], make_other(first))
    res, out2, _ = _run(d, tmp_path, name="together")
    assert res.exit_code == 0, res.output
    return first, read_dicts(out2 / "people.csv")


def test_whole_names_in_a_given_name_column_are_never_written_elsewhere(tmp_path: Path) -> None:
    """A ``Nombre`` (first-name) column holding whole names: another file never writes one of them."""
    first, second = _two_runs(tmp_path, lambda syn: [[v, 1] for v in dict.fromkeys(r["Nombre"] for r in syn)])
    real_other = {r["Nombre"].casefold() for r in first}  # now real people of staff.csv
    assert not real_other & {r["Nombre"].casefold() for r in second}
    assert len(second) == 40 and len({r["Nombre"] for r in second}) == 2


def _accented(v: str) -> str:
    return v.translate(str.maketrans({"a": "á", "e": "é", "o": "ó", "u": "ú", "i": "í"}))


def test_name_pairs_are_compared_without_accents(tmp_path: Path) -> None:
    """``María García`` and ``Maria Garcia`` are one person: the accent-free spelling is never written."""
    first, second = _two_runs(tmp_path, lambda syn: [[_accented(v), 1]
                                                     for v in dict.fromkeys(r["Nombre"] for r in syn)])
    real_other = {r["Nombre"].casefold() for r in first}
    assert not real_other & {r["Nombre"].casefold() for r in second}
    assert pii.person_words("María García") == ["maria", "garcia"]


# =========================================================================== no real name is tool text
def test_name_word_lists_are_data_not_tool_text() -> None:
    """The leak scanner excuses words of the tool's own literals; the new name/word lists must be data (more
    than ``_MAX_LITERAL_LIST`` entries), so a real ``Taylor`` or ``Campos`` is never excused."""
    for coll in (hints._LOCAL_WORDS, hints._SCRIPT_NAME_HEADERS, names._COMMON_EXTRA, names._VALUE_WORDS,
                 names.FAKER_SOURCES):
        assert len(coll) > leaks._MAX_LITERAL_LIST
    first, last = names.known_names()
    for module in (names, hints):
        src = Path(module.__file__).read_text(encoding="utf-8")
        toks: set[str] = set()
        for s in leaks.literal_texts(src):
            toks |= leaks._public_tokens_of(s)
        added = {"taylor", "carter", "washington", "jackson", "campos", "garcia", "nguyen", "smith"}
        assert not toks & added, module.__name__
    assert ast.parse(Path(names.__file__).read_text(encoding="utf-8"))  # parses (lists are literal displays)
    assert len(first) > 5000 and len(last) > 5000

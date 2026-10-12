"""Deterministic generator of the AREA G privacy fixtures (adversarial PII files).

Run from the repository root::

    python tests/fixtures/privacy/make_fixtures.py

Writes, next to this script:

* ``school/people.csv``: 120 students: zero-padded IDs (``004173``), first/last
  names (with an apostrophe, a hyphen and an accent), derived emails
  ``first.last.id@quillfield-isd.org``, phones, SSNs, street addresses, a
  ``teacher_name`` column with only 3 values (a factor by count, PII by header),
  counselor comments that contain the students' names, and harmless columns.
* ``school/enrollments.csv``: 1-5 rows per student (same zero-padded IDs: one
  namespace across files), course/term org codes, the teacher's email (3
  values), a score and a short note that repeats a student's name.
* ``school/staff.csv``: patterned staff IDs (``T0042``), ``Last, First`` names,
  emails and phone numbers.
* ``sensitive_values.json``: every real sensitive value of the three files, by
  kind, for the tests that grep the outputs (the tests never print them).
* ``adversarial/roster.csv`` (UTF-8 with BOM, CRLF): 10-digit state IDs next to
  10-digit guardian phones (a synthetic ID may equal a real phone's digits by
  chance), ``Last, First`` student names with typographic apostrophes and
  hyphens (``D\u2019Alessandrova``, ``Nkemelu-Haverford``), nicknames, short
  medical alerts (2-3 words, a free-text column under the factor limit), an
  all-missing column, a program factor and scores.
* ``adversarial/one_row.csv``: a single student with names and an email.
* ``adversarial_values.json``: the real sensitive values of ``adversarial/``.

All names are chosen outside Faker's en_US name lists, so an invented name can
never coincide with a fixture name. Same script, same bytes (LF line endings,
UTF-8, no BOM).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOMAIN = "quillfield-isd.org"
FIRST = ["Zephyrine", "Quillon", "Thaddeus", "Ottoline", "Leocadia", "Evander", "Fenella", "Ignatius", "Persephone",
         "Ximena", "Bartholomew", "Cosimo", "Delphine", "Horatio", "Isolde", "Jolyon", "Kasimir", "Lucasta",
         "Marisol", "Nerys", "Octavian", "Philippa", "Rosalind", "Seraphina", "Tybalt", "Ulrika", "Valentijn",
         "Wilhelmina", "Yevgenia", "Zoë", "Anouk", "Bertrand", "Cressida", "Dagny", "Eilidh", "Florian", "Griselda",
         "Hamish", "Ingeborg"]
LAST = ["Ylvisaker", "Brzezinski", "Okonkwo-Vance", "O'Shaughnessy", "Quackenbush", "Thibodeaux", "Szymanski",
        "Kowalczyk", "Lindqvist", "Petrakis", "Vanderbilt", "Nakashima-Ruiz", "Abernathy", "Castellano",
        "Duquesne", "Eriksson", "Fitzwilliam", "Hollingsworth", "Ishikawa", "Jankowski", "Whitfield", "Zielinski",
        "Achterberg", "Bjornstad", "Cavendish", "Delacroix", "Etxeberria", "Fairweather", "Gallardo", "Haverkamp",
        "Iwasaki", "Jablonski", "Kittredge", "Lafontaine", "Nyberg", "Oyelaran", "Pellegrino"]
STREETS = ["Thornbury Ln", "Wexcombe Rd", "Ashgrove Ct", "Pellucid Way", "Marrowbone St", "Quintock Ave",
           "Fernleigh Dr", "Gorsebrook Pl"]
TEACHERS = [("Ms.", "Adaeze", "Okonkwo-Vance"), ("Mr.", "Lorcan", "Brzezinski"), ("Dr.", "Sunniva", "Ylvisaker")]
COMMENT_TEMPLATES = [
    "Met with {first} {last} and a guardian about missing assignments in the second quarter.",
    "{first} reported that {last} family moved to a new address over the winter break.",
    "Counselor called home: {first} {last} has been absent on three consecutive Mondays this term.",
    "Referred {first} to the reading specialist after the October screening results came back low.",
    "",
]
COURSES = ["ALG1", "BIO", "ENG9", "HIST", "CHEM", "ART2"]


def make() -> dict[str, list[str]]:
    rng = random.Random(20261004)
    ids = sorted(rng.sample(range(1000, 99999), 120))
    people = ["student_id,first_name,last_name,email,phone,ssn,home_address,teacher_name,counselor_comment,"
              "class_standing,gender,lunch_status,gpa"]
    sens: dict[str, set[str]] = {k: set() for k in ("ids", "first_names", "last_names", "emails", "phones", "ssns",
                                                    "addresses", "teacher_names", "comments", "notes",
                                                    "staff_ids", "staff_names")}
    students = []
    for i, n in enumerate(ids):
        sid = f"{n:06d}"
        first = FIRST[i % len(FIRST)]
        last = LAST[(i * 7 + 3) % len(LAST)]
        email = f"{first.casefold()}.{last.casefold().replace(chr(39), '')}.{sid}@{DOMAIN}"
        phone = f"({rng.randint(201, 989)}) {rng.randint(201, 989)}-{rng.randint(1000, 9999)}"
        if "555" in phone:
            phone = phone.replace("555", "556")
        ssn = f"{rng.randint(100, 665):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"
        addr = f"{rng.randint(100, 9899)} {STREETS[rng.randrange(len(STREETS))]}"
        title, tf, tl = TEACHERS[i % 3]
        teacher = f"{title} {tf} {tl}"
        comment = COMMENT_TEMPLATES[i % len(COMMENT_TEMPLATES)].format(first=first, last=last)
        standing = ["Freshman", "Sophomore", "Junior", "Senior"][rng.randrange(4)]
        gender = "F" if rng.random() < 0.52 else "M"
        lunch = ["Free", "Reduced", "Paid"][rng.choices(range(3), weights=(4, 1, 5))[0]]
        gpa = f"{rng.uniform(1.2, 4.0):.2f}"
        q = f'"{comment}"' if "," in comment else comment
        people.append(",".join([sid, first, last, email, phone, ssn, addr, teacher, q, standing, gender, lunch, gpa]))
        students.append((sid, first, last))
        sens["ids"].add(sid)
        sens["first_names"].add(first)
        sens["last_names"].add(last)
        sens["emails"].add(email)
        sens["phones"].add(phone)
        sens["ssns"].add(ssn)
        sens["addresses"].add(addr)
        sens["teacher_names"].add(teacher)
        if comment:
            sens["comments"].add(comment)
    enroll = ["student_id,course,term,teacher_email,score,note"]
    for sid, first, last in students:
        for _ in range(rng.randint(1, 5)):
            title, tf, tl = TEACHERS[rng.randrange(3)]
            temail = f"{tf.casefold()}.{tl.casefold()}@{DOMAIN}"
            sens["emails"].add(temail)
            score = str(rng.randint(40, 100)) if rng.random() > 0.06 else "NA"
            note = ""
            if rng.random() < 0.25:
                note = f"{first} {last} asked to retake the unit test after the absence"
                sens["notes"].add(note)
            enroll.append(",".join([sid, COURSES[rng.randrange(len(COURSES))], rng.choice(["Fall", "Spring"]),
                                    temail, score, note]))
    staff = ["staffid,staff_name,email,phone,building"]
    for k in range(12):
        tid = f"T{rng.randint(10, 9999):04d}"
        first = FIRST[(k * 5 + 2) % len(FIRST)]
        last = LAST[(k * 3 + 11) % len(LAST)]
        name = f"{last}, {first}"
        email = f"{first[0].casefold()}{last.casefold().replace(chr(39), '')}@{DOMAIN}"
        phone = f"{rng.randint(201, 989)}.{rng.randint(201, 989)}.{rng.randint(1000, 9999)}".replace("555", "556")
        staff.append(",".join([tid, f'"{name}"', email, phone, rng.choice(["North", "South", "Annex"])]))
        sens["staff_ids"].add(tid)
        sens["staff_names"].add(name)
        sens["emails"].add(email)
        sens["phones"].add(phone)
        sens["first_names"].add(first)
        sens["last_names"].add(last)
    for _, tf, tl in TEACHERS:
        sens["first_names"].add(tf)
        sens["last_names"].add(tl)
    out = {"school/people.csv": people, "school/enrollments.csv": enroll, "school/staff.csv": staff,
           "sensitive_values.json": [json.dumps({k: sorted(v) for k, v in sens.items()}, indent=1,
                                                ensure_ascii=True)]}
    out.update(make_adversarial())
    return out


FIRST2 = ["Ysolde", "Amarantha", "Thorvald", "Idunn", "Perpetua", "Casimira", "Ondine", "Severin", "Lysander",
          "Tamsin", "Ottilie", "Cyprian", "Dashiell", "Evadne", "Gwenllian", "Hesper", "Ignatia", "Jessamy",
          "Kerensa", "Marguerite"]
LAST2 = ["D\u2019Alessandrova", "Nkemelu-Haverford", "O\u2019Rourkevich", "Quennell", "Brightwater", "Oyelowo",
         "Tregarthen", "Pennymore", "Kasprowicz", "Achterhof", "Fennimore", "Marchetti-Valecourt"]
ALERTS = ["Severe peanut allergy", "Bee sting allergy", "Asthma inhaler", "Type 1 diabetes",
          "Epilepsy seizure plan", "Latex allergy"]
PROGRAMS = ["ELL", "SPED", "GT", "None"]


def make_adversarial() -> dict[str, list[str]]:
    """``adversarial/`` files (see the module docstring); lines of roster.csv end in CR for CRLF output."""
    rng = random.Random(41)
    sens: dict[str, set[str]] = {k: set() for k in ("state_ids", "names", "phones", "alerts", "emails", "ids")}
    roster = ["\ufeffstate_id,student_name,nickname,guardian_phone,medical_alert,program,notes,score"]
    ids = sorted(rng.sample(range(10**9, 10**10), 80))
    for i, sid in enumerate(ids):
        first = FIRST2[(i * 3) % len(FIRST2)]
        last = LAST2[(i * 5 + 1) % len(LAST2)]
        nick = FIRST2[(i * 7 + 4) % len(FIRST2)]
        phone = f"({rng.randint(201, 989)}) {rng.randint(201, 989)}-{rng.randint(1000, 9999)}".replace("555", "556")
        alert = ALERTS[rng.randrange(len(ALERTS))] if rng.random() < 0.4 else ""
        roster.append(",".join([str(sid), f'"{last}, {first}"', nick, phone, alert, rng.choice(PROGRAMS), "",
                                str(rng.randint(40, 100))]))
        sens["state_ids"].add(str(sid))
        sens["names"].update({first, last, nick})
        sens["phones"].add(phone)
        if alert:
            sens["alerts"].add(alert)
    roster = [line + "\r" for line in roster]
    first, last = FIRST2[1], LAST2[4]
    email = f"{first.casefold()}.{last.casefold()}@brightwater-academy.org"
    one = ["student_id,first_name,last_name,email", f"0048213,{first},{last},{email}"]
    sens["names"].update({first, last})
    sens["emails"].add(email)
    sens["ids"].add("0048213")
    return {"adversarial/roster.csv": roster, "adversarial/one_row.csv": one,
            "adversarial_values.json": [json.dumps({k: sorted(v) for k, v in sens.items()}, indent=1,
                                                   ensure_ascii=True)]}


def main() -> None:
    for rel, lines in make().items():
        path = HERE / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()

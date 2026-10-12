"""Deterministic generator of the AREA C (text and PII) fixtures. Run from anywhere:

    python tests/fixtures/text/make_fixtures.py

Writes, next to this script:

* ``free_text.csv`` (UTF-8, LF, 240 rows): person names (O'Brien, D'Angelo,
  hyphens, accents, particles, ``Last, First``), derived and free e-mails,
  phones in several masks with extensions, SSNs in two forms, street addresses
  (units, directionals, PO boxes), city/state/ZIP (leading zeros, ZIP+4, rare
  codes), two multi-select columns (``ELL;SPED`` combinations and pipe-delimited
  activities with many combinations), lat/lon, a 3-value teacher_name, a guardian
  name, comments that contain names (and embedded newlines), medical alerts,
  portfolio URLs, IP addresses, locker codes and career goals (high-card labels).
* ``names_cp1252.csv`` (cp1252, CRLF, 30 rows): accented names, to check that
  synthetic names stay encodable in the source encoding.
* ``adversarial_crlf.csv`` (UTF-8, CRLF, 120 rows): hardening cases (see
  :func:`make_adversarial`).

Every value is invented by this script (seeded); none is a real person's data.
"""

from __future__ import annotations

import csv
import io
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
N = 240

FIRST = ["José", "Zoë", "Renée", "Chloé", "Mary-Kate", "Anna-Sofia", "Tavish", "Brennagh", "Oluwaseun", "Xiomara",
         "Thaddeus", "Ignatius", "Perpetua", "Wilhelmina", "Bartholomew", "Eustace", "Philippa", "Cornelius",
         "Evangeline", "Lysander", "Octavia", "Ezekiel", "Marisol", "Dagny", "Soledad", "Anouk", "Ingrid",
         "Leopold", "Ottoline", "Seraphina"]
LAST = ["O'Brien", "D'Angelo", "Smith-Jones", "García", "Nguyen", "McDonald", "de la Cruz", "Okonkwo",
        "Van der Berg", "Featherstonehaugh", "Abernathy", "Quintanilla", "Zielinski", "Thistlewood", "Ravensworth",
        "Kowalczyk", "Achterberg", "Moreno-Ibáñez", "Fitzwilliam", "Oyelaran", "Szczepanski", "Delacroix",
        "Haverbrook", "Mbeki", "Lindqvist"]
MIDDLE = ["J.", "R", "A.", "M.", "K", "Marie", "Lou", "Anne", "Elise", "Jun"]
TEACHERS = [("Mrs. Alvarez-Whitcomb", 100), ("Mr. Thibodeaux", 90), ("Ms. O'Neil", 50)]
CITIES = [("Dallas", 123), ("Fort Worth", 48), ("Grand Prairie", 24), ("Irving", 24), ("Mesquite", 18),
          ("Duncanville", 2), ("Cedar Hill", 1)]
STREETS = ["Oak", "Main", "Elm", "Cedar Ridge", "Martin Luther King Jr", "Pecan Hollow", "Lakeshore", "Sycamore"]
SUFFIXES = ["St", "Street", "Ave", "Dr", "Blvd", "Ln", "Rd", "Ct"]
PROGRAMS = [("ELL", 30), ("SPED", 25), ("ELL;SPED", 30), ("GT", 12), ("504", 6), ("504;SPED", 8), ("ELL;GT", 5)]
ACTIVITIES = ["Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate"]
ALERTS = ["Peanut allergy - EpiPen in nurse office", "Asthma - inhaler in backpack", "Type 1 diabetes",
          "Seizure action plan on file"]
CAREERS = ["Veterinarian", "Marine Biologist", "Electrician", "Pediatric Nurse", "Architect", "Game Designer",
           "Pharmacist", "Firefighter", "Paralegal", "Astronomer", "Welder", "Chef", "Physical Therapist",
           "Translator", "Pilot", "Software Developer", "Dental Hygienist", "Journalist", "Carpenter", "Geologist",
           "Social Worker", "Accountant", "Photographer", "Zookeeper", "Undecided"]
DOMAINS = [("gmail.com", 50), ("yahoo.com", 20), ("lakeview-isd.org", 15), ("outlook.com", 13),
           ("family-okonkwo.net", 2)]


def _ascii(s: str) -> str:
    table = str.maketrans("éëèáíóúñçÉÁÍÓÚÑ", "eeeaiouncEAIOUN")
    return "".join(ch for ch in s.translate(table).lower() if ch.isalnum())


def _weighted(rng: random.Random, pairs: list[tuple[str, int]]) -> str:
    return rng.choices([p for p, _ in pairs], weights=[w for _, w in pairs])[0]


def _phone(rng: random.Random) -> str:
    area = rng.choices(["214", "972", "469", "817", "940"], weights=[60, 30, 4, 0.5, 0.5])[0]
    ex = str(rng.choice([x for x in range(201, 999) if x != 555]))
    line = f"{rng.randrange(10000):04d}"
    form = rng.choices([0, 1, 2, 3], weights=[58, 25, 10, 7])[0]
    if form == 0:
        return f"({area}) {ex}-{line}"
    if form == 1:
        return f"{area}-{ex}-{line}"
    if form == 2:
        return f"{area}.{ex}.{line}"
    return f"({area}) {ex}-{line} x{rng.randrange(10, 999)}"


def _ssn(rng: random.Random) -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    d = f"{area:03d}{rng.randrange(1, 100):02d}{rng.randrange(1, 10000):04d}"
    return d if rng.random() < 0.1 else f"{d[:3]}-{d[3:5]}-{d[5:]}"


def _address(rng: random.Random) -> str:
    if rng.random() < 0.05:
        return f"PO Box {rng.randrange(100, 9999)}"
    num = int(rng.lognormvariate(6.5, 1.0)) + 1
    street = rng.choice(STREETS)
    pre = rng.choice(["N ", "S ", ""]) if rng.random() < 0.2 else ""
    out = f"{num} {pre}{street} {rng.choice(SUFFIXES)}"
    r = rng.random()
    if r < 0.15:
        out += f" Apt {rng.randrange(1, 40)}{rng.choice('ABCD')}"
    elif r < 0.25:
        out += f", Unit {rng.randrange(1, 300)}"
    return out


def _zip(rng: random.Random) -> str:
    z = rng.choices(["75201", "75202", "76102", "75050", "07030", "02134", "75203"],
                    weights=[30, 20, 15, 13, 6, 1.2, 0.4])[0]
    if rng.random() < 0.12:
        return f"{z}-{rng.randrange(1000, 9999)}"
    return z


def _comment(rng: random.Random, first: str, last: str, teacher: str) -> str:
    kind = rng.randrange(7)
    if kind == 0:
        return f"Spoke with Mrs. {last} about attendance on {rng.randrange(1, 13)}/{rng.randrange(1, 29)}."
    if kind == 1:
        return f"{first} reads at level {rng.randrange(2, 9)}; great progress since fall."
    if kind == 2:
        return f"Called home.\nNo answer, left voicemail for {first}'s guardian."
    if kind == 3:
        return f"{teacher} reports {first} missed {rng.randrange(2, 7)} assignments; parent conference set."
    if kind == 4:
        return "Excellent participation in class discussions this week."
    if kind == 5:
        return f"IEP review scheduled with {first} {last} and counselor."
    return f"Moved seats closer to the board. Follow up in {rng.randrange(2, 5)} weeks."


def make_free_text(rng: random.Random) -> list[list[str]]:
    header = ["student_id", "first_name", "middle_name", "last_name", "student_name", "student_email",
              "parent_email", "parent_phone", "ssn", "home_address", "city", "state", "zip", "programs",
              "activities", "home_lat", "home_lon", "teacher_name", "Guardian", "comments", "medical_alert",
              "portfolio_url", "last_login_ip", "locker_code", "career_goal"]
    first_w = [max(1, int(20 / (i + 1))) for i in range(len(FIRST))]
    last_w = [max(1, int(14 / (i + 1))) for i in range(len(LAST))]
    teachers = [t for t, c in TEACHERS for _ in range(c)]
    rng.shuffle(teachers)
    cities = [c for c, k in CITIES for _ in range(k)]
    rng.shuffle(cities)
    rows = [header]
    used_lockers: set[str] = set()
    for i in range(N):
        first = rng.choices(FIRST, weights=first_w)[0]
        last = rng.choices(LAST, weights=last_w)[0]
        middle = rng.choice(MIDDLE) if rng.random() < 0.55 else ("N/A" if rng.random() < 0.1 else "")
        email = f"{_ascii(first)}.{_ascii(last)}{rng.randrange(10, 99)}@students.lakeview-isd.org"
        dom = _weighted(rng, DOMAINS)
        pmail = (f"{_ascii(first)[:1]}{_ascii(last)}{rng.randrange(1, 999)}@{dom}" if rng.random() < 0.9
                 else f"{_ascii(last)}.family@{dom}")
        if rng.random() < 0.05:
            pmail = ""
        phone = _phone(rng) if rng.random() < 0.93 else rng.choice(["", "N/A"])
        ssn = _ssn(rng) if rng.random() < 0.95 else ""
        state = rng.choices(["TX", "OK", "LA"], weights=[236, 3, 1])[0]
        prog = _weighted(rng, PROGRAMS) if rng.random() < 0.45 else ""
        acts = "|".join(sorted(rng.sample(ACTIVITIES, rng.choice([1, 1, 2, 2, 3, 4])),
                               key=ACTIVITIES.index)) if rng.random() < 0.7 else ""
        lat = f"{rng.uniform(32.6, 33.0):.6f}"
        lon = f"{rng.uniform(-97.0, -96.5):.6f}"
        guardian = f"{rng.choice(['Maritza', 'Desmond', 'Lucinda', 'Florentino', 'Agnieszka', 'Theodora'])} {last}"
        teacher = teachers[i]
        comment = _comment(rng, first, last, teacher) if rng.random() < 0.7 else ""
        alert = rng.choice(ALERTS) if rng.random() < 0.12 else ""
        url = ""
        if rng.random() < 0.3:
            url = rng.choice([f"https://sites.google.com/view/{_ascii(first)}{_ascii(last)}{rng.randrange(1, 99)}",
                              f"https://www.{_ascii(last)}family.net/{_ascii(first)}",
                              f"http://github.com/{_ascii(first)}-{_ascii(last)}"])
        ip = (f"10.12.{rng.randrange(0, 255)}.{rng.randrange(1, 255)}" if rng.random() < 0.6
              else f"192.168.{rng.randrange(0, 3)}.{rng.randrange(1, 255)}")
        if rng.random() < 0.03:
            ip = f"2600:1700:{rng.randrange(0, 65535):x}::{rng.randrange(1, 4095):x}"
        while True:
            locker = f"{rng.choice('ABCDEFGH')}-{rng.randrange(0, 10000):04d}"
            if locker not in used_lockers:
                used_lockers.add(locker)
                break
        career = rng.choices(CAREERS, weights=[max(1, 30 - 1 * k) for k in range(len(CAREERS))])[0]
        rows.append([f"{100000 + 7 * i:07d}", first, middle, last, f"{last}, {first}", email, pmail, phone, ssn,
                     _address(rng), cities[i], state, _zip(rng), prog, acts, lat, lon, teacher, guardian, comment,
                     alert, url, ip, locker, career])
    return rows


def make_names_cp1252(rng: random.Random) -> list[list[str]]:
    firsts = ["José", "Renée", "Chloé", "Zoë", "Ignacio", "Marisol", "Soledad", "Ramón", "Inés", "Agustín"]
    lasts = ["García", "Muñoz", "Peña", "Ibáñez", "Núñez", "Ortiz", "Calderón", "Fernández"]
    rows = [["FirstName", "LastName", "score"]]
    for _ in range(30):
        rows.append([rng.choice(firsts), rng.choice(lasts), str(rng.randrange(50, 100))])
    return rows


def _write(path: Path, rows: list[list[str]], encoding: str, line_ending: str) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=line_ending, quoting=csv.QUOTE_MINIMAL)
    w.writerows(rows)
    path.write_bytes(buf.getvalue().encode(encoding))


ADV_N = 120
GUARDIANS = ["John & Mary Smith", "Ana / Luis García", "Mr. and Mrs. Lee", "Pat O'Neil", "Dr. Renée McDonald",
             "Kofi & Ama Mensah", "Sung-Min / Ji-Woo Park", "Mrs. Ingrid Lindqvist"]
ADV_ACTIVITIES = ["Band", "Choir", "Soccer", "Chess", "Robotics", "Drama", "Art", "Debate", "Math Club"]
ADV_NOTES = ["Called home.\r\nNo answer; left message for {g}.", "Great week \U0001F44D", "See note:\nparent called",
             "Moved seats (row {k}).", "{g} picked up early at 2:15"]
ADV_CITIES = ["Dallas, TX", "Fort Worth, TX", "Tulsa, OK", "Shreveport, LA", "Irving, TX", "Plano, TX"]


def make_adversarial(rng: random.Random) -> list[list[str]]:
    """Adversarial edge cases for AREA C (written as UTF-8, CRLF): '&'/'/' couples in a name column,
    'City, ST' values, a multi-select with a whole-value NA token ('None') and mixed '; ' separators,
    numeric columns under phone/SSN headers, notes with embedded CRLF/LF, emoji and one long whitespace run,
    and a zero-padded 2-digit code whose whole 01-99 space is real (masks must leave the shape)."""
    header = ["Guardian", "city_state", "activities", "phone_ext", "ssn_last4", "notes", "seat"]
    rows = [header]
    seats = [f"{i:02d}" for i in range(1, 100)] + [f"{rng.randrange(1, 100):02d}" for _ in range(ADV_N - 99)]
    rng.shuffle(seats)
    for i in range(ADV_N):
        g = rng.choice(GUARDIANS)
        k = rng.choice([0, 0, 1, 1, 2, 3])
        if k == 0:
            acts = rng.choice(["None", "None", ""])
        else:
            sep = rng.choice([";", "; "])
            acts = sep.join(sorted(rng.sample(ADV_ACTIVITIES, k), key=ADV_ACTIVITIES.index))
            if rng.random() < 0.05:
                acts += ";"
        note = rng.choice(ADV_NOTES).format(g=g.split()[0], k=rng.randrange(1, 9)) if rng.random() < 0.8 else ""
        if i == 7:
            note = "1 a" + " " * 3000 + "x"
        rows.append([g, rng.choice(ADV_CITIES), acts, str(rng.randrange(100, 999)), str(rng.randrange(1000, 9999)),
                     note, seats[i]])
    return rows


def main() -> None:
    _write(HERE / "free_text.csv", make_free_text(random.Random(20261004)), "utf-8", "\n")
    _write(HERE / "names_cp1252.csv", make_names_cp1252(random.Random(1252)), "cp1252", "\r\n")
    _write(HERE / "adversarial_crlf.csv", make_adversarial(random.Random(4242)), "utf-8", "\r\n")


if __name__ == "__main__":
    main()

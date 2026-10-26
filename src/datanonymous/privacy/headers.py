"""D34: column headers, sheet names, file names and folder names hold no personal data (AREA G).

Scripts depend on the mirrored paths and headers (D20), so the tool never renames anything. Every name the
run would write is screened instead: each column header (the CSV/TSV/TXT header row, Excel header cells,
Parquet/Feather field names, Stata variable names, JSON keys), each Excel sheet name, each file name and each
folder of the mirrored relative path; with ``--keep-titles`` also every title, notes and footer text written
verbatim. Values follow the D35 allowlist instead (``detect.text.copy_gate``: a value is kept only when it is
provably safe, otherwise pseudonymised); a name cannot be pseudonymised, because code references it, so it is
judged by its content: the checks below run on the name itself as written and decoded (URL escapes:
``Kaden%20Yazzie%20IEP``, ``Ana+Garcia``; K-10: OOXML ``_xHHHH_`` escapes, ``Ana_x0020_Garcia``, and HTML character
references, ``Ana&nbsp;Garcia``, ``Ana&#32;Garcia``: :func:`decoded`), and the person-name check also with digits
glued between two words read as a separator (``Ana1Garcia``: :func:`name_kinds`). A flagged name is redacted and
indexed by the leak scanner in those decoded forms too (:func:`_escaped_forms`).

D34 is a **best-effort layer** (frozen by the user on 2026-10-06 after three harden/attack passes): it keeps
the checks that are reliable (real IDs, personal-data shapes, explicit person layouts, known first + last name
pairs, the namesake and program allowlist, ``--keep-name``) and drops the low-precision name heuristics that
refused legitimate files and still leaked (a given name beside a word off the lists, single given names or
surnames among siblings, a surname folder with a given-name child). Unusual names (everyday words, names off
the lists, other scripts) in file, sheet or folder names or in one-column-per-student headers may pass: the
accepted residual risk the README documents.

1. **personal-data shapes**, by the copy gate's one shape test of values (``detect.text.unit_shapes``, R7-03):
   e-mail addresses, phone numbers, SSNs and street addresses, also inside a longer name (an underscore counts as
   a space), and a bare valid North American number standing alone (a JSON key, a file, sheet or folder named
   ``3125550187``). A URL alone is no personal data. Ten bare digits are a phone number after a phone or family
   word (``Ph 5551234567``, ``Parent_5125550199``); otherwise only when they are a valid North American number
   (area code and exchange starting with 2-9, no ``N11`` service code, no ``N9X`` area code) that is no time stamp
   (``roster_1696342522``, ``export_2023101512``) and does not follow a word of a numbered thing
   (:data:`_PHONE_NUMBERED_WORDS`: an export, job, record or survey number: ``Extract_5839201746``,
   ``Qualtrics_SV_5839201746``, ``PO 4500012345``); these exceptions only ever excuse a bare run of digits, never
   another shape (:func:`_phone_in_name`); nine digits after ``SSN`` are an SSN (K-12: also after ``SS#``, ``SS
   No``, ``Soc Sec`` and ``W2``/``W-2``; the 3-2-4 layout with any separators is an SSN by the shared shape test,
   ``123.45.6789``), and a local number ``NXX-XXXX`` after a phone or family word is a phone number (``Call
   486-1234``). K-19: three digit groups 3-3-4 joined only by spaces or underscores are excused like a bare run
   (``Lexile_800_900_1000``, ``Cut_100_200_3000``). B8: a 3-2-4 run of a number, a month and a year joined by
   spaces or underscores alone is no SSN without an SSN label (``Bus_Route_214_12_2024``: :func:`_ssn_in_name`), and a
   year or a school year beside fewer digits than a phone number holds is no phone number
   (``Grades_2024-25_612345``: :func:`_year_and_number`). R8-01, R8-02: the shapes are
   broad (an e-mail address in any script, a phone number in any national layout: ``020 7946 0958``,
   ``Parent_07911_123456``); a phone number in a non-North-American layout is excused only like a bare run (after a
   word of a numbered thing) or as a list of numbers of at most three digits (``Cut_200_250_300``);
2. **person-name content** (:func:`person_name_hit`), with the name lists of ``detect.names`` (Faker's
   given-name and surname lists) and the names they miss (``privacy.world_names``: newcomer communities,
   ``Hodan``, ``Warsame``, ``Oluwaseun``, ``Dmitri``; given names of recent birth years, ``Zyaire``,
   ``Jaxtyn``); everyday and header words never count as evidence:

   * a *name pair*: a known given name next to a known surname, either order, neither of them an everyday or
     header word, joined by ``_ - .`` or a space or written in camelCase (``Garcia_Ana``, ``Ana Garcia``,
     ``AnaGarcia``, ``smith-john``; ``John F Kennedy``: initials and particles in between; ``Ana Rose Garcia``:
     an everyday middle name between them; not ``Summer Davis``, ``Ana Price``, ``Ana Hall``, ``Oak Hill``,
     ``Tom Bean``). A two-letter East Asian surname that is no everyday word beside a known given name (``Le
     Minh``, ``Wei Li``) is a pair too;
   * a layout ordinary labels do not use: ``Smith, John`` / ``Okafor, Summer`` / ``Begay, Kaden`` / ``Adeyemi,
     Oluwaseun`` (surname, comma or semicolon, given name; the surname may be off the lists or also an
     institution word: ``Park, Jennifer``, ``Day, Doris``, ``St. Clair, Ana``; the given name may be off the
     lists after a distinctive surname, never a company or credential word: ``Smith, Inc.``), ``Smith, J.`` /
     ``Hall, J.`` / ``Garcia Lopez, M. A.`` (surname, comma, initials), ``J. Smith`` (initials with periods, then
     a name), ``Smith J.`` / ``Smith_J`` / ``SmithJ`` / ``J Smith`` (a distinctive surname and an initial;
     ``Garcia P1`` is a period label), ``Jayden M.`` / ``Jayden_M`` / ``JaydenM`` (a distinctive given name and
     an initial; not a Roman numeral: ``Elizabeth I``; ``Max T.``: a given name that is a header word, with a
     period, as the whole name), ``Rodriguez.M`` / ``garcia.a`` (a surname, a period and an initial), user names
     in lower or upper case (``jgarcia``, ``agarcia2``, ``j.garcia``, ``anagarcia``, ``smithj``,
     ``rodriguezm``, ``nguyent2``, ``JSMITH``, ``Roster_AGARCIA``, ``msgarcia``: an initial, a common given name
     or a title glued before or after a common surname, :func:`popular_names`) and titles: ``Mr. Smith`` /
     ``Ms. Garcia`` / ``Dr. Lee`` / ``Sr. Garcia`` /
     ``Senor Garcia`` / ``Profe Lopez`` / ``Herr Schmidt`` / ``Frau Mueller`` (a title, then a name; ``Mrs. G``,
     ``Mr. G's Class``: a title and a bare initial) and roles written like a title before a name on the lists
     (``Counselor Nwosu``, ``Principal Garcia``, ``Nurse Garcia``, ``Father Murphy``, ``Rabbi Cohen``; not
     ``Counselor Name``). A class named after its teacher (``Mr Smith Class``) names that teacher: it is flagged
     too. A given name and an initial followed by a capitalized word is a name with a middle initial
     (``Victor H Hexter``): judged with that word, so ``Victor H Hexter ES`` is a school;
   * a name in a script the Latin lists cannot read (:func:`_script_name_hit`): two Arabic name words (``محمد
     علي``), a Chinese surname and given-name characters (``张伟``), a Korean surname and given name
     (``김민준``). Cyrillic and Greek words are transliterated (``Иван Петров`` is ``Ivan Petrov``) and read
     with the Latin lists.

   Single words never count (``Grant``, ``Price``, ``Rose``, ``Hall``, ``Will``, ``Mark``, ``Chase``, ``Lane``,
   ``Reading``, ``May``), nor do header words (:data:`DATA_WORDS`, :data:`DOCUMENT_WORDS` and the tool's hint
   vocabularies: ``Max_Date``, ``First_Name``, ``Math_Score``), nor institution words (:data:`INSTITUTION_WORDS`:
   ``Smith Hall``, ``Kennedy Center``, ``Lincoln Campus``). A name is a place, a business or a public name only
   when it is followed by a school name form (``Jesse H Jones High School``, ``Lily B Clayton Elementary``,
   ``Sam Rayburn ECHS``, ``Dr. Phillips High School``, ``Gus Garcia YMLA``, AskTED's upper-case ``JACK YATES H
   S`` and ``VICTOR H HEXTER EL``: :data:`_SCHOOL_NOUNS`; not ``School Records``), also after one or two more
   capitalized words of a longer school name (``Kay Bailey Hutchison Elementary``, ``Bertha Sadler Means
   YWLA``), or a company word (``Patel Brothers``, ``Garcia LLC``), preceded by a place word, a clergy title or a
   school number (``Fort Worth``, ``St John``, ``San Jose``, ``Bishop Lynch``, ``PS 123 Mahalia Jackson``), when a
   US state follows it (``Allen, Texas``, ``Columbus, Georgia``, ``Sherman Texas``), or when it is a public name
   of ``privacy.namesakes``: namesakes of schools, programs and federal funds, authors,
   characters, scientists and historical figures of curriculum units (``Rosa Parks``, ``Sam Houston``, ``Robert
   E. Lee`` but not ``Robert Lee``, ``Harry Potter``, ``Atticus Finch``, ``Carl D. Perkins``, ``Miss Nelson Is
   Missing``), unless another name word stands right beside them (``Jose Marti Garcia``); eponyms of laws,
   structures, events and eras (``Charles's Law``, ``Lewis Dot Structures``, ``Ming Dynasty``; two name words
   only in the possessive or as hyphenated surnames: ``John Brown's Raid``, ``Hardy-Weinberg``); court cases
   (``Morse v. Frederick``); program, product and vendor names (``KIPP``, ``Green Dot``, ``Cal Grant``, ``John
   Deere``, ``Dell Latitude``; ``Sara Lee``, ``Edward Jones`` and ``Oscar Mayer`` only beside a business word or
   a number). Any other institution, business or eponym word beside a full name leaves it a person's name:
   ``Ana Garcia College Essay``, ``Garcia_Ana_Scholarship``, ``Liam Johnson Senior Exit``, ``Mr. Garcia Senior
   Seminar``, ``Ms. Okafor Room``, ``Room Ms Garcia``, ``Senior Garcia, Ana``, ``Kindergarten Chinedu Okafor``,
   ``Ana Garcia Dental Screening``, ``Ana Garcia Trial Placement``, ``Ana_Garcia_Day_1``. Cyrillic and Greek
   look-alike letters inside a Latin word are read as Latin (a Cyrillic ``a`` in ``Garcia``);
3. **equality with a real ID of the run** (:class:`IdKeys`), *chance-aware*: a part of the name (split at
   separators, letter/digit boundaries: ``score_S482913``, ``score482913``, ``S482913math``) equals a real ID of
   an ID namespace of the run, exactly or by its D8 key (case, punctuation, leading zeros: ``0482913``,
   ``s-482913``): the pivoted table with one column per student, a file or folder per student. Only IDs with at
   least 4 digits and no more letters than digits count (a lookup table's ``id`` codes ``MATH``/``READ`` never
   make words real IDs); row-number namespaces kept as 1..n (D9) are not secret. A match is weighed by the
   chance that an ordinary number equals a real ID (:meth:`IdKeys.matches`): the share of numbers of its length
   that are real IDs, for 4-digit numbers at least the share of the numbers around it (sequential student
   numbers 1001-1400 make nearly every number of that range a real ID: :meth:`IdKeys.number_chance`), and at
   most the share of numbers that are one of the IDs the named file, folder or sheet itself holds (a
   per-student ``IEP_496487.csv`` whose ID column holds student 496487 only). A letter-prefixed or zero-padded
   exact ID, or a match whose chance is below 1 in 1,000, flags the name alone; a weaker match (many IDs of that
   length, a number inside a longer name with fewer than 5 digits) needs repeated evidence: sibling names (one
   table's headers, one workbook's sheets, one folder's files and folders) that share the same text around
   distinct real IDs (``score482900``, ``score482901``: the pivot; ``iep/IEP_387347.csv`` and
   ``iep/IEP_405159.csv`` beside 50,000 six-digit IDs) where the chance that so many of the numbers in that text
   are real IDs is below 1 in 100 (``C(n, m)`` times the product of the ``m`` matches' chances, ``n`` the
   siblings with a number in that text), or three distinct IDs or more among all the numbers of the siblings
   with such a chance below 1 in a million (:func:`id_group_flags`: a hundred of 2,000 item headers
   ``Q17_482913`` equal to a real ID by chance are no evidence). K-01: a number labelled as a person's that equals a
   real ID flags the name alone, whatever the namespace's density and also when it reads as a year-month or a date
   (``Student_200105``, ``IEP_20210915``; a 4-digit year never): right after a person or document label (``Student
   612587``, ``Stu_612587``, ``IEP_512340``, ``Evaluation_513820``, ``504_512340``, ``ID 512340``, ``Student Number
   612587``, ``Student No. 612587``, ``Perm ID 612587``, ``Student, 612587``: :func:`person_label_before`), one word
   after a document label when it is no date or time (``IEP Meeting 612587``, ``504 Plan 612587``), or right before
   a person or document label (``612587_IEP_Goals``, ``612587 Student``: :func:`person_label_after`); B2: a label
   elsewhere in the name (``Student_Schedule_612714``, ``IEP_Progress_Report_612714``: :func:`labelled_name`) holds a
   real ID at a chance up to :data:`GROUP_CHANCE` (:data:`NEAR_LABEL_FACTOR`); B1: a school year is never the label's
   ID (``IEP_Caseload_2425``, ``2425_Student_Roster``, ``IEP_202425``), and a date or time as people write them in names
   (:func:`name_date`) is the label's ID only right after it (``Student_160816``) or when few dates of its layout
   are real IDs (:meth:`IdKeys.date_share`) and its siblings are not all dates (a folder of daily exports
   ``101524_Student_Attendance.csv``); a sibling named
   by a number alone under a folder or workbook whose name holds such a label (``iep/1053.csv``,
   ``iep_progress.xlsx``: :func:`person_context`, :func:`id_group_flags`) does too. D37 (user, 2026-10-11): two or
   more sibling names that share the same text around real IDs (numbers alone, ``score1007``, ``_1007``,
   ``1007_score``; initials around them, ``FK (1004)``; three or more with words around them that differ, surnames:
   ``Garcia (1004)``) of ONE namespace or of the run's IDs together are a pivot by
   person whatever the namespace's density (``Skill,1180,1349,...`` beside 3,500 students numbered 1001-4500, where
   each number alone is no evidence; a pair of columns), and a few numbers that are no such ID (at most one, or a
   quarter of the set: withdrawn students) never excuse the rest (:func:`pivot_flags`, :func:`mostly_real_ids`).
   Excused: numbers that all read as dates, school years or clock times (also on five-minute marks such as lunch
   periods ``1235,1305,1330``), numbers counting up in one constant step, numbered things, and a numbered parent
   where chance explains them (each number weighed by at least the share of real IDs around it, A3:
   :meth:`IdKeys.local_share`). A2 (K-18): a zero-padded exact ID is strong on its own (the padding
   is the ID's: ``Goals_059295``, ``caseload/024891/``) unless the run's names hold zero-padded numbers of its
   width that are no real ID (:meth:`IdKeys.note_run_names`: a folder of county-district numbers ``015901``,
   ``015907``, ``015915`` beside 30,000 zero-padded IDs) and no sibling contradicts it (:func:`id_group_flags`);
   then it is weighed by the share of zero-padded numbers of its width that are real IDs
   (:meth:`IdKeys.padded_chance`). A number that reads as a time stamp (``HHMMSS``), a short date
   (``YYMMDD``, ``MMDDYY``, ``MMDDYYYY``, ``MMDD``: ``Attendance_1001``) or a numbered thing (``Room_10101``,
   ``Course 10199``, ``CIP_150301``, ``Report_153017``, ``Pct_151200_to_152000``: :data:`NUMBERED_WORDS` before
   it) is no evidence on its own, and nor is a name that is a number alone under a parent whose name says what
   its children's numbers are (``enrollment_by_zip.xlsx`` with a sheet per ZIP code, ``rooms/``:
   :func:`numbered_context`). A1: such an excuse gives way to its siblings' evidence: numbers after a word of a
   numbered thing that siblings repeat around distinct real IDs (``Progress_Report_612714``, ``Lunch_Account_#``,
   ``Case_#_Notes`` beside 6,000 six-digit IDs) are weighed by their chance before the excuse (at least the share
   of real IDs around them, at the limit of 1 in a million), and a time or short date by its chance when not every
   sibling number reads so (:func:`id_group_flags`). A8: real IDs that the run's names hold too often for chance,
   counted over the whole run by digit count (per-student documents one per folder: ``Transcript_373208``,
   ``BIP_737720``, ``Cumulative_Folder_169858``: :meth:`IdKeys.note_run_names`), are strong wherever they appear;
   one such document alone among thousands of IDs of its length stays chance (the README's residual).
   Never taken for IDs: years (1900-2100), school years (``201920``, ``20192020``), year-months and dates
   (``202309``, ``20230915``), parts shorter than 4 characters; a whole name with separators is compared
   exactly only, so the date header ``9/3/2024`` is never the ID ``932024``.

Sibling names (one table's headers, one workbook's sheets, one folder's files and folders) are screened together
for repeated real IDs only (check 3). Single words among siblings (one column, sheet, file or folder per student
by given name or surname: ``Aaliyah``, ``Brayden``, ``Okafor``) and a surname folder with a given-name child
(``Garcia/Ana/``) are not screened: the residual risk of the best-effort layer. A file name is split from its
extension only at a known extension (``io.registry.known_extension``): ``Ana.Garcia`` is screened whole. An
Office owner or lock file (``~$rcia_Ana_IEP.docx``, ``.~lock.Garcia_Ana_IEP.xlsx#``) or a macOS AppleDouble file
(``._Garcia_Ana_IEP.csv``) is skipped by discovery and flagged with the document whose name it holds (Word drops
the first two characters of a long name: :func:`lock_file_rest`), or when that name holds any distinctive name
word (``~$Okafor Summer.docx``: it is never written, only listed). With ``--keep-titles``, a title or notes text
written verbatim is screened like a header, clause by clause (:func:`_verbatim_kinds`).

``--keep-name TEXT`` (repeatable; config ``keep_name``) keeps a name that was flagged but is not personal data
(an exact header, sheet, file or folder name; a file name with or without its extension).

What a flag does
================
A flagged name that the output would hold FAILs its file (D19: not written, exit code 3) with a fix: rename
the column, sheet, file or folder in the source or reshape the table (one row per person), or skip the file
with ``--exclude-path GLOB``. ``--exclude COL`` matches a column by its name only, never by position, so the
fix says rename. A flagged folder fails every file under it. A flagged name that is never written (an
``--exclude`` column, a sheet that holds no table, a skipped or ``--exclude-path`` file) FAILs nothing but is
redacted all the same (also in the manifest's settings, where the user typed it). So are the names the run
prints but never writes: the folders of the input, out-dir, report-dir and ``--config`` paths (the input root
folder, the parent folders of a single-file input), the column, sheet and range references typed in the settings
(also a near miss of a flagged name: ``Okonkwo Chinedu`` for ``Okonkwo, Chinedu``, and a glob naming part of
one: ``*Okafor*``), and the names quoted in read messages of files that hold no table (the typed name of ``no
sheet named 'Data'``, which counts the workbook's sheets but never lists them; a header-row check, an unsupported
column). A usage error is screened before it is printed,
also for the values typed on the command line that it repeats unquoted and for the folders of a typed path in
any printed form (:func:`redact_message`); a YAML error of ``--config`` names the line and column, never the
line itself (``config._yaml_problem``). Stata variable
labels and data labels, Excel named ranges and table names are never written (``io.statio``,
``io.excel_book``), so they are not screened.

The flagged text is never printed. It is shown as ``[redacted-N]`` (one number per distinct name, numbered
in sorted path order, then by file, sheet and column), and the FAIL reason locates it without quoting it:
the column number (``column 7 of 12``), the sheet number (``sheet 2 of 3``), the file's place in its folder
(``file 3 of 5 in students/, sorted by name ignoring case; .csv, 1,234 bytes``) or the folder's place in its
parent folder. Concretely:

* a file whose path holds a flagged folder or file name keeps it only privately (``FileResult.out_rel_real``,
  used to find an earlier copy in the out dir); its ``rel_path`` / ``out_rel_path`` (report keys and paths,
  manifest, console, stale-output notes, README) and the paths of skipped files show the redacted form;
* a FAILed file's details are withheld from the report and the console (``FileResult.withheld``): its
  column table, sheet names and layout are left out (they would quote the names);
* a run-level info message (:class:`NameScreenMessage`) carries the PRIVATE redactor (:class:`NameScreen`),
  which ``report.report`` applies to the whole report, the console summary, ``inspect`` and ``--json`` output
  and ``-v`` tracebacks, and ``report.docs`` to the out-dir docs: every remaining occurrence of a flagged name
  (also in markdown-escaped, quoted, JSON-escaped, URL-encoded and console-escaped forms, any case) becomes
  its ``[redacted-N]``; a usage error quoting a name is screened before it is printed (:func:`redact_message`);
* the leak scanner (``privacy.leaks``) indexes every flagged name as sensitive text (:meth:`Flag.terms`: the
  whole name, its word sequence, its person-name words and the real ID it holds, never its other words, so a
  clean file that shares a word such as ``Reading`` or ``Gradebook`` is unaffected): it is never excused as a
  public header, and a written output whose re-read header or sheet name holds one FAILs (a bug backstop). A
  written name that holds a flagged name is flagged too, by the same matcher (:meth:`NameScreen.repeats`, K-16: the
  scanner's token sequences, so ``Jack_Yates_HS_Enrollment.csv`` beside a flagged sheet ``Jack Yates`` gets the D34
  verdict and its ``--keep-name`` fix; A4: a person's name words also in the other order, ``Mark_Patel_Walkthrough``
  beside a folder ``Patel, Mark``; B3: also its first given name and surname alone, ``Summer_Davis`` beside
  ``Davis, Summer Joy`` and ``Mark_Patel`` beside ``Mark A. Patel``: :func:`_given_surname_pairs`,
  :func:`_with_given_name`; A5: also a printed folder of the input, out-dir, report-dir or config path, screened
  before the written names are, ``Davis_Summer_Reading_Log.csv`` in an input folder ``Davis, Summer``); a flag that
  holds a real ID only is repeated by that ID where it is strong
  evidence against the flagged IDs alone (P-10: ``612587_Notes`` beside a flagged ``Student 612587``; never a number of
  a dense pivot, a date or time such as an attendance sheet ``1007``, or a number alone whose own parent says what its
  numbers are, K-18: a sheet ``30303`` of ``enrollment_by_zip.xlsx``). P-10: a real ID that is a plain number is no
  never-excused text of the scanner (it judges real IDs by its ID rules: an exact maximum ``1375`` of another file is
  a statistic), and a flagged name that is a number alone is redacted only where it is a name (in a path or quoted),
  never where the run shows the same number as an unflagged name (:func:`_public_numbers`);
  the out-dir paths it prints (outputs left from an earlier run) are screened the same way
  (:meth:`NameScreen.display_path`);
* the name words of a flagged person name join the run-wide real-name index (``generate.pii.plan_names``),
  so no generated fake name spells them.
"""

from __future__ import annotations

import ast
import bisect
import datetime as _dt
import html
import json
import math
import os
import re
import unicodedata
import urllib.parse
import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .. import hints
from ..detect import names as name_lists
from ..detect.ids import normalize_id
from ..detect.text import (EMB_BARE_PHONE_RE, EMB_INTL_PHONE_RE, EMB_PHONE_RE, EMB_SSN_RE, PHONE_RUN_MIN, nanp_digits,
                           phone_runs, trunk_zero, unit_shapes)
from ..model import FileResult, Message
from . import namesakes, world_names

D34_PREFIX = "D34: "
PLACEHOLDER = "[redacted-{}]"
#: What a flag kind is called in messages (never the value).
KIND_TEXT: dict[str, str] = {
    "name": "a person's name", "email": "an e-mail address", "url": "a URL", "phone": "a phone number",
    "ssn": "an SSN", "address": "a street address", "id": "a real ID of this run",
}
_KIND_ORDER = ("name", "id", "email", "phone", "ssn", "address", "url")
#: Personal-data shapes that flag a name (a URL alone names no person).
_SHAPE_KINDS = frozenset({"email", "phone", "ssn", "address"})

# --------------------------------------------------------------------------- vocabularies
#: Words of school, building, place and institution names: a person's name right before or after one names
#: the place (``Lincoln High``, ``Martin Luther King Jr Elementary``, ``Smith Hall``, ``Kennedy Center``).
#: Kept narrow on purpose: ``Course``, ``Class``, ``Homeroom``, ``Team``, ``Report`` are not here, so
#: ``Garcia_Ana_Course_Grades`` and ``Mr Smith Class`` still name a person. These words are never name words; a
#: full name before one is still a person unless it is a school name form (:data:`_SCHOOL_NOUNS`: ``Ana Garcia
#: College Essay``, ``Ana_Garcia_Day_1``). ``surname``: also a common surname, an everyday word (``Hall``,
#: ``Park``: ``Ana Hall`` is no name pair); ``weak``: a word of dates and fields (``Day``, ``Field``).
#: One word list of more than 50 entries, so the leak scanner treats it as data, never as text the tool writes
#: (``privacy.leaks.literal_texts``).
_INSTITUTIONS: dict[str, str] = {
    "academies": "place", "academy": "place", "airport": "place", "arena": "place", "auditorium": "place",
    "ave": "place", "avenue": "place", "award": "place", "awards": "place", "bldg": "place", "blvd": "place",
    "borough": "place", "boulevard": "place", "bridge": "place", "building": "place", "campus": "place",
    "cathedral": "place", "center": "place", "centre": "place", "chapel": "place", "charter": "place",
    "church": "place", "city": "place", "clinic": "place", "college": "place", "conservatory": "place",
    "county": "place", "cusd": "place", "day": "weak", "district": "place", "dorm": "place", "dormitory": "place",
    "drive": "place", "elem": "place", "elementary": "place", "es": "place", "field": "weak", "fields": "weak",
    "foundation": "place", "fund": "place", "gym": "place", "gymnasium": "place", "hall": "surname",
    "halls": "place", "high": "place", "highway": "place", "holiday": "weak", "hospital": "place",
    "house": "place", "hs": "place", "hwy": "place", "institute": "place", "intermediate": "place", "isd": "place",
    "island": "place", "junior": "place", "kindergarten": "place", "lake": "place", "library": "place",
    "lodge": "place", "magnet": "place", "memorial": "place", "middle": "place", "montessori": "place",
    "ms": "place", "museum": "place", "parish": "place", "park": "surname", "parkway": "place", "plaza": "place",
    "polytechnic": "place", "port": "place", "prek": "place", "prep": "place", "preparatory": "place",
    "preschool": "place", "primary": "place", "prize": "place", "residence": "place", "river": "place",
    "rm": "place", "road": "place", "room": "place", "scholarship": "place", "scholarships": "place",
    "school": "place", "schools": "place", "secondary": "place", "seminary": "place", "senior": "place",
    "square": "place", "stadium": "place", "station": "place", "street": "place", "theater": "place",
    "theatre": "place", "town": "place", "township": "place", "trust": "place", "univ": "place",
    "university": "place", "usd": "place", "village": "place", "way": "place",
    # clergy titles of Catholic school namesakes: a place before a name ('Bishop Lynch', 'Cardinal Newman'), a
    # name word anywhere else ('Ana Bishop')
    "bishop": "clergy", "archbishop": "clergy", "cardinal": "clergy", "monsignor": "clergy", "msgr": "clergy",
    "pope": "clergy", "blessed": "clergy"
}
INSTITUTION_WORDS: frozenset[str] = frozenset(w for w, kind in _INSTITUTIONS.items() if kind != "clergy")
#: Clergy titles before a name that make it a place (``Bishop Lynch``, like ``Saint``).
CHURCH_TITLES: frozenset[str] = frozenset(w for w, kind in _INSTITUTIONS.items() if kind == "clergy")
#: Header and data words that are never name evidence in a header, although some are on the name lists
#: (``Max_Date``, ``Min_Score``, ``Math_Mark``, ``ELA``, ``Delta``, ``Gross_Pay``, ``Fee_Paid``, ``Nick_Name``).
#: One word list of more than 50 entries (data, never text the tool writes).
DATA_WORDS: frozenset[str] = frozenset({
    "absences", "absent", "act", "actual", "address", "advisory", "age", "algebra", "amount", "answer", "answers",
    "ap", "art", "arts", "attempted", "attendance", "average", "avg", "balance", "band", "baseline", "begin",
    "behavior", "benchmark", "biology", "birth", "birthdate", "bottom", "budget", "calculus", "cell", "change",
    "chapter", "chemistry", "choice", "choices", "choir", "city", "class", "cnt", "code", "codes", "contact",
    "cost", "council", "count", "counts", "course", "cousin", "credit", "credits", "current", "date", "dates",
    "day", "days", "dean", "delta", "diff", "discipline", "dob", "domain", "due", "earned", "ela", "email",
    "emergency", "end", "english", "enrolled", "enrollment", "entry", "ethnicity", "exam", "exams", "excused",
    "exit", "expected", "fail", "failed", "fall", "false", "family", "fee", "fees", "final", "finals", "first",
    "flag", "flags", "fluency", "foster", "french", "full", "fy", "gain", "gender", "geometry", "german", "given",
    "goal", "gpa", "gr", "grade", "grades", "gross", "growth", "health", "high", "history", "home", "homeroom",
    "hours", "hr", "hrs", "ib", "id", "ids", "incident", "index", "infant", "initial", "initials", "item", "items",
    "key", "keys", "kind", "last", "left", "legal", "lesson", "level", "levels", "lexile", "list", "lists", "long",
    "loss", "low", "lunch", "mail", "mark", "marked", "marking", "marks", "mastery", "math", "mathematics", "maths",
    "max", "maximum", "meal", "mean", "median", "mid", "middle", "midterm", "min", "minimum", "mins", "minutes",
    "mobile", "mode", "module", "month", "months", "mtd", "music", "na", "name", "names", "nbr", "net", "new",
    "next", "nick", "nickname", "no", "none", "num", "number", "objective", "old", "opt", "option", "options",
    "oral", "other", "outcome", "paid", "pass", "passed", "pay", "pct", "pe", "percent", "percentage", "percentile",
    "period", "periods", "phone", "phonics", "physics", "plan", "points", "post", "pre", "predicted", "preferred",
    "present", "prev", "previous", "price", "proctor", "proficiency", "program", "projected", "pts", "qtd",
    "quarter", "quarters", "question", "questions", "quiz", "quizzes", "race", "rank", "ranking", "rate", "rates",
    "ratio", "raw", "read", "reading", "referral", "response", "responses", "result", "results", "right", "roll",
    "salary", "sat", "scale", "scaled", "science", "score", "scores", "sd", "seconds", "secs", "section",
    "semester", "semesters", "service", "session", "sessions", "sex", "short", "skill", "spanish", "spring",
    "standard", "start", "state", "status", "std", "stdev", "strand", "sub", "subject", "subtotal", "sum", "summer",
    "suspension", "sy", "tardies", "tardy", "target", "term", "terms", "test", "tests", "tier", "time", "times",
    "toddler", "top", "total", "totals", "trimester", "true", "type", "unexcused", "unit", "units", "unknown",
    "value", "values", "var", "variance", "vocabulary", "wage", "week", "weeks", "weight", "weighted", "winter",
    "withdrawal", "withdrawn", "work", "write", "writing", "year", "years", "yes", "yr", "yrs", "ytd", "zip"
})
#: Words of the documents a school keeps about a person (``Kaden Yazzie IEP``, ``Ana Garcia Dental Screening``,
#: ``Aiden Quarshie Goals``): header words, never name evidence, so a name beside one is judged on its own
#: (plurals are read too). One word list of more than 50 entries (data, never text the tool writes).
DOCUMENT_WORDS: frozenset[str] = frozenset({
    "iep", "essay", "seminar", "placement", "screening", "medication", "rubric", "portfolio", "journal", "log",
    "card", "record", "transcript", "letter", "packet", "worksheet", "homework", "evaluation", "eval",
    "progress", "roster", "gradebook", "schedule", "contract", "waiver", "physical", "immunization", "allergy",
    "allergies", "visit", "checklist", "observation", "interview", "survey", "profile", "summary", "data",
    "tracker", "tracking", "reflection", "sample", "scan", "photo", "picture", "badge", "accommodation",
    "modification", "meeting", "agenda", "notes", "receipt", "release", "detention", "transfer", "registration",
    "recommendation", "reference", "resume", "certificate", "diploma", "application", "review", "folder",
    "binder", "conference", "objective", "probe", "intervention", "consent", "request", "form", "copy", "draft",
    "signed", "scanned", "updated", "backup", "archive", "export", "upload", "download", "document", "docs",
    "info", "information", "details", "history", "timeline", "contacts", "file", "files", "case", "caseload",
    "behavior", "fba", "bip", "ard", "arc", "sst", "rti", "ell", "esl", "lep", "sped", "gt", "dyslexia",
    "speech", "therapy", "counseling", "nurse", "health", "medical", "clinic", "exit", "entry", "plan",
    "goals", "minutes", "report", "reports", "status", "results", "scores", "grades", "work", "assignments",
})
#: Everyday words that are also surnames and label things with a letter (``Block_A``, ``Bell_B``, ``Ward_C``,
#: ``Lane_4``): never the surname of a ``Smith_J`` layout. One word list of more than 50 entries (data).
_EVERYDAY_SURNAMES: frozenset[str] = frozenset({
    "block", "bell", "ward", "lane", "wing", "gate", "bay", "band", "cell", "track", "pod", "wave", "tier", "team",
    "star", "gold", "rock", "stone", "wood", "lake", "hill", "dale", "bridge", "bank", "cross", "fort", "green",
    "brown", "white", "black", "gray", "grey", "rose", "hope", "joy", "may", "june", "case", "page", "post", "rice",
    "bean", "lamb", "fish", "bird", "fox", "wolf", "bear", "hawk", "crane", "swan", "drake", "duck", "nash",
    "booth", "camp", "court", "hall", "house", "church", "park", "field", "day", "king", "knight", "bishop",
    "dean", "price", "cost", "rich", "young", "strong", "short", "long", "small", "little", "large", "best",
    "good", "west", "east", "north", "south", "ford", "mill", "mills", "pond", "marsh", "moss", "reed", "rush",
    "snow", "frost", "storm", "rain", "summers", "winters", "major", "minor", "cook", "baker", "mason", "turner",
    "bond", "love", "sharp", "head", "foot", "hand", "arm", "bone", "skin", "blood", "heart", "lock", "key",
    "chase", "grant", "will", "mark", "max", "sage", "early", "late", "noble", "power", "gage", "gauge", "plan",
    "ray", "ring", "polo", "beam", "moon", "sun", "sky", "rivers", "waters", "woods", "wells", "banks", "fields",
})
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_PARTICLES = frozenset({"de", "del", "della", "la", "le", "van", "von", "der", "den", "da", "di", "du", "dos", "das",
                        "bin", "ibn", "al", "y", "mc", "mac", "st"})
#: Titles before a name (``detect.names`` titles, ``coach``, Spanish, French and German honorifics: ``Sra.``,
#: ``Senor``, ``Profe``, ``Mtra.``, ``Herr``, ``Frau``; ``Sr.`` is a title before a name and a suffix after one);
#: those that also abbreviate ordinary header words (``Prof`` iciency, ``Rev`` enue, ``Fr`` iday, ``Miss`` ed,
#: ``Coach``, ``Dr`` ive, ``Maestro``) need a name on the lists that is no everyday word after them.
_TITLES = frozenset({"mr", "mrs", "ms", "miss", "mx", "dr", "prof", "rev", "fr", "sra", "srta", "mme", "mlle",
                     "coach", "senor", "senora", "senorita", "profe", "profa", "profesor", "profesora", "maestra",
                     "maestro", "mtra", "mtro", "herr", "frau", "fraulein", "monsieur", "madame", "mademoiselle",
                     "mister", "missus"})
_AMBIGUOUS_TITLES = frozenset({"prof", "rev", "fr", "miss", "coach", "dr", "profe", "profa", "profesor", "profesora",
                               "maestra", "maestro", "mister", "missus", "madame", "monsieur"})
#: Roles written before a name like a title (``Counselor Nwosu``, ``Principal Garcia``, ``Nurse Garcia``,
#: ``Father Murphy``, ``Rabbi Cohen``, ``Imam Ali``): a title only before a name on the lists that is no everyday
#: word (``Counselor Name``, ``Nurse Visits``, ``Principal Amount`` and ``Father Name`` stay header words).
_ROLE_TITLES = frozenset({"counselor", "counsellor", "nurse", "principal", "officer", "deputy", "sergeant", "sgt",
                          "father", "brother", "sister", "rabbi", "imam", "pastor", "deacon", "reverend", "chaplain",
                          "sheriff", "teacher", "tutor", "advisor", "adviser", "librarian", "registrar",
                          "superintendent", "supt", "dean", "coordinator", "director", "assistant", "asst", "aide",
                          "para", "paraprofessional", "custodian", "secretary", "trainer", "therapist", "manager"})
#: Honorifics that name a person even before a bare initial (``Mrs. G``, ``Mr. G's Class``).
_INITIAL_TITLES = frozenset({"mr", "mrs", "ms", "mx", "sra", "srta", "senor", "senora", "senorita", "herr", "frau",
                             "mme", "mlle"})
#: Company and credential words after ``Surname,`` (``Smith, Inc.``, ``Garcia, PhD``): no given name.
_COMPANY_WORDS = frozenset({"inc", "llc", "ltd", "corp", "co", "company", "pllc", "llp", "plc", "gmbh", "esq",
                            "phd", "edd", "md", "rn", "lpn", "cpa", "jd", "dds", "dvm", "lcsw", "lpc", "bcba"})
#: Place words before a name (``Fort Worth``, ``St John``, ``San Jose``, ``Mount Vernon``: ``detect.names``).
_PLACE_WORDS = frozenset(w for w, kind in name_lists._VALUE_WORDS.items() if kind == "place") - {"st"}
#: Endings of everyday English words (``Learning``, ``Assessment``, ``Readiness``): an unknown word with one is
#: no surname beside a given name (``Amira_Learning``).
_EVERYDAY_ENDINGS = ("ing", "tion", "sion", "ment", "ness", "ity", "ics", "ence", "ance", "ive", "ous", "ful",
                     "less", "able", "ible", "ship", "hood", "ism", "ist", "ure", "ary", "ery", "ory", "ize", "ise",
                     "ogy", "ware", "book", "work", "link", "lab", "labs", "net", "hub", "box", "kit", "base",
                     "data", "soft", "tech", "logic", "learn", "point", "path", "smart", "zone", "plus")


@lru_cache(maxsize=1)
def header_words() -> frozenset[str]:
    """Words that are never name evidence in a header: :data:`DATA_WORDS` and the tool's own header
    vocabularies (``hints``: person-role, ID, org-code, non-person, name-part, phone and contact words, the
    localized header words, and the words of things named after people or places that are no institution
    words: ``Course``, ``Month``, ``Program``)."""
    words: set[str] = set(DATA_WORDS) | DOCUMENT_WORDS
    for group in (hints.PERSON_ROLE_WORDS, hints.PERSON_ID_WORDS, hints.ID_WORDS, hints.ORG_CODE_WORDS,
                  hints.ORG_CODE_SUFFIXES, hints.NON_PERSON_NOUNS, hints.NAME_PART_QUALIFIERS,
                  hints.PHONE_NUMBER_WORDS, hints.PHONE_LINE_WORDS, hints.PHONE_SHORT_WORDS,
                  hints.CONTACT_QUALIFIERS, hints.PHONE_ATTRIBUTE_WORDS, hints.localized_header_words(),
                  hints.NAME_LIKE_THING_WORDS):
        words.update(group)
    return frozenset(words - INSTITUTION_WORDS)


@lru_cache(maxsize=2)
def us_states(given_names_too: bool = False) -> frozenset[str]:
    """One-word US state names that are no common given name (``Texas``, ``Nevada``, ``Ohio``; not ``Georgia``
    or ``Virginia``): a city before one is a place (``Allen, Texas``, ``Sherman Texas``). ``given_names_too``:
    every one-word state name (after a comma, ``Columbus, Georgia`` is a place too)."""
    prov = name_lists._provider("address.en_US")
    states = {name_lists.fold_word(x) for x in getattr(prov, "states", ()) if " " not in x} if prov else set()
    return frozenset(states if given_names_too else states - popular_names()[0])


@lru_cache(maxsize=1)
def known_names() -> tuple[frozenset[str], frozenset[str]]:
    """(given names, surnames) the screen reads: ``detect.names.known_names`` plus the names of US school
    populations Faker's lists miss (``privacy.world_names``)."""
    first, last = name_lists.known_names()
    return first | world_names.GIVEN_NAMES, last | world_names.SURNAMES


@lru_cache(maxsize=1)
def popular_names() -> tuple[frozenset[str], dict[str, int]]:
    """(given names, surname -> frequency rank) common enough to read inside a run-together lower-case word
    (``jgarcia``, ``anagarcia``): Faker's US lists of the most common given names and the 1,000 most common
    surnames (ranked by frequency), plus ``privacy.world_names`` (ranked last). The full lists hold short and
    rare names (``Omma``, ``Tata``, ``Reis``) that end ordinary words (``comma``, ``stata``, ``preis``)."""
    first: set[str] = set(world_names.GIVEN_NAMES)
    last: dict[str, int] = {}
    prov = name_lists._provider("person.en_US")
    if prov is not None:
        for attr in ("first_names_female", "first_names_male"):
            first |= {name_lists.fold_word(w) for w in getattr(prov, attr, ())}
        for rank, w in enumerate(getattr(prov, "last_names", ())):
            last.setdefault(name_lists.fold_word(w), rank)
    for w in world_names.SURNAMES:
        last.setdefault(w, 1000)
    return frozenset(w for w in first if len(w) >= 3), {w: r for w, r in last.items() if len(w) >= 4}


#: Ordinary English words shaped like an initial glued to a common surname (``fallen``: f + Allen, ``smiles``:
#: s + Miles), never a user name. One literal collection of more than 50 entries (data).
_USERNAME_LOOKALIKES: frozenset[str] = frozenset({
    "across", "along", "arose", "award", "aware", "breed", "brush", "chart", "cross", "drivers", "fallen",
    "freed", "glove", "gross", "layers", "legal", "marcher", "plane", "price", "shall", "sharding", "shines",
    "smiles", "space", "spark", "speck", "state", "styler", "thorn", "whale", "sharper", "chicks", "madams",
    "sparks", "swells", "shunt", "chanson", "ahead", "amass", "ashore", "aside", "atone", "bright", "crust",
    "flash", "flight", "glass", "gloss", "grain", "prose", "shale", "shook", "spike", "spine", "sport", "stack",
    "stall", "swing", "trust", "wheat", "whole", "brook", "clamp", "crook", "dwell", "plain", "shard", "spain",
    "players", "prayers", "slayers", "blowing", "glowing", "flowers", "showers", "towers", "smithy", "moored",
    "younger", "sharpen", "whitey", "cooked", "hilly", "browned", "greener", "kingly", "parkway",
})


def _username_surname(rest: str, popular_last: dict[str, int]) -> bool:
    """``rest`` is a surname common enough to read at the end of a user name (``garcia`` of ``jgarcia``): one
    of the 150 most common US surnames (4 letters), of the 300 most common (5 letters or more), or a surname of
    ``privacy.world_names`` (5 letters or more); never an everyday word (``hall``, ``ward``, ``rose``)."""
    rank = popular_last.get(rest)
    if rank is None or not _distinctive(rest) or rest in _EVERYDAY_SURNAMES:
        return False
    if rank >= 1000:
        return len(rest) >= 5
    return rank < (150 if len(rest) == 4 else 300)


# --------------------------------------------------------------------------- tokens
_WORD_RE = re.compile(r"[A-Za-z]+(?:['\u2019][A-Za-z]+)*|[0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_POSSESSIVE = re.compile(r"['\u2019]s$", re.I)
#: camelCase pieces joined to the next piece when the whole is a known name (``McDonald``, ``DeShawn``).
_JOINING = _PARTICLES | {"o"}


#: Cyrillic and Greek letters that look like Latin ones (``Garcia`` with a Cyrillic ``a``), and Latin letters
#: that accent folding leaves alone (O with stroke, L with stroke, sharp s, AE, thorn, D with stroke): screened as
#: their Latin look-alikes.
_CONFUSABLES = str.maketrans({
    "\u0430": "a", "\u0410": "A", "\u0435": "e", "\u0415": "E", "\u043e": "o", "\u041e": "O", "\u0440": "p",
    "\u0420": "P", "\u0441": "c", "\u0421": "C", "\u0443": "y", "\u0423": "Y", "\u0445": "x", "\u0425": "X",
    "\u0456": "i", "\u0406": "I", "\u0458": "j", "\u0408": "J", "\u0455": "s", "\u0405": "S", "\u0412": "B",
    "\u041d": "H", "\u041a": "K", "\u041c": "M", "\u0422": "T", "\u043a": "k", "\u043c": "m", "\u0442": "t",
    "\u043d": "h", "\u0432": "b", "\u0261": "g", "\u03bf": "o", "\u039f": "O", "\u03b1": "a",
    "\u0391": "A", "\u03b5": "e", "\u0395": "E", "\u03b9": "i", "\u0399": "I", "\u03ba": "k", "\u039a": "K",
    "\u03bd": "v", "\u039d": "N", "\u03c1": "p", "\u03a1": "P", "\u03c4": "t", "\u03a4": "T", "\u03c5": "u",
    "\u03a5": "Y", "\u03c7": "x", "\u03a7": "X", "\u0392": "B", "\u0397": "H", "\u039c": "M", "\u0396": "Z",
    "\u00d8": "O", "\u00f8": "o", "\u0141": "L", "\u0142": "l", "\u00df": "ss", "\u00c6": "AE", "\u00e6": "ae",
    "\u00de": "Th", "\u00fe": "th", "\u0110": "D", "\u0111": "d", "\u0152": "OE", "\u0153": "oe",
})


def clean_text(text: str) -> str:
    """``text`` as screened: invisible format characters (zero-width spaces and joiners, soft hyphens, bidi
    marks) removed, words written in Cyrillic or Greek alone transliterated (``Иван Петров`` is ``Ivan Petrov``),
    Cyrillic/Greek look-alike letters inside Latin words made Latin (a Cyrillic ``а`` in ``Gаrcia``), accents and
    ligatures folded, full-width forms made ASCII (``hints.fold_accents``)."""
    if not text.isascii():
        text = unicodedata.normalize("NFC", "".join(ch for ch in text if unicodedata.category(ch) != "Cf"))
        text = _transliterate(text).translate(_CONFUSABLES)
    return hints.fold_accents(text)


@dataclass(frozen=True)
class _Tok:
    raw: str  # as written (accents folded)
    key: str  # lower-case ASCII letters (``detect.names.fold_word``) or the digits
    start: int
    end: int


def _tokens(s: str) -> list[_Tok]:
    """Word tokens of an accent-folded name: letter runs (camelCase split; ``Mc``/``Mac``/``De`` + a capital
    joined when the whole is a known name) and digit runs."""
    first, last = known_names()
    out: list[_Tok] = []
    for m in _WORD_RE.finditer(s):
        w = m.group()
        if w.isdigit():
            out.append(_Tok(w, w, m.start(), m.end()))
            continue
        pieces: list[tuple[str, int]] = []
        pos = m.start()
        for part in _CAMEL_RE.split(w):
            if part:
                pieces.append((part, pos))
                pos += len(part)
        k = 0
        while k < len(pieces):
            part, start = pieces[k]
            if k + 1 < len(pieces):
                joined = name_lists.fold_word(part + pieces[k + 1][0])
                if name_lists.fold_word(part) in _JOINING and (joined in first or joined in last):
                    part = part + pieces[k + 1][0]
                    k += 1
            raw = _POSSESSIVE.sub("", part) if len(part) > 2 else part
            out.append(_Tok(part, name_lists.fold_word(raw), start, start + len(part)))
            k += 1
    return out


def _is_capital(tok: _Tok, case_info: bool) -> bool:
    return not case_info or tok.raw[:1].isupper()


def _distinctive(key: str) -> bool:
    return len(key) > 2 and key not in name_lists.common_words()


def _token_kinds(toks: list[_Tok], s: str = "") -> list[str]:
    """The kind of each token of a name (``s``: the cleaned text the tokens were read from)."""
    first, last = known_names()
    words = header_words()
    out = []
    for t in toks:
        k = t.key
        if t.raw.isdigit():
            out.append("digit")
        elif len(k) == 1:
            out.append("initial")
        elif k in _SUFFIXES:
            out.append("suffix")
        elif k in _PARTICLES:
            out.append("particle")
        elif k in _TITLES or k in _ROLE_TITLES:
            out.append("title")
        elif k in INSTITUTION_WORDS:
            out.append("inst")
        elif k in _PLACE_WORDS:
            out.append("place")
        elif k in namesakes.PROGRAM_WORDS:
            out.append("program")
        elif k in words or (len(k) > 4 and k.endswith("s") and (
                k[:-1] in words or (k.endswith("es") and k[:-2] in words))):
            out.append("word")  # a header word or its plural ('Skills', 'Classes')
        elif k in first or k in last:
            out.append("name")
        else:
            out.append("other")
    # multi-word program, product and vendor names (``Green Dot``, ``Cal Grant``, ``John Deere``): never name words
    phrases = namesakes.program_phrases()
    i = 0
    while i < len(toks):
        for ph in phrases.get(toks[i].key, ()):
            if tuple(t.key for t in toks[i:i + len(ph)]) == ph:
                for k in range(i, i + len(ph)):
                    out[k] = "program"
                i += len(ph) - 1
                break
        i += 1
    _mark_namesakes(toks, out)
    if not s and toks:  # rebuild a text with the tokens at their places
        buf = [" "] * toks[-1].end
        for t in toks:
            buf[t.start:t.end] = t.raw
        s = "".join(buf)
    _mark_eponyms(toks, out, s)
    _mark_court_cases(toks, out, s)
    return out


def _blocking_name(toks: list[_Tok], kinds: list[str], i: int) -> bool:
    """Token ``i`` is a name word that makes a namesake beside it part of a longer person's name
    (``Jose Marti Garcia``): on the lists, no everyday word (``Neil Armstrong Moon`` stays a namesake)."""
    k = toks[i].key
    return kinds[i] == "name" and _distinctive(k) and k not in _EVERYDAY_SURNAMES


def _mark_namesakes(toks: list[_Tok], kinds: list[str]) -> None:
    """Runs that spell a namesake (``privacy.namesakes.find_phrases``: ``Harriet Tubman``, ``Atticus Finch``,
    ``Robert E. Lee``, ``Mrs. Frisby``) become ``program`` tokens, unless a name word stands right beside the run
    (``Jose Marti Garcia`` is a person); a brand that is also a common person's name (``Sara Lee``) only before a
    business word or a number (``Sara Lee Bread``, ``Edward Jones 403b``)."""
    n = len(toks)
    if n < 2:
        return
    skippable = [kd in _TRANSPARENT for kd in kinds]
    found = namesakes.find_phrases([t.key for t in toks], skippable)
    covered = {k for lo, hi, _ in found for k in range(lo, hi + 1)}  # two namesakes side by side block nothing
    for lo, hi, kind in found:
        before = lo - 1
        while before >= 0 and kinds[before] in _TRANSPARENT:
            before -= 1
        after = hi + 1
        while after < n and kinds[after] in _TRANSPARENT:
            after += 1
        if (before >= 0 and before not in covered and _blocking_name(toks, kinds, before)) or (
                after < n and after not in covered and _blocking_name(toks, kinds, after)):
            continue
        if kind == "person" and not (after < n and (kinds[after] == "digit"
                                                    or toks[after].key in namesakes.BUSINESS_WORDS)):
            continue
        for k in range(lo, hi + 1):
            if kinds[k] != "digit":
                kinds[k] = "program"


def _mark_eponyms(toks: list[_Tok], kinds: list[str], s: str) -> None:
    """A run of name words right before a word of ``privacy.namesakes.EPONYM_WORDS`` names a law, model,
    structure, event or era (``Charles's Law``, ``Lewis Dot Structures``, ``John Brown's Raid``, ``Ming
    Dynasty``, ``Calvin-Benson Cycle``): its words become ``program`` tokens (the eponym word a header word). A
    run of two name words or more is an eponym only in the possessive (``John Brown's Raid``) or as two
    hyphenated surnames (``Hardy-Weinberg``): ``Ana Garcia Trial Placement`` and ``Ana Garcia Medication
    Administration Record`` name a person."""
    case_info = any(ch.islower() for ch in s) and any(ch.isupper() for ch in s)
    for i in range(1, len(toks)):
        if toks[i].key not in namesakes.EPONYM_WORDS or not toks[i].raw.isalpha():
            continue
        j = i - 1
        run: list[int] = []
        while j >= 0 and kinds[j] in ("name", "other", "initial", "particle", "suffix", "program") and (
                not case_info or toks[j].raw[:1].isupper() or kinds[j] == "particle"):
            if kinds[j] in ("name", "other"):
                run.append(j)
            j -= 1
        if len(run) >= 2:
            last = toks[i - 1]
            possessive = bool(_POSSESSIVE.search(last.raw)) or s[last.end:last.end + 1] in ("'", "’")
            hyphen_pair = len(run) == 2 and s[toks[run[1]].end:toks[run[0]].start] == "-"
            if not (possessive or hyphen_pair):
                continue
        for k in run:
            kinds[k] = "program"
        if kinds[i] in ("name", "other"):
            kinds[i] = "word"


def _mark_court_cases(toks: list[_Tok], kinds: list[str], s: str) -> None:
    """``v.`` or ``vs.`` between two capitalized names makes a court case (``Morse v. Frederick``, ``Brown v.
    Board of Education``, ``Roe vs. Wade``): the name words on both sides become ``program`` tokens. Only in a
    name written in mixed case with a lower-case ``v`` (``Ana V. Garcia``, ``ana v garcia``: a middle initial)."""
    n = len(toks)
    if not (any(ch.islower() for ch in s) and any(ch.isupper() for ch in s)):
        return
    for i in range(1, n - 1):
        t = toks[i]
        if t.raw not in ("v", "vs", "Vs", "VS") or s[toks[i - 1].end:t.start] not in (" ", "_") \
                or not toks[i - 1].raw[:1].isupper() or not toks[i + 1].raw[:1].isupper():
            continue
        if kinds[i - 1] not in ("name", "other", "word", "inst", "place", "program") \
                or kinds[i + 1] not in ("name", "other", "word", "inst", "place", "program", "particle"):
            continue
        for rng in (range(i - 1, max(-1, i - 4), -1), range(i + 1, min(n, i + 4))):
            for k in rng:
                if kinds[k] not in ("name", "other", "particle", "initial", "suffix"):
                    break
                kinds[k] = "program"


_TRANSPARENT = frozenset({"particle", "initial", "suffix"})
#: School name forms that make a person's name before them a school (``Jesse H Jones High School``, ``Lily B
#: Clayton Elementary``, ``Sam Rayburn ECHS``, ``Mickey Leland College Prep``, ``Dr. Phillips High School``): a
#: school noun (not one followed by a word of :data:`_SCHOOL_DOC`: ``Ana Garcia School Records``), or school
#: levels followed by a school noun or ending the name (``Jack Yates Senior High``, ``Lincoln High``). Any other
#: institution word (``Senior``, ``College``, ``Scholarship``, ``Room``, ``House``, ``Hospital``, ``Library``)
#: right after a name names a document of that person (``Ana Garcia College Essay``, ``Ms. Okafor Room``).
_SCHOOL_NOUNS = frozenset({"school", "schools", "sch", "academy", "academies", "elementary", "elem", "es", "ms",
                           "hs", "isd", "cisd", "cusd", "usd", "prep", "preparatory", "montessori", "magnet", "echs",
                           "ecc", "elc", "jhs", "shs", "ywla", "ymla", "mast"})
#: Words of a school's level or kind between a name and its school noun (``Jack Yates Senior High School``,
#: ``Gus Garcia Young Men's Leadership Academy``, ``Sudie L Williams Talented and Gifted Academy``, ``Barbara M
#: Manns Education Center``).
_SCHOOL_LEVELS = frozenset({"high", "middle", "junior", "senior", "intermediate", "primary", "secondary", "early",
                            "college", "career", "charter", "stem", "steam", "community", "technical", "tech",
                            "vocational", "alternative", "international", "leadership", "learning", "public",
                            "catholic", "day", "childhood", "fine", "arts", "performing", "visual", "young", "men",
                            "mens", "women", "womens", "boys", "girls", "talented", "gifted", "and", "of", "for",
                            "law", "education", "educational", "science", "sciences", "math", "mathematics",
                            "technology", "engineering", "health", "medical", "professions", "global", "language",
                            "languages", "dual", "bilingual", "classical", "collegiate", "military"})
_LEVEL_ENDS = frozenset({"high", "middle", "intermediate"})
#: A school noun only after a school level (``Education Center``, ``Magnet Campus``; not ``Kennedy Center``).
_LEVEL_NOUNS = frozenset({"center", "centre", "campus"})
#: Initials that spell a school form in AskTED's upper-case campus names (``JACK YATES H S``, ``J H S``).
_INITIAL_SCHOOLS = frozenset({"hs", "ms", "jhs", "jh", "es"})
#: A school type before a school number (New York's ``PS 123 Mahalia Jackson``, ``MS 51 William Alexander``,
#: ``JHS 189 Daniel Carter Beard``, ``P.S. 9``).
_SCHOOL_NUMBER_PREFIXES = frozenset({"ps", "is", "ms", "jhs", "hs", "ims", "mhs", "pk"})
_SCHOOL_DOC = frozenset({"year", "years", "yr", "records", "record", "transcript", "transcripts", "history", "file",
                         "files", "folder", "photo", "photos", "picture", "nurse", "counselor", "transfer",
                         "supplies", "days", "bus", "id", "lunch", "uniform"})
#: Company words that make a name before them a business (``Patel Brothers``, ``Smith & Sons``, ``Garcia LLC``).
_CORPORATE = frozenset({"inc", "llc", "ltd", "corp", "pllc", "llp", "plc", "co", "company", "brothers", "sons",
                        "associates", "partners", "enterprises", "industries"})
#: Words right before a name that make it a place (``Fort Worth``, ``St John``, ``San Jose``, ``Bishop Lynch``);
#: never an institution word (``Senior Garcia, Ana``, ``Scholarship - Ana Garcia``, ``Room Ms Garcia`` name a
#: person) and never a title.
_PLACE_BEFORE = _PLACE_WORDS | CHURCH_TITLES | frozenset({"st", "ste", "saint", "sainte"})


def _place_before(toks: list[_Tok], kinds: list[str], lo: int) -> bool:
    """A place word or clergy title right before the name starting at ``lo`` (``Fort Worth``, ``San Jose``, ``Bishop
    Lynch``), or a school type and number (``PS 123 Mahalia Jackson``). (:func:`person_name_hit`; also D36,
    ``detect.org_names``.)"""
    i = lo - 1
    while i >= 0 and kinds[i] == "initial":
        i -= 1
    if i >= 0 and toks[i].key in _PLACE_BEFORE:
        return True
    if i >= 1 and kinds[i] == "digit":
        k = i - 1
        while k >= 0 and k >= i - 3 and kinds[k] == "initial":
            k -= 1
        prefix = "".join(t.key for t in toks[k + 1:i]) if k < i - 1 else toks[i - 1].key
        return prefix in _SCHOOL_NUMBER_PREFIXES
    return i >= 1 and kinds[i] == "particle" and toks[i - 1].key in _PLACE_BEFORE


def _school_form(toks: list[_Tok], kinds: list[str], j: int) -> bool:
    """School levels, then a school noun not followed by a word of a person's record (``High School``, ``YMLA``,
    ``Education Center``; not ``School Records``), initials spelling one (``H S``) or AskTED's upper-case ``EL``, at
    ``j``; or a level ending the name (``Lincoln High``)."""
    n = len(toks)
    k = j
    while k < n and toks[k].key in _SCHOOL_LEVELS and toks[k].key not in _SCHOOL_NOUNS:
        k += 1
    if k < n and (toks[k].key in _SCHOOL_NOUNS or toks[k].raw == "EL" or (k > j and toks[k].key in _LEVEL_NOUNS)):
        return not (k + 1 < n and toks[k + 1].key in _SCHOOL_DOC)
    if k < n and kinds[k] == "initial":
        m = k
        while m < n and m < k + 3 and kinds[m] == "initial":
            m += 1
        if "".join(t.key for t in toks[k:m]) in _INITIAL_SCHOOLS and (m >= n or kinds[m] != "initial"):
            return True
    return k > j and toks[k - 1].key in _LEVEL_ENDS and all(kinds[x] == "digit" for x in range(k, n))


def _place_after(toks: list[_Tok], kinds: list[str], hi: int, case_info: bool) -> bool:
    """A school name form or a company word right after the name ending at ``hi``, initials and suffixes aside
    (:data:`_SCHOOL_NOUNS`: ``Jesse H Jones High School``, ``Lily B Clayton Elementary``, ``JACK YATES H S``; ``Patel
    Brothers``), also after one or two more capitalized words of a longer school name (``Kay Bailey Hutchison
    Elementary``, ``Bertha Sadler Means YWLA``, ``Yvonne A. Ewell Townview Magnet Center``). (:func:`person_name_hit`;
    also D36, ``detect.org_names``: ``Mrs. Quenby's Academy`` is a school.)"""
    n = len(toks)
    j = hi + 1
    if _school_form(toks, kinds, j):
        return True
    while j < n and kinds[j] in ("initial", "suffix"):
        j += 1
    if j < n and toks[j].key in _CORPORATE:
        return True
    for _ in range(3):
        if _school_form(toks, kinds, j):
            return True
        if j >= n or kinds[j] not in ("name", "other", "initial", "particle", "suffix", "inst") \
                or not toks[j].raw.isalpha() or not _is_capital(toks[j], case_info):
            return False
        j += 1
    return False


@dataclass(frozen=True)
class NameHit:
    """Person-name content found in a name: the layout and its name words (PRIVATE)."""

    kind: str  # 'pair' | 'title' | 'initials'
    words: tuple[str, ...]


def person_name_hit(text: str) -> NameHit | None:
    """The person-name content of a header, sheet, file or folder name (module docstring, check 2), or None."""
    script = _script_name_hit(text)
    if script is not None:
        return script
    s = clean_text(text)
    if not any(ch.isalpha() for ch in s):
        return None
    toks = _tokens(s)
    if not toks:
        return None
    kinds = _token_kinds(toks, s)
    first, last = known_names()
    common = name_lists.common_words()
    states = us_states()
    all_states = us_states(given_names_too=True)
    case_info = any(ch.islower() for ch in s) and any(ch.isupper() for ch in s)
    n = len(toks)

    def sep(i: int) -> str:  # text between token i and the next one (or the end)
        return s[toks[i].end:toks[i + 1].start] if i + 1 < n else s[toks[i].end:]

    def run_of(i: int) -> tuple[int, int]:
        """First and last name token of the name run around the name token ``i`` (initials, particles and
        suffixes inside it)."""
        lo = hi = i
        k = i - 1
        while k >= 0 and kinds[k] in _TRANSPARENT | {"name"}:
            if kinds[k] == "name":
                lo = k
            k -= 1
        k = i + 1
        while k < n and kinds[k] in _TRANSPARENT | {"name"}:
            if kinds[k] == "name":
                hi = k
            k += 1
        return lo, hi

    def place_before(lo: int) -> bool:
        return _place_before(toks, kinds, lo)

    def place_after(hi: int) -> bool:
        return _place_after(toks, kinds, hi, case_info)

    def words(lo: int, hi: int) -> tuple[str, ...]:
        return tuple(toks[k].key for k in range(lo, hi + 1) if kinds[k] == "name")

    def namesake(lo: int, hi: int) -> bool:
        """The run is a notable namesake (``Rosa Parks``) or starts with a clergy title (``Bishop Lynch``)."""
        ws = words(lo, hi)
        return namesakes.is_namesake(ws) or (len(ws) >= 2 and toks[lo].key in CHURCH_TITLES)

    def exempt(lo: int, hi: int) -> bool:
        """The name run toks[lo..hi] names a school, a business, a namesake or a program: a school name form or
        company word right after it, a place word or clergy title right before it."""
        return place_after(hi) or place_before(lo) or namesake(lo, hi)

    def strong_surname(i: int) -> bool:
        k = toks[i].key
        return (kinds[i] == "name" and k in last and len(k) >= 3 and _distinctive(k)
                and k not in _EVERYDAY_SURNAMES)

    def given_only(i: int) -> bool:
        """A distinctive given name that is no surname (``Jayden``, ``Aaliyah``, ``Sofia``)."""
        k = toks[i].key
        return (kinds[i] == "name" and k in first and k not in last and len(k) >= 3 and _distinctive(k)
                and k not in _EVERYDAY_SURNAMES)

    def initial_ok(i: int) -> bool:
        """An initial: upper case, or lower case in a name written all in lower case (``j.garcia``)."""
        return 0 <= i < n and kinds[i] == "initial" and (toks[i].raw[:1].isupper() or not case_info)

    def glued_digit(i: int) -> bool:
        """Token ``i`` runs straight into digits (``P1``, ``Q3``: a label, no initial)."""
        return i + 1 < n and kinds[i + 1] == "digit" and sep(i) == ""

    def title_case(i: int) -> bool:
        raw = toks[i].raw
        return not case_info or (raw[:1].isupper() and raw[1:].islower())

    # 1. titles: 'Mr. Smith', 'Ms. Garcia', 'Dr. J. Lee', 'Sr. Garcia', 'Herr Schmidt', 'Counselor Nwosu', 'Mrs. G'
    # (a place word before the title, not the title itself)
    for i in range(n):
        key = toks[i].key
        contextual_sr = key == "sr" and kinds[i] == "suffix" and (
            i == 0 or kinds[i - 1] not in ("name", "other", "initial", "suffix", "particle"))
        if kinds[i] != "title" and not contextual_sr:
            continue
        role = key in _ROLE_TITLES or contextual_sr
        ambiguous = role or key in _AMBIGUOUS_TITLES or (case_info and toks[i].raw.isupper() and len(key) > 1)
        j = i + 1
        while j < n and kinds[j] == "initial":
            j += 1
        if j > i + 1 and (j >= n or kinds[j] not in ("name", "other")):
            # a title and a bare initial: 'Mrs. G', 'Mr. G's Class'
            if key in _INITIAL_TITLES and not ambiguous and initial_ok(i + 1) and not glued_digit(i + 1) \
                    and not place_before(i) and not (i + 2 < n and kinds[i + 2] == "initial" and sep(i + 1) == ""):
                return NameHit("title", ())
            continue
        if j >= n or kinds[j] not in ("name", "other") or not _is_capital(toks[j], case_info):
            continue
        if kinds[j] == "other":  # a surname off the lists after 'Mr' / 'Mrs' / 'Ms' ('Mr. Quarshie')
            if ambiguous or len(toks[j].key) < 3 or not toks[j].raw.isalpha() or not case_info \
                    or toks[j].raw.isupper():
                continue
            if not (place_after(j) or place_before(i)):
                return NameHit("title", (toks[j].key,))
            continue
        if ambiguous and (not _distinctive(toks[j].key) or (role and toks[j].key in _EVERYDAY_SURNAMES)):
            continue
        lo, hi = run_of(j)
        if not (place_after(hi) or place_before(i) or namesake(lo, hi)):
            return NameHit("title", words(lo, hi))
    # 2. comma layouts: 'Smith, John' / 'Okafor, Summer' / 'Park, Jennifer' / 'Smith, J.' / 'Hall, J.' /
    # 'Adeyemi, Oluwaseun' (a given name off the lists after a distinctive surname); ';' like ','. A US state
    # after the comma is a place ('Allen, Texas')
    for i in range(n - 1):
        between = sep(i)
        if ("," not in between and ";" not in between) or kinds[i] not in ("name", "inst", "word", "other"):
            continue
        left, right = toks[i], toks[i + 1]
        if not (_is_capital(left, case_info) and _is_capital(right, case_info)) or not left.raw.isalpha():
            continue
        lk, rk = left.key, right.key
        if rk in all_states:
            continue
        hit = None
        if kinds[i + 1] == "initial" and right.raw.isupper():
            dotted = sep(i + 1).startswith(".")
            initials = "".join(t.key for t in toks[i + 1:i + 3] if t.raw.isalpha() and len(t.key) == 1)
            if initials == "dc" and lk == "washington":
                continue  # 'Washington, D.C.'
            if (lk in last and (dotted or (kinds[i] == "name" and _distinctive(lk)))) or (
                    kinds[i] == "other" and dotted and len(lk) >= 3):
                hit = "initials"
        elif kinds[i + 1] in ("name", "word") and rk in first:
            r_strong = kinds[i + 1] == "name" and _distinctive(rk)
            l_strong = kinds[i] == "name" and lk in last and _distinctive(lk)
            l_unknown = kinds[i] == "other" and len(lk) >= 3
            if (lk in last and (l_strong or r_strong)) or (l_unknown and r_strong):
                hit = "pair"
        elif kinds[i + 1] == "other" and kinds[i] == "name" and lk in last and len(lk) >= 3 and _distinctive(lk) \
                and lk not in _EVERYDAY_SURNAMES and right.raw.isalpha() and len(rk) >= 3 and rk not in common \
                and rk not in _COMPANY_WORDS and not rk.endswith(_EVERYDAY_ENDINGS) and title_case(i + 1):
            hit = "pair"
        if hit is None:
            continue
        lo = i
        while lo >= 1 and kinds[lo - 1] in ("name", "particle") and sep(lo - 1).strip(" -") == "":
            lo -= 1
        hi = i + 1
        while hi + 1 < n and kinds[hi + 1] in ("name", "other", "initial", "particle", "suffix") \
                and sep(hi).strip(" .-") == "" and (kinds[hi + 1] != "other" or hi == i + 1 and kinds[hi] == "other"):
            hi += 1
        ws = tuple(toks[k].key for k in range(lo, hi + 1) if kinds[k] not in ("initial", "particle", "suffix")
                   and len(toks[k].key) > 1)
        saint = lo >= 1 and toks[lo - 1].key in ("st", "ste", "saint", "sainte")  # 'St. Clair, Ana': a surname
        if (place_before(lo) and not saint) or namesakes.is_namesake(ws):
            continue
        return NameHit("pair" if hit == "pair" else "initials", ws)
    # 3. initials: 'J. Smith' / 'Smith J.' / 'j.garcia'
    for i in range(n):
        if kinds[i] != "name" or not _is_capital(toks[i], case_info):
            continue
        key = toks[i].key
        hit = False
        if i >= 1 and initial_ok(i - 1) and sep(i - 1).startswith(".") and (
                toks[i - 1].raw.isupper() or strong_surname(i)):
            hit = True
        if not hit and i + 1 < n and initial_ok(i + 1) and toks[i + 1].raw.isupper() \
                and sep(i).strip(" _") == "" and sep(i + 1).startswith(".") \
                and not sep(i + 1)[1:2].isalnum() and key in last and _distinctive(key):
            hit = True
        if hit:
            lo, hi = run_of(i)
            if not exempt(lo, hi):
                return NameHit("initials", words(lo, hi))
    # 4. a distinctive surname and a bare initial: 'Smith_J_IEP', 'GarciaA', 'SmithJ', 'J Smith', 'j_garcia',
    # 'Rodriguez.M', 'garcia.a' ('Garcia P1' is a period label: an initial running into digits is none)
    for i in range(n):
        if not strong_surname(i) or not _is_capital(toks[i], case_info):
            continue
        if initial_ok(i + 1) and sep(i).strip(" _-.") == "" and not sep(i + 1)[:1].isalnum() \
                and not glued_digit(i + 1) and not (i + 2 < n and kinds[i + 2] == "initial" and sep(i + 1) == ""):
            lo, hi = run_of(i)
            if not exempt(lo, hi):
                return NameHit("initials", words(lo, hi))
        if i >= 1 and initial_ok(i - 1) and sep(i - 1).strip(" _") == "" \
                and (i == 1 or not sep(i - 2).endswith(" ") or sep(i - 2).strip() != ""):
            lo, hi = run_of(i)
            if not exempt(lo, hi):
                return NameHit("initials", words(lo, hi))
    # 5. name pairs: a known given name next to a known surname, either order (initials, particles, suffixes in
    # between), neither of them an everyday or header word ('Ana Garcia', 'GARCIA_ANA'; not 'Summer Davis', 'Ana
    # Price', 'Ana Hall', 'Oak Hill', 'Tom Bean'). A pair is a place only by a school name form right after its run
    # or a place word right before it ('Kay Bailey Hutchison Elementary', 'Fort Worth'); a city before a US state
    # is no pair ('Sherman Texas'). A two-letter East Asian surname that is no everyday word beside a known given
    # name is a pair too ('Le Minh', 'Wei Li')
    def pair_word(i: int) -> bool:
        k = toks[i].key
        return kinds[i] == "name" and _distinctive(k) and not _everyday(k)

    runs: set[tuple[int, int]] = set()
    for i in range(n):
        if kinds[i] != "name":
            continue
        lo, hi = run_of(i)
        if (lo, hi) in runs:
            continue
        runs.add((lo, hi))
        ws = words(lo, hi)
        if place_after(hi) or namesakes.is_namesake(ws) or toks[lo].key in CHURCH_TITLES:
            continue
        # any given name and surname of the run, an everyday middle name between them ('Ana Rose Garcia')
        idx = [k for k in range(lo, hi + 1) if pair_word(k)]
        for x_i, x in enumerate(idx):
            for y in idx[x_i + 1:]:
                a, b = toks[x].key, toks[y].key
                if ((a in first and b in last) or (a in last and b in first)) and a not in states \
                        and b not in states and not place_before(x):
                    return NameHit("pair", ws)
    asian = world_names.ASIAN_SURNAMES
    for i in range(n - 1):
        if not _NAME_GAP.fullmatch(sep(i)) or (not case_info and len([k for k in kinds if k != "digit"]) != 2):
            continue
        for g, u in ((i, i + 1), (i + 1, i)):
            gk = toks[g].key
            if asian.get(gk) == "short" and toks[g].raw.isalpha() \
                    and title_case(g) and pair_word(u) and toks[u].key in first and title_case(u) \
                    and not (place_after(i + 1) or place_before(i)):
                return NameHit("pair", (toks[i].key, toks[i + 1].key))
    # 5b. a distinctive given name and a last initial: 'Jayden M.', 'Jayden_M', 'JaydenM', 'Sofia R. - Reading Log'
    # (not a Roman numeral without a period: 'Elizabeth I', 'Henry V'); a given name that is a header word only
    # with a period as the whole name ('Max T.', 'Mark B.'). A capitalized word after the initial makes it a middle
    # initial: the name is judged with that word ('Victor H Hexter ES' is a school)
    letters = [k for k in range(n) if kinds[k] != "digit"]
    for i in range(n - 1):
        if not initial_ok(i + 1) or sep(i).strip(" _-.") != "" or glued_digit(i + 1) or sep(i + 1)[:1].isalnum():
            continue
        dotted = sep(i + 1).startswith(".")
        if not dotted and toks[i + 1].key in ("i", "v", "x"):
            continue
        if given_only(i) and title_case(i):
            lo, hi = run_of(i)
            m = i + 2
            if m < n and kinds[m] in ("name", "other") and toks[m].raw.isalpha() and _is_capital(toks[m], case_info) \
                    and sep(i + 1).strip(" _.") == "":
                hi = max(hi, run_of(m)[1] if kinds[m] == "name" else m)
            if not exempt(lo, hi):
                return NameHit("initials", (toks[i].key,))
        elif kinds[i] == "word" and toks[i].key in first and dotted and len(letters) == 2 and i == letters[0] \
                and toks[i].raw.isalpha() and len(toks[i].key) >= 3 and title_case(i) and case_info:
            return NameHit("initials", (toks[i].key,))
    # 5c. user names and run-together names in lower (or upper) case: an initial or two glued before or after a
    # common surname ('jgarcia', 'agarcia2', 'mrodriguez', 'smithj', 'rodriguezm', 'nguyent2', 'JSMITH',
    # 'Roster_AGARCIA'), a title glued to one ('msgarcia', 'mrsgarcia'), a common given name glued to a common
    # surname ('anagarcia', 'garciaana')
    popular_first, popular_last = popular_names()
    for i in range(n):
        t = toks[i]
        k = t.key
        if kinds[i] != "other" or not t.raw.isalpha() or not (t.raw.islower() or t.raw.isupper()) or len(k) < 5 \
                or k in common or k in _USERNAME_LOOKALIKES:
            continue
        rest = k[1:]
        if _username_surname(rest, popular_last):
            return NameHit("pair", (rest,))
        if k[-1] != "s" and _username_surname(k[:-1], popular_last):
            return NameHit("pair", (k[:-1],))
        for title in _GLUED_TITLES:
            if k.startswith(title) and len(k) - len(title) >= 4 and _username_surname(k[len(title):], popular_last):
                return NameHit("title", (k[len(title):],))
        if len(k) >= 7:
            for cut in range(3, len(k) - 3):
                a, b = k[:cut], k[cut:]
                for g, sur in ((a, b), (b, a)):
                    if g in popular_first and _username_surname(sur, popular_last) and _distinctive(g) \
                            and g not in _EVERYDAY_SURNAMES:
                        return NameHit("pair", (a, b))
    return None


def _everyday(key: str) -> bool:
    """An everyday word that is also a surname, or its plural (``Hill``, ``Hills``, ``Bean``, ``Wells``): never one
    word of a name pair."""
    return key in _EVERYDAY_SURNAMES or (len(key) > 3 and key.endswith("s") and key[:-1] in _EVERYDAY_SURNAMES)


#: The text between a two-letter surname and a given name: nothing (camelCase), a space, ``_``, ``-`` or ``.``, a
#: dash between spaces.
_NAME_GAP = re.compile(r"[ _]{0,2}[-.\u2013\u2014]?[ _]{0,2}")
#: Titles glued to a surname in a lower-case user name (``msgarcia``, ``mrsgarcia``, ``drpatel``).
_GLUED_TITLES = ("mrs", "miss", "mr", "ms", "dr", "coach")


#: Cyrillic and Greek letters as read in a word written in that script alone (``Иван Петров`` is ``Ivan Petrov``,
#: ``Μαρία`` is ``Maria``); a word that mixes them with Latin letters is read by look-alike (:data:`_CONFUSABLES`).
#: One dict of more than 50 entries (data).
_TRANSLIT: dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "ґ": "g", "д": "d", "е": "e", "ё": "e", "є": "ye", "ж": "zh",
    "з": "z", "и": "i", "і": "i", "ї": "yi", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "у": "u", "ў": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh",
    "щ": "shch", "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ђ": "dj", "ј": "j", "љ": "lj",
    "њ": "nj", "ћ": "c", "џ": "dz", "ѓ": "g", "ќ": "k", "ѕ": "dz",
    "α": "a", "β": "v", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "i", "θ": "th", "ι": "i", "κ": "k",
    "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "y",
    "φ": "f", "χ": "ch", "ψ": "ps", "ω": "o",
}
_SCRIPT_WORD = re.compile(r"[^\W\d_]+")


def _is_cyrillic_or_greek(ch: str) -> bool:
    return "Ͱ" <= ch <= "Ͽ" or "Ѐ" <= ch <= "ӿ" or "ἀ" <= ch <= "῿"


def _transliterate(text: str) -> str:
    """``text`` with every word written in Cyrillic or Greek letters alone in Latin letters (:data:`_TRANSLIT`;
    case kept: ``Жанна`` is ``Zhanna``, ``ИВАН`` is ``IVAN``)."""
    def word(m: re.Match) -> str:
        w = m.group()
        if not any(_is_cyrillic_or_greek(ch) for ch in w) or any(ch.isascii() for ch in w):
            return w
        if any("Ͱ" <= ch <= "Ͽ" or "ἀ" <= ch <= "῿" for ch in w):  # Greek: accents off
            w = unicodedata.normalize("NFC", "".join(ch for ch in unicodedata.normalize("NFD", w)
                                                     if not unicodedata.combining(ch)))
        upper = w.isupper() and len(w) > 1
        out = []
        for ch in w:
            lo = ch.lower()
            t = _TRANSLIT.get(lo)
            if t is None:
                out.append(ch)
            elif ch != lo:
                out.append(t.upper() if upper else t[:1].upper() + t[1:])
            else:
                out.append(t)
        return "".join(out)

    return _SCRIPT_WORD.sub(word, text)


_ARABIC_WORD = re.compile(r"[ؠ-يٱ-ۓ]+")
_ARABIC_MARKS = re.compile(r"[ً-ٰٟـ]")
_ARABIC_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                              "ى": "ي", "ة": "ه"})
_HAN_RUN = re.compile(r"[一-鿿]+")
_HANGUL_RUN = re.compile(r"[가-힣]+")


def _script_name_hit(text: str) -> NameHit | None:
    """A person's name written in a script the Latin lists cannot read: two Arabic name words
    (``world_names.ARABIC_NAMES``: ``محمد علي``; ``عبد`` and the word after it are one), a Chinese surname and
    one or two given-name characters as a word of their own (``张伟``, ``王梓涵``), a Korean surname and a common
    two-syllable given name (``김민준``, ``김 민준``)."""
    if text.isascii():
        return None
    t = unicodedata.normalize("NFC", text)
    if _ARABIC_WORD.search(t):
        words = [w.translate(_ARABIC_FOLD) for w in _ARABIC_WORD.findall(_ARABIC_MARKS.sub("", t))]
        names = 0
        k = 0
        while k < len(words):
            w = words[k]
            if w == "عبد" and k + 1 < len(words):  # 'Abd' + the next word: one name
                names += 1
                k += 2
                continue
            names += w in world_names.ARABIC_NAMES or (w.startswith("عبد") and len(w) > 4)
            k += 1
        if names >= 2:
            return NameHit("pair", ())
    for run in _HAN_RUN.findall(t):
        if 2 <= len(run) <= 3 and run[0] in world_names.HAN_SURNAMES \
                and all(ch in world_names.HAN_GIVEN for ch in run[1:]):
            return NameHit("pair", ())
    runs = _HANGUL_RUN.findall(t)
    for k, run in enumerate(runs):
        if len(run) == 3 and run[0] in world_names.HANGUL_SURNAMES and run[1:] in world_names.HANGUL_GIVEN:
            return NameHit("pair", ())
        if len(run) == 1 and run in world_names.HANGUL_SURNAMES and k + 1 < len(runs) \
                and runs[k + 1] in world_names.HANGUL_GIVEN:
            return NameHit("pair", ())
    return None


def person_name_kind(text: str) -> str | None:
    """``'pair'``, ``'title'`` or ``'initials'`` when ``text`` holds person-name content, else None."""
    hit = person_name_hit(text)
    return hit.kind if hit else None


# --------------------------------------------------------------------------- real IDs
_ID_SPLIT = re.compile(r"[\s_.,;:|/\\()\[\]{}#]+")
_ID_SPLIT_DASH = re.compile(r"[\s_.,;:|/\\()\[\]{}#\-]+")
MIN_ID_LENGTH = 4
#: A name alone with a match of at most this chance holds a real ID (module docstring, check 3).
SINGLE_CHANCE = 1e-3
#: Sibling names sharing their text around distinct real IDs: chance at most this that so many of the numbers
#: in that text are real IDs (the pivot, the per-student files of a folder: :func:`id_group_flags`).
GROUP_CHANCE = 1e-2
#: Words of a parent name (folder, workbook) that say what its children's numbers are (``enrollment_by_zip.xlsx``
#: with a sheet per ZIP code, ``rooms/``, ``by_campus/``): such a child named by a number alone is no ID.
CONTEXT_NUMBER_WORDS = frozenset({"zip", "zips", "zipcode", "zipcodes", "postal", "room", "rooms", "course",
                                  "courses", "section", "sections", "campus", "campuses", "school", "schools",
                                  "building", "buildings", "bldg", "bus", "buses", "route", "routes", "cip", "naics",
                                  "soc", "fips", "tract", "tracts", "cds", "nces", "ceeb", "lea", "esc", "district",
                                  "districts", "region", "regions", "zone", "zones", "locker", "lockers", "isbn",
                                  "sku", "account", "accounts", "code", "codes"})


#: The words of :data:`CONTEXT_NUMBER_WORDS` that name a code system: a parent holding one says its children are
#: such codes wherever the word stands (``ZIP_Enrollment``, ``enrollment_by_zip``, ``NAICS_2022``). The other words
#: name a place or a thing (``School``, ``Campus``, ``District``, ``Section``): only a collection of them says what the
#: children are (P-06: plural, ``Rooms``, ``by_campus``), never one of them that the parent itself is
#: (``Lincoln_Middle_School_Gradebook``, ``Westside_Campus_IEP_Progress``, ``Springfield_School_District``).
_CODE_SYSTEM_WORDS = frozenset({"zip", "zips", "zipcode", "zipcodes", "postal", "cip", "naics", "soc", "fips", "cds",
                                "nces", "ceeb", "isbn", "sku"})
#: Words before a singular place or thing word that make the parent a collection of them (``enrollment_by_campus``).
_COLLECTION_WORDS = frozenset({"by", "per", "each", "every"})


def _name_words(name: str) -> list[str]:
    """The folded word tokens of a name, camelCase split (``enrollmentByZip`` -> ``enrollment by zip``)."""
    toks = _WORD_RE.findall(clean_text(name))
    return [t.casefold() for part in toks for t in (_CAMEL_RE.split(part) if not part.isdigit() else [part]) if t]


def numbered_context(parent: str | None) -> bool:
    """The parent name says what its children's numbers are: a word of :data:`CONTEXT_NUMBER_WORDS` that no number
    follows (not ``Period 3``, whose number is its own) and that names the children: a code system
    (:data:`_CODE_SYSTEM_WORDS`: ``enrollment_by_zip``), a plural (``Rooms``, ``Bus_Routes``, ``Campuses``) or a word
    after ``by``/``per``/``each``/``every`` (``by_campus``). P-06: a singular place word that the parent itself names
    (``Lincoln_Middle_School_Gradebook``, ``Springfield_School_District``) says nothing about its children."""
    if not parent:
        return False
    toks = _name_words(parent)
    for k, t in enumerate(toks):
        if t not in CONTEXT_NUMBER_WORDS or (k + 1 < len(toks) and toks[k + 1].isdigit()):
            continue
        plural = t.endswith("s") and (t[:-1] in CONTEXT_NUMBER_WORDS
                                      or (t.endswith("es") and t[:-2] in CONTEXT_NUMBER_WORDS))
        if t in _CODE_SYSTEM_WORDS or plural or (k > 0 and toks[k - 1] in _COLLECTION_WORDS):
            return True
    return False


def person_context(parent: str | None) -> bool:
    """K-01 (P-05): the parent name (a folder, a workbook) holds a person or document label (:data:`_AFTER_LABELS`
    words, also in the plural: ``iep/``, ``IEPs/``, ``students/``, ``Evaluations/``, ``iep_progress.xlsx``,
    ``Westside_Campus_IEP_Progress.xlsx``) and no numbered context (:func:`numbered_context`): its children named by a
    number alone are per-person names."""
    if not parent or numbered_context(parent):
        return False
    words = set(_name_words(parent)) | {t.casefold() for t in _WORD_RE.findall(clean_text(parent))}  # 'IEPs' whole
    return any(not t.isdigit() and (t in _AFTER_LABELS or (t.endswith("s") and t[:-1] in _AFTER_LABELS))
               for t in words)
#: Sibling names holding three distinct real IDs, whatever their text: joint chance at most this (also the run-wide
#: evidence of :meth:`IdKeys.note_run_names` and a numbered parent's excuse of a pivot, :func:`pivot_flags`).
ANY_GROUP_CHANCE = 1e-6
#: A8: the fewest weak real IDs of one digit count in the run's names that can be too many for chance.
EVIDENT_MIN = 3
#: Chance floor of a number of fewer than 5 digits inside a longer name (``Lexile 1000``, ``1500 m``).
SHORT_NUMBER_CHANCE = 0.05
#: Digit counts whose chance is also read locally (:meth:`IdKeys.number_chance`): the half-width of the window
#: of numbers around a 4-digit number whose share of real IDs it takes.
LOCAL_WINDOW: dict[int, int] = {4: 50}
#: Words right before a number that make it a numbered thing, no ID (``Room_10101``, ``Course 10199``,
#: ``CIP_150301``, ``Report_153017``, ``Pct_151200_to_152000``; a route as ``Rt 1201`` or ``Rte_1305`` like
#: ``Route 1201``). ``id``, ``score`` and ``student`` are not here: they are the stubs of a pivot. One word list
#: of more than 50 entries (data, never text the tool writes).
NUMBERED_WORDS: frozenset[str] = frozenset({
    "room", "rm", "rooms", "course", "courses", "crs", "section", "sec", "sect", "cip", "period", "per", "bldg",
    "building", "lab", "unit", "lesson", "chapter", "ch", "item", "items", "question", "q", "version", "ver", "v",
    "rev", "report", "rpt", "job", "batch", "run", "export", "page", "pg", "form", "test", "lexile", "zip",
    "zipcode", "postal", "ext", "extension", "code", "codes", "cd", "catalog", "cat", "class", "bus", "route",
    "rt", "rte", "locker", "seat", "table", "group", "cohort", "account", "acct", "invoice", "order", "ticket",
    "case", "step",
    "phase", "level", "lvl", "standard", "std", "objective", "obj", "task", "quiz", "exam", "isbn", "model", "part",
    "serial", "sku", "no", "nbr", "num", "number", "school", "sch", "site", "campus", "district", "dist",
    "program", "prog", "grant", "award", "fund", "project", "proj", "contract", "po", "req", "request", "ref",
    "term", "year", "yr", "fy", "sy", "week", "wk", "day", "month", "mo", "hour", "hr", "time", "count", "cnt",
    "total", "pct", "percent", "pts", "points", "rit", "scale", "band", "bin", "range", "from", "to", "thru",
    "through", "between", "over", "under", "above", "below", "max", "min", "top", "bottom", "file", "folder",
    "box", "lot", "zone", "region", "area", "tract", "census", "block", "fips", "nces", "ceeb", "naics", "soc",
    "onet", "esc", "lea", "cds", "ipeds", "unitid", "opeid", "cdn", "ver", "build", "release", "edition",
})
#: Words of numbered things before 10 bare digits that make them no phone number (:func:`_phone_in_name`):
#: :data:`NUMBERED_WORDS` and the words of export, survey, transaction and record numbers (``Extract_5839201746``,
#: ``Qualtrics_SV_5839201746``, ``Student ID 5125550199``).
_PHONE_NUMBERED_WORDS: frozenset[str] = NUMBERED_WORDS | frozenset({
    "extract", "download", "upload", "snapshot", "dump", "survey", "surveys", "sv", "response", "responses",
    "submission", "submissions", "transaction", "txn", "confirmation", "conf", "receipt", "tracking", "claim",
    "record", "records", "id", "ids", "uid", "pid", "eid", "tsds", "sis",
})


def _year(v: str) -> bool:
    return 1900 <= int(v) <= 2100


def date_like(c: str) -> bool:
    """A number that reads as a year, school year, year-month or date (``2019``, ``201920``, ``20192020``,
    ``202309``, ``20230915``): never taken for an ID in a name."""
    if not (c.isascii() and c.isdigit()):
        return False
    if len(c) == 4:
        return _year(c)
    if len(c) == 6:
        y, rest = c[:4], int(c[4:])
        return _year(y) and (rest == (int(y) + 1) % 100 or 1 <= rest <= 12)
    if len(c) == 8:
        y, rest = c[:4], c[4:]
        if _year(y) and _year(rest) and int(rest) == int(y) + 1:
            return True
        mm, dd = int(rest[:2]), int(rest[2:])
        return _year(y) and 1 <= mm <= 12 and 1 <= dd <= 31
    if len(c) in (12, 14):  # YYYYMMDDHHMM(SS)
        return date_like(c[:8]) and int(c[8:10]) <= 23 and int(c[10:12]) <= 59
    return False


def school_year(d: str) -> bool:
    """B1: a school-year code, two years in a row (``2425`` = 2024-25, ``202425``, ``20242025``): the most common
    number in a school's file names (``IEP_Caseload_2425``, ``2425_Student_Roster``), never a person's ID, labelled or
    not, like a year."""
    if not (d.isascii() and d.isdigit()):
        return False
    if len(d) == 4:
        return int(d[2:]) == (int(d[:2]) + 1) % 100
    if len(d) == 6:
        return _year(d[:4]) and int(d[4:]) == (int(d[:4]) + 1) % 100
    if len(d) == 8:
        return _year(d[:4]) and _year(d[4:]) and int(d[4:]) == int(d[:4]) + 1
    return False


def natural_number(d: str) -> bool:
    """A number that reads as a time stamp, a short date or a school year (``152233`` = 15:22:33, ``250115`` =
    2025-01-15, ``011525``, ``01152025``, ``1001`` = October 1, ``2425`` = 2024-25): weak evidence of an ID, whatever
    the namespace."""
    if not d.isdigit():
        return False
    if len(d) == 6:
        a, b, c = int(d[:2]), int(d[2:4]), int(d[4:])
        if a <= 23 and b <= 59 and c <= 59:  # HHMMSS
            return True
        if 1 <= b <= 12 and 1 <= c <= 31:  # YYMMDD
            return True
        if 1 <= a <= 12 and 1 <= b <= 31:  # MMDDYY
            return True
        return 1 <= a <= 31 and 1 <= b <= 12  # DDMMYY
    if len(d) == 8:
        a, b, y = int(d[:2]), int(d[2:4]), d[4:]
        return _year(y) and ((1 <= a <= 12 and 1 <= b <= 31) or (1 <= a <= 31 and 1 <= b <= 12))
    if len(d) == 4:  # MMDD: a daily file or sheet (``Attendance_1001`` = October 1); a school year (``2425``)
        return 1 <= int(d[:2]) <= 12 and 1 <= int(d[2:]) <= 31 or school_year(d)
    return False


def _calendar_day(month: int, day: int) -> bool:
    """``month``/``day`` is a day of the calendar (February 29 too: the century of a 2-digit year is unknown)."""
    return 1 <= month <= 12 and 1 <= day <= (29 if month == 2 else 30 if month in (4, 6, 9, 11) else 31)


def name_date(d: str) -> bool:
    """B1, B2: a number that reads as a date, a time or a school year as people write them in file, sheet and folder
    names: a year, a school year or a year-month (:func:`date_like`, :func:`school_year`), a day of the calendar as
    ``MMDD``, ``MMDDYY``, ``YYMMDD``, ``YYYYMMDD``, ``MMDDYYYY`` or ``DDMMYYYY`` (``101524_Student_Attendance``,
    ``IEP_Meeting_1015``), or a clock time on a five-minute mark (``1330_IEP_Meeting``). Beside a person or document
    label only these readings make a number a date (:meth:`IdKeys.matches`): a time with seconds alone (``150631`` =
    15:06:31; a date-time stamp's time follows its date, :func:`_stamp_time`), a day that is no day of the calendar
    (``150631`` = 2015-06-31) and a day-first ``DDMMYY`` that is no ``YYMMDD`` (every day-first date of 2001-2028
    is one) are no dates people write beside a label: the label's number stays the person's ID."""
    if not (d.isascii() and d.isdigit()):
        return False
    if date_like(d) or school_year(d):
        return True
    if len(d) == 4:
        return _calendar_day(int(d[:2]), int(d[2:])) or _schedule_times([d])
    if len(d) == 6:
        return _calendar_day(int(d[:2]), int(d[2:4])) or _calendar_day(int(d[2:4]), int(d[4:]))
    if len(d) == 8:
        return _year(d[4:]) and (_calendar_day(int(d[:2]), int(d[2:4])) or _calendar_day(int(d[2:4]), int(d[:2])))
    return False


#: B1: the layouts of dates a number of 4, 6 or 8 digits reads in (:func:`name_date`; ``MMDD`` is read in a leap year).
_DATE_LAYOUTS: dict[int, tuple[str, ...]] = {4: ("%m%d",), 6: ("%m%d%y", "%y%m%d", "%Y%m"),
                                             8: ("%Y%m%d", "%m%d%Y", "%d%m%Y")}
#: B1: the days on either side of a date whose numbers in its layout :meth:`IdKeys.date_share` looks at.
_DATE_WINDOW_DAYS = 730


def _date_neighbours(d: str) -> list[list[str]]:
    """B1: for each layout ``d`` reads in as a date or a clock time (:data:`_DATE_LAYOUTS`, ``HHMM`` on a five-minute
    mark), the other numbers of that layout near it (:meth:`IdKeys.date_share`)."""
    out: list[list[str]] = []
    for layout in _DATE_LAYOUTS.get(len(d), ()):
        text, fmt = ("2000" + d, "%Y%m%d") if layout == "%m%d" else (d, layout)
        try:
            day = _dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
        if day.strftime(layout) != d:
            continue  # a parse that is no fixed-width reading of the digits
        if layout == "%m%d":
            dates = [_dt.date(2000, 1, 1) + _dt.timedelta(days=k) for k in range(366)]
        elif layout == "%Y%m":
            dates = [_dt.date(day.year + (day.month - 1 + k) // 12, (day.month - 1 + k) % 12 + 1, 1)
                     for k in range(-24, 25) if 1 <= day.year + (day.month - 1 + k) // 12 <= 9999]
        else:
            lo = max(_dt.date.min.toordinal(), day.toordinal() - _DATE_WINDOW_DAYS)
            hi = min(_dt.date.max.toordinal(), day.toordinal() + _DATE_WINDOW_DAYS)
            dates = [_dt.date.fromordinal(k) for k in range(lo, hi + 1)]
        out.append([v for v in (x.strftime(layout) for x in dates) if v != d and len(v) == len(d)])
    if _schedule_times([d]):
        out.append([f"{h:02d}{m:02d}" for h in range(24) for m in range(0, 60, 5) if f"{h:02d}{m:02d}" != d])
    return out


def id_candidates(text: str) -> set[str]:
    """The whole name and its parts (split at spaces, ``_ . , ; : | / ( ) [ ] { } #``, and also at ``-``) of at
    least :data:`MIN_ID_LENGTH` characters that are no year/date (:func:`date_like`). A part made of digits only
    needs one digit more (``Lexile 1000``, ``1500 m``: a number in a label, not a 4-digit record ID)."""
    s = clean_text(text).strip()
    parts = (set(_ID_SPLIT.split(s)) | set(_ID_SPLIT_DASH.split(s))) - {s}
    cands = {c for c in parts if len(c) >= MIN_ID_LENGTH + (1 if c.isdigit() else 0)} | {s}
    return {c for c in cands if len(c) >= MIN_ID_LENGTH and not date_like(c)}


def person_id_shape(value: str) -> bool:
    """A real ID value that can be screened for in names: at least :data:`MIN_ID_LENGTH` digits and no more
    letters than digits (``482913``, ``S482913``, ``STU-0042``; not the codes ``MATH`` or ``ELA1``)."""
    v = str(value)
    digits = sum(ch.isdigit() for ch in v)
    letters = sum(ch.isalpha() for ch in v)
    return digits >= MIN_ID_LENGTH and letters <= digits


_DIGIT_CORE = re.compile(r"[a-z]{1,4}0*([1-9][0-9]{3,})")
_DIGIT_RUN = re.compile(r"[0-9]+")
_LABEL_BEFORE = re.compile(r"([A-Za-z]+)[\s_.\-#:=]*$")
#: K-01: words of a person or of a document about one right before a number (``Student 612587``, ``Stu_612587``,
#: ``IEP_512340``, ``Evaluation_513820``, ``ID 512340``, ``SID 512340``; ``504`` too: ``504_512340``): a number
#: there that equals a real ID holds it whatever the density of its namespace. ``ID`` (and the other ID words of
#: ``hints.ID_WORDS``: ``Number``, ``No``, ``Num``, ``Nbr``, ``#``) counts on its own (``ID`` only) or after a person
#: word (``Student ID``, ``Student Number``, ``Student No.``, ``Stu Num``, ``StuID``, ``Teacher_ID``), never after
#: another noun (``Course ID 1050``, ``Item ID``, ``Room No 1014``).
PERSON_LABEL_WORDS: frozenset[str] = frozenset({"student", "stu", "pupil", "learner", "iep", "eval", "evaluation",
                                                "id", "sid"})
#: The labels that name the person or the document themselves (every label but a bare ``ID``).
_SUBJECT_LABELS: frozenset[str] = PERSON_LABEL_WORDS - {"id"}
#: P-01: the labels that make a number right BEFORE them a person's (``160816_IEP_Goals``, ``632084 Student``,
#: ``325127_Evaluation``, ``160816_504_Plan``); and P-05: the words of a parent folder or workbook whose children
#: named by a number alone are per-person names (``iep/1053.csv``, ``students/``, ``iep_progress.xlsx``).
_AFTER_LABELS: frozenset[str] = (_SUBJECT_LABELS - {"sid"}) | {"504"}
#: P-02: the labels of a document about one person that one word may separate from the number (``IEP Meeting
#: 1053``, ``504 Plan 1053``, ``Eval 2023 160816``; not a date or a time: ``IEP Due 1015``).
_DOC_LABELS: frozenset[str] = frozenset({"iep", "504", "eval", "evaluation"})
#: Words before an ID word that make it a person's ID (``hints.PERSON_ID_WORDS``: ``Student``, ``Staff``,
#: ``Teacher``...; ``Stu``) and the subject labels, also in the plural (``Students ID``).
_PERSON_ID_QUALIFIERS: frozenset[str] = frozenset(hints.PERSON_ID_WORDS) | _SUBJECT_LABELS
_ID_WORDS: frozenset[str] = frozenset(hints.ID_WORDS)
#: The person-ID header names of D10 (``hints.DEFAULT_ID_SET``) but a bare ``id``, which counts alone only.
_NAMED_ID_HEADERS: frozenset[str] = hints.DEFAULT_ID_SET - {"id"}
#: K-01: the separators between a label and its number (P-02: also a comma, a slash, a bar, brackets and the
#: dashes: ``Student, 512340``, ``Student/512340``, ``Student\u2013512340``).
_LABEL_SEP = r"[\s_.,;:/\\|#=()\[\]{}\-\u2010-\u2015]*"
#: The token right before a position: a letter run or a digit run, then separators only (an opening bracket too:
#: ``Student (512340)``, ``Course (ID 1050)``).
_TOKEN_BEFORE = re.compile(r"([A-Za-z]+|[0-9]+)" + _LABEL_SEP + "$")
#: The token right after a position: separators only, then a letter run or a digit run.
_TOKEN_AFTER = re.compile(_LABEL_SEP + r"([A-Za-z]+|[0-9]+)")


def _label_piece(tok: str) -> str:
    """The last camelCase piece of a letter run, case-folded (``StudentIEP`` -> ``iep``)."""
    pieces = [p for p in _CAMEL_RE.split(tok) if p]
    return pieces[-1].casefold() if pieces else ""


def _person_qualifier(word: str) -> bool:
    """``word`` makes the ID word after it a person's ID (:data:`_PERSON_ID_QUALIFIERS`, also in the plural)."""
    return word in _PERSON_ID_QUALIFIERS or (word.endswith("s") and word[:-1] in _PERSON_ID_QUALIFIERS)


def _words_before_number(s: str, start: int, n: int = 4) -> list[str]:
    """Up to ``n`` tokens right before position ``start`` of ``s`` (separators only between them), nearest first, as
    folded words: a letter run as its camelCase pieces (``StudentID`` -> ``id``, ``student``), a digit run as is, and
    a ``#`` between two tokens as an ID word (``Student #512340``)."""
    words: list[str] = []
    pos = start
    for _ in range(n):
        m = _TOKEN_BEFORE.search(s, 0, pos)
        if m is None:
            break
        if "#" in s[m.end(1):pos]:
            words.append("#")
        tok = m.group(1)
        words += [tok] if tok.isdigit() else [p.casefold() for p in reversed(_CAMEL_RE.split(tok)) if p]
        pos = m.start(1)
    return words


def _label_before_kind(s: str, start: int) -> str:
    """K-01: how the number at position ``start`` of ``s`` follows a person or document label (separators only
    between): ``'direct'`` right after one (a subject label, :data:`PERSON_LABEL_WORDS`, ``504``; ID words,
    ``hints.ID_WORDS`` and ``#``, after a person word or ``ID`` alone; a person-ID header name of D10), ``'gap'`` one
    word after a document label (P-02: ``IEP Meeting 1053``, ``504 Plan 1053``, ``Eval 2023 160816``), else ``''``."""
    words = _words_before_number(s, start)
    if not words:
        return ""
    w0 = words[0]
    if w0 == "504" or w0 in _SUBJECT_LABELS:
        return "direct"
    # a person-ID header name of D10 (``hints.DEFAULT_ID_NAMES``: ``Perm ID``, ``State ID``, ``Local ID``, ``SSID``,
    # ``LASID``, ``GTID``); a bare ``ID`` only alone (below), never after another noun (``Course ID``)
    plain = [w for w in words if w != "#"]
    if any("".join(reversed(plain[:n])) in _NAMED_ID_HEADERS for n in range(1, min(3, len(plain)) + 1)):
        return "direct"
    k = 0
    while k < len(words) and (words[k] in _ID_WORDS or words[k] == "#"):
        k += 1
    if k:  # ID words: a person's ID after a person word, or 'ID' alone
        return "direct" if ("id" in words[:k] if k == len(words) else _person_qualifier(words[k])) else ""
    if any(w0 == q + iw for q in _PERSON_ID_QUALIFIERS for iw in _ID_WORDS if w0.startswith(q)):
        return "direct"  # 'STUID', 'STUDENTNUMBER' (no camelCase to split)
    return "gap" if len(words) >= 2 and words[1] in _DOC_LABELS and (not w0.isdigit() or date_like(w0)) else ""


def person_label_before(s: str, start: int, natural: bool = False) -> bool:
    """K-01: the number at position ``start`` of ``s`` follows a person or document label (separators only between),
    so a real ID there is strong evidence on its own: a subject label (:data:`PERSON_LABEL_WORDS`, ``504``); ID words
    (``hints.ID_WORDS``, ``#``) after a person word (P-02: ``Student ID``, ``Student Number``, ``Student No.``, ``Stu
    Num``, ``Student_Nbr``, ``Pupil Number``, ``Student ID Number``, ``StudentID``, ``STUDENTNUMBER``) or ``ID`` alone
    (``ID 512340``, ``ID Number 512340``); a person-ID header name of D10 (``Perm ID``, ``State ID``, ``SSID``); or a
    document label one word before (``IEP Meeting 1053``, ``504 Plan 1053``, ``IEP Report 160816``, ``Eval 2023
    160816``) when the number is no date or time (``natural``: ``IEP Due 1015``, ``IEP_Meeting_101524``, ``IEP
    Meeting 1330``)."""
    kind = _label_before_kind(s, start)
    return kind == "direct" or (kind == "gap" and not natural)


def person_label_after(s: str, start: int, end: int) -> bool:
    """K-01 (P-01): the number at ``start:end`` of ``s`` comes right before a person or document label
    (:data:`_AFTER_LABELS`: ``160816_IEP_Goals``, ``632084 Student``, ``325127_Evaluation``, ``160816_504_Plan``) and
    is not claimed by a word of a numbered thing before it (:data:`NUMBERED_WORDS`: ``Section_1014_Student_Roster``,
    ``Course 1014 Student List``)."""
    m = _TOKEN_AFTER.match(s, end)
    if m is None:
        return False
    tok = m.group(1)
    word = tok if tok.isdigit() else next((p.casefold() for p in _CAMEL_RE.split(tok) if p), "")
    if word not in _AFTER_LABELS:
        return False
    before = next((w for w in _words_before_number(s, start, 1) if w != "#"), "")
    return before not in NUMBERED_WORDS


def numbered_word_before(s: str, start: int) -> bool:
    """The letter word right before position ``start`` of ``s`` (or its last camelCase piece) is a word of a numbered
    thing (:data:`NUMBERED_WORDS`: ``Room_10101``, ``Q1014``, ``CourseNo 1050``): the number is that thing's."""
    label = _LABEL_BEFORE.search(s, 0, start)
    if label is None:
        return False
    return label.group(1).casefold() in NUMBERED_WORDS or _label_piece(label.group(1)) in NUMBERED_WORDS


def person_label(s: str, start: int, end: int, natural: bool = False) -> bool:
    """K-01: the number at ``start:end`` of ``s`` is labelled as a person's (:func:`person_label_before`,
    :func:`person_label_after`)."""
    return person_label_before(s, start, natural) or person_label_after(s, start, end)


#: B2: the chance factor of a person or document label elsewhere in a name (:func:`labelled_name`): such a name holds a
#: real ID on its own at a chance up to :data:`GROUP_CHANCE` (the limit of sibling names that share their text).
NEAR_LABEL_FACTOR = SINGLE_CHANCE / GROUP_CHANCE


def labelled_name(s: str, start: int, end: int) -> bool:
    """B2: the name ``s`` holds a person or document label away from its number at ``start:end`` (a singular word of
    :data:`_AFTER_LABELS`, also a camelCase piece: ``Student_Schedule_612714``, ``Student_Report_Card_612714``,
    ``IEP_Annual_Review_Notes_612714``, ``Evaluation_SY2425_612714``), and no word of a numbered thing that is no
    document word claims the number (``Course_612714_Student_Roster``, ``Student Count 612714``, ``Item 612714 IEP``;
    a document word does not: ``IEP_Progress_Report_612714``, ``Student_File_612714``)."""
    if not any(t in _AFTER_LABELS for t in _name_words(s[:start] + " " + s[end:])):
        return False
    label = _LABEL_BEFORE.search(s, 0, start)
    if label is None:
        return True
    claims = {label.group(1).casefold(), _label_piece(label.group(1))} & NUMBERED_WORDS
    return not claims - DOCUMENT_WORDS


@dataclass(frozen=True)
class IdMatch:
    """A real ID found in a name (PRIVATE): the matched text, its D8 key, the chance that an ordinary token of
    its shape equals a real ID (0: a letter-prefixed or zero-padded exact ID), and the name around it with the
    ID replaced by ``#`` (folded: the stub siblings share in a pivot)."""

    token: str
    key: str
    chance: float
    stub: str
    #: the chance before a per-number excuse (A1: weighed again when siblings repeat it, :func:`id_group_flags`)
    base: float = 1.0
    #: the per-number excuse that set :attr:`chance` to 1: ``natural`` (a time stamp, a short date or a school year),
    #: ``numbered`` (after a word of a numbered thing), ``context`` (a number alone under a numbered parent), or ``""``;
    #: B1: ``dated``, a date or time beside a label, keeps its chance but is a date among siblings that all read as
    #: dates (:func:`id_group_flags`)
    excuse: str = ""
    #: an exact zero-padded real ID (A2: :meth:`IdKeys.padded_chance`, :func:`id_group_flags`)
    padded: bool = False

    @property
    def strong(self) -> bool:
        """The match alone holds a real ID (a chance at most :data:`SINGLE_CHANCE`)."""
        return self.chance <= SINGLE_CHANCE


@dataclass
class IdKeys:
    """The real IDs of the run's ID namespaces (``generate.ids.IdNamespace``: raw values and D8 keys), row-number
    namespaces kept as 1..n (D9) left out. PRIVATE, in memory only."""

    namespaces: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._built: tuple[set[str], set[str], set[str], set[str], dict[int, float]] | None = None
        self._sorted: dict[int, list[int]] = {}
        self._sorted_all: dict[int, list[int]] = {}
        self._keyset: set[int] = set()
        self._padded_widths: Counter = Counter()
        self._per_namespace: list[tuple[frozenset[str], dict[int, float]]] | None = None
        #: A2 (K-18): the widths at which the run's names hold a zero-padded number that is no real ID (a county-
        #: district number ``015901`` beside the run's zero-padded IDs): only there is a zero-padded exact ID weighed by
        #: density (:meth:`note_run_names`; empty: every zero-padded exact ID is strong, as before K-18)
        self.padded_codes: frozenset[int] = frozenset()
        #: A8 (K-01): the D8 keys of real IDs that the run's names hold as a set too improbable for chance
        #: (:meth:`note_run_names`): a name holding one holds a real ID on its own
        self.evident: frozenset[str] = frozenset()
        self._date_shares: dict[str, float] = {}  # B1: :meth:`date_share` per number

    def __bool__(self) -> bool:
        return bool(self.namespaces)

    def _index(self) -> tuple[set[str], set[str], set[str], set[str], dict[int, float]]:
        """(raw IDs, letter-bearing D8 keys, digit keys (plain IDs and the digits of letter-prefixed ones),
        zero-padded raw IDs, chance per digit count) of the IDs of :func:`person_id_shape`."""
        if self._built is None:
            exact: set[str] = set()
            keys: set[str] = set()
            digit_keys: set[str] = set()
            padded: set[str] = set()
            for ns in self.namespaces:
                for v in getattr(ns, "raw_values", ()):  # every raw form (their D8 keys are derived here)
                    v = v.strip() if isinstance(v, str) else str(v).strip()
                    if v.isdigit() and v.isascii():  # fast path: plain digit IDs
                        if len(v) >= MIN_ID_LENGTH:
                            exact.add(v)
                            digit_keys.add(v.lstrip("0") or "0")
                            if v[0] == "0":
                                padded.add(v)
                        continue
                    if not person_id_shape(v):
                        continue
                    exact.add(v)
                    k = normalize_id(v)
                    if k.isdigit():
                        digit_keys.add(k)
                    else:
                        keys.add(k)
                        m = _DIGIT_CORE.fullmatch(k)
                        if m:
                            digit_keys.add(m.group(1))
            counts = Counter(len(k) for k in digit_keys)
            chance = {L: min(1.0, c / (9 * 10 ** (L - 1))) for L, c in counts.items()}
            self._sorted = {L: sorted(int(k) for k in digit_keys if len(k) == L) for L in LOCAL_WINDOW}
            self._keyset = {int(k) for k in digit_keys if len(k) in LOCAL_WINDOW}
            self._padded_widths = Counter(len(v) for v in padded)
            self._built = (exact, keys, digit_keys, padded, chance)
        return self._built

    def namespace_keys(self) -> list[tuple[frozenset[str], dict[int, float]]]:
        """(digit keys, density per digit count) of each namespace on its own (K-01: a pivot holds the IDs of ONE
        namespace): its plain IDs and the digits of its letter-prefixed ones by their D8 key, and the share of all
        numbers of each digit count that are its IDs. Built on first use only (a sibling set of numbers that are all
        real IDs)."""
        if self._per_namespace is None:
            out: list[tuple[frozenset[str], dict[int, float]]] = []
            for ns in self.namespaces:
                ks: set[str] = set()
                for v in getattr(ns, "raw_values", ()):
                    v = v.strip() if isinstance(v, str) else str(v).strip()
                    if v.isdigit() and v.isascii():
                        if len(v) >= MIN_ID_LENGTH:
                            ks.add(v.lstrip("0") or "0")
                        continue
                    if not person_id_shape(v):
                        continue
                    k = normalize_id(v)
                    if k.isdigit():
                        ks.add(k)
                    elif (m := _DIGIT_CORE.fullmatch(k)) is not None:
                        ks.add(m.group(1))
                counts = Counter(len(k) for k in ks)
                out.append((frozenset(ks), {L: min(1.0, c / (9 * 10 ** (L - 1))) for L, c in counts.items()}))
            self._per_namespace = out
        return self._per_namespace

    def padded_chance(self, d: str) -> float:
        """K-18: the chance that an ordinary zero-padded number ``d`` (``015915``: a county-district number) equals a
        zero-padded real ID of its width: the share of the numbers of that width with a leading zero that are one
        (30,000 zero-padded six-digit IDs 000001-030000 leave nearly a third of them real IDs). A2: it applies only
        where the run's names show that zero-padded numbers of that width are an ordinary code (:attr:`padded_codes`);
        elsewhere a zero-padded exact ID is strong (:meth:`matches`)."""
        self._index()
        return min(1.0, self._padded_widths.get(len(d), 0) / 10 ** (len(d) - 1))

    def padded_code(self, d: str) -> bool:
        """A2: ``d`` is a zero-padded number of a width the run's zero-padded IDs have that is no real ID (a county-
        district number ``015901``; a time or a date too: it is a code of the same width that is no ID)."""
        if not d or d[0] != "0" or not d.isascii() or not d.isdigit():
            return False
        key = d.lstrip("0")
        self._index()
        return len(d) in self._padded_widths and len(key) >= MIN_ID_LENGTH and key not in self._built[2]

    def local_share(self, key: str) -> float:
        """A3: the share of the numbers around ``key`` (of its digit count, ``key`` itself left out) that are real
        IDs: a window of 100 numbers for 4 digits, 1,000 for 5, 10,000 for 6 and so on, kept inside that digit count
        (ZIP codes 30303-30312 inside 28,000 sequential IDs 10001-38000: about 1; any number among 6,000 IDs spread
        over every six-digit number: about 0.007)."""
        if not key.isdigit() or len(key) < MIN_ID_LENGTH:
            return 1.0
        _, _, digit_keys, _, _ = self._index()
        L = len(key)
        arr = self._sorted_all.get(L)
        if arr is None:
            arr = self._sorted_all[L] = sorted(int(k) for k in digit_keys if len(k) == L)
        x, w = int(key), 5 * 10 ** (L - 3)
        lo, hi = max(10 ** (L - 1), x - w), min(10 ** L - 1, x + w)
        near = bisect.bisect_right(arr, hi) - bisect.bisect_left(arr, lo) - (key in digit_keys)
        return min(1.0, near / max(1, hi - lo))

    def number_chance(self, key: str) -> float:
        """The chance that an ordinary number of ``key``'s digits equals a real ID: the share of such numbers
        that are real IDs, and for short IDs (:data:`LOCAL_WINDOW`) at least the share of numbers around
        ``key`` that are (sequential 4-digit student numbers 1001-1400 leave nearly every number of that
        range a real ID: a room ``1375`` beside them is no evidence)."""
        _, _, _, _, chance = self._index()
        L = len(key)
        ch = chance.get(L, 1.0)
        w = LOCAL_WINDOW.get(L)
        arr = self._sorted.get(L)
        if w and arr and key.isdigit():
            x = int(key)
            near = bisect.bisect_right(arr, x + w) - bisect.bisect_left(arr, x - w)
            ch = max(ch, (near - (x in self._keyset)) / (2 * w))  # the share among its neighbours
        return min(1.0, ch)

    def local_keys(self, raw_values: Iterable[Any]) -> frozenset[str]:
        """D8 keys (and the digits of letter-prefixed IDs) of ``raw_values``, the IDs one file, folder or sheet
        holds (:meth:`matches` ``local``)."""
        out: set[str] = set()
        for v in raw_values:
            if not isinstance(v, str):  # a native null (the ID registry reads text values only)
                continue
            v = v.strip()
            if not v or not person_id_shape(v):
                continue
            k = normalize_id(v)
            out.add(k)
            if not k.isdigit():
                m = _DIGIT_CORE.fullmatch(k)
                if m:
                    out.add(m.group(1))
        return frozenset(out)

    def matches(self, text: str, local: frozenset[str] | None = None, context: bool = False) -> list[IdMatch]:
        """Every real ID in ``text`` (module docstring, check 3), weighed by chance.

        Letter-bearing parts (``S482913``, ``s-482913``, the ``S482913`` of ``S482913math`` or ``mathS482913``)
        match exactly or by their D8 key: chance 0. Digit runs match a plain ID or the digits of a prefixed one
        by their D8 key (leading zeros): chance :meth:`number_chance`, 0 for a labelled ID (K-01), a zero-padded exact
        ID (A2: unless the run's names use zero-padded numbers of its width as an ordinary code, :attr:`padded_codes`,
        then :meth:`padded_chance`) or an ID the run's names prove (A8: :attr:`evident`), 1 for a time stamp, a short
        date or a numbered thing (:func:`natural_number`, :data:`NUMBERED_WORDS`; :attr:`IdMatch.excuse`, with the
        chance before it as :attr:`IdMatch.base`), at least :data:`SHORT_NUMBER_CHANCE` for fewer than 5 digits
        inside a longer name. Beside a label (B1, B2): a school year is excused (siblings that are no dates revive it);
        a date or time as people write them in names (:func:`name_date`) right before a label, one word after a document
        label or with a label elsewhere in the name is weighed by at least :meth:`date_share` times
        :data:`NEAR_LABEL_FACTOR` (excuse ``dated``); any other number with a label elsewhere in the name
        (:func:`labelled_name`) by its chance times :data:`NEAR_LABEL_FACTOR`. ``local``: the IDs the named
        file, folder or sheet itself holds (:meth:`local_keys`); a number equal to one of them has at most the
        chance that an ordinary number is one of those few IDs (a per-student file ``IEP_496487.csv`` whose ID
        column holds student 496487 only). Years and dates never match; digits are never glued across
        separators (``9/3/2024`` is not ``932024``). ``context``: the parent name says what its children's numbers
        are (:func:`numbered_context`): a name that is a number alone has chance 1."""
        if not self.namespaces:
            return []
        s = clean_text(text).strip()
        # an ID here has at least MIN_ID_LENGTH digits in one run (person_id_shape): nothing else is looked up
        if len(s) < MIN_ID_LENGTH or not any(len(d) >= MIN_ID_LENGTH for d in _DIGIT_RUN.findall(s)):
            return []
        exact, keys, digit_keys, padded, _ = self._index()
        if not exact:
            return []
        out: list[IdMatch] = []
        seen: set[tuple[str, int]] = set()
        local_counts = Counter(len(k) for k in local if k.isdigit()) if local else Counter()

        def add(tok: str, key: str, ch: float, start: int, end: int, base: float | None = None, excuse: str = "",
                is_padded: bool = False) -> None:
            if (key, start) in seen:
                return
            seen.add((key, start))
            base = ch if base is None else base
            if local and key in local and key.isdigit():
                lc = local_counts[len(key)] / (9 * 10 ** (len(key) - 1))
                if lc <= SINGLE_CHANCE:  # one of the few IDs the named file itself holds
                    ch, base, excuse = min(ch, lc), min(base, lc), ""
            out.append(IdMatch(tok, key, ch, (s[:start] + "#" + s[end:]).casefold(), base, excuse, is_padded))

        # letter-bearing parts and their letter+digit pieces: exact or D8 key
        if any(ch.isalpha() for ch in s) and keys:
            for rx in (_ID_SPLIT, _ID_SPLIT_DASH):
                pos = 0
                for part in rx.split(s):
                    start = s.find(part, pos)
                    pos = start + len(part)
                    if not part or not any(ch.isdigit() for ch in part) or not any(ch.isalpha() for ch in part):
                        continue
                    pieces = [(part, start)]
                    for m in re.finditer(r"([A-Za-z]+)([0-9]+)", part):
                        letters = _CAMEL_RE.split(m.group(1))
                        for k in range(len(letters)):
                            lead = "".join(letters[k:])
                            pieces.append((lead + m.group(2), start + m.end(2) - len(m.group(2)) - len(lead)))
                    for cand, c_start in pieces:
                        key = normalize_id(cand)
                        if (cand in exact or key in keys) and not key.isdigit():
                            add(cand, key, 0.0, c_start, c_start + len(cand))
        # digit runs: plain IDs and the digits of prefixed ones
        for m in _DIGIT_RUN.finditer(s):
            d = m.group()
            key = d.lstrip("0")
            if len(key) < MIN_ID_LENGTH or key not in digit_keys or (len(d) == 4 and _year(d)):
                continue  # a 4-digit year is never an ID, labelled or not
            # K-01: right after a person or document label ('direct': 'Student 612587', 'IEP_512340'), right before
            # one or one word after a document label ('tied': '160816_IEP_Goals', 'IEP Meeting 612587'); B2: a label
            # elsewhere in the name ('near': 'Student_Schedule_612714')
            before = _label_before_kind(s, m.start())
            tied = before or ("after" if person_label_after(s, m.start(), m.end()) else "")
            near = not tied and labelled_name(s, m.start(), m.end())
            # B1, B2: beside a label, a number is a date only as people write dates in names (no bare time with seconds,
            # no day that is no day of the calendar) or as the time of a date-time stamp
            dated = bool(tied or near) and (name_date(d) or _stamp_time(s, m.start(), d))
            # years and dates are no IDs without a label (P-07: 'Student_200105', 'IEP_20210915' are)
            if date_like(d) and not (tied or near):
                continue
            whole = m.start() == 0 and m.end() == len(s)
            school = school_year(d)

            def weighed() -> float:
                # K-18: a zero-padded exact ID is weighed by the share of padded numbers of its width that are real IDs
                ch = self.padded_chance(d) if d in padded else self.number_chance(key)
                return max(ch, SHORT_NUMBER_CHANCE) if not whole and len(key) < 5 else ch

            if whole and context:  # the parent says what its children's numbers are (a number alone has no label)
                base, excuse = self.number_chance(key), "context"
            elif key in self.evident:  # A8: a real ID the run's names hold among others too many to be chance
                base, excuse = 0.0, ""
            elif tied == "direct" and not school:
                # K-01: 'Student 612587', 'IEP_512340', 'Student_160816_IEP_Goals': whatever the density, also when it
                # reads as a date (P-07: 'Student_200105', 'IEP_20210915')
                base, excuse = 0.0, ""
            elif school and (tied or near):
                # B1: a school year is never the label's ID ('IEP_2425', 'IEP_Caseload_2425', '2425_Student_Roster',
                # 'IEP_202425'), unless a sibling that is no date makes them all the person's (:func:`id_group_flags`)
                base, excuse = (0.0 if tied else weighed() * NEAR_LABEL_FACTOR), "natural"
            elif dated:
                # B1: a date or a time beside a label ('101524_Student_Attendance', 'IEP_Meeting_1015',
                # '1330_IEP_Meeting', 'Student_Attendance_20241015') is weighed by the chance that a date of its layout
                # is a real ID (at least the share of real IDs among the dates of its layout around it: a district
                # numbering its students by entry year makes 20241015 one as often as not), with the label's factor;
                # siblings that all read as dates are dates (excuse 'dated', :func:`id_group_flags`)
                base, excuse = max(weighed(), self.date_share(d)) * NEAR_LABEL_FACTOR, "dated"
            elif tied:  # K-01 (P-01, P-02): '160816_IEP_Goals', '632084 Student', 'IEP Meeting 612587'
                base, excuse = 0.0, ""
            elif d in padded and len(d) not in self.padded_codes and not _stamp_time(s, m.start(), d):
                # A2: a zero-padded exact ID where the run's names hold no zero-padded number of that width that is no
                # real ID: the padding is the ID's (as before K-18), whatever word comes before it; never the time of a
                # date-time stamp (``students_20241015_013005``)
                base, excuse = 0.0, ""
            elif near:  # B2: a label elsewhere in the name holds a real ID at a chance up to GROUP_CHANCE
                base, excuse = weighed() * NEAR_LABEL_FACTOR, ""
            else:
                base = weighed()
                excuse = ("natural" if natural_number(d) else "numbered" if numbered_word_before(s, m.start())
                          else "")
            add(d, key, base if excuse in ("", "dated") else 1.0, m.start(), m.end(), base, excuse, d in padded)
        return out

    def date_share(self, d: str) -> float:
        """B1: the share of real IDs among the numbers that read like ``d`` as a date or a clock time near it (``d``
        itself left out), the largest over the layouts it reads in (:func:`name_date`): the days of the two years before
        and after it written the same way (``YYYYMMDD``, ``MMDDYYYY``, ``DDMMYYYY``, ``MMDDYY``, ``YYMMDD``), every day
        of the year (``MMDD``), the months of the two years around it (``YYYYMM``) or every five-minute mark of the day
        (``HHMM``). A district numbering its students by entry year and sequence (``20240001``-``20241500``) makes
        nearly every date of a year a real ID; 6,000 IDs spread over every six-digit number make one date in 150 one."""
        got = self._date_shares.get(d)
        if got is not None:
            return got
        _, _, digit_keys, _, _ = self._index()
        best = 0.0
        for near in _date_neighbours(d):
            if near:
                best = max(best, sum(1 for v in near if v.lstrip("0") in digit_keys) / len(near))
        self._date_shares[d] = best
        return best

    def match(self, text: str, local: frozenset[str] | None = None) -> bool:
        """``text`` alone holds a real ID (a match of chance at most :data:`SINGLE_CHANCE`)."""
        return any(m.strong for m in self.matches(text, local))

    def note_run_names(self, names: Iterable[tuple[str, frozenset[str] | None, bool]]) -> None:
        """Run-wide evidence from every name the run would write (``(text, local IDs, numbered parent)``: folders, file
        names, sheet names and headers, decoded; :func:`screen_run` before its steps 1-3):

        * A2 (K-18): :attr:`padded_codes`, the widths at which a name holds a zero-padded number that is no real ID
          (:meth:`padded_code`; ``015901_Alamo_Heights_ISD_TAPR`` beside ``015907_San_Antonio_ISD_TAPR``): only at
          those widths is a zero-padded exact ID weighed by density, elsewhere it is strong;
        * A8 (K-01): :attr:`evident`, the real IDs that the run's names hold among numbers of one digit count where
          chance cannot explain so many (the sibling check of :func:`id_group_flags` run over the whole run, so
          per-student documents one per folder count together: ``transcripts/Transcript_373208.pdf``,
          ``behavior/BIP_737720.pdf``, ``health/Immunization_Record_227588.pdf`` beside 6,000 six-digit IDs): three
          distinct real IDs or more, each weak alone (a chance above :data:`SINGLE_CHANCE`; a number after a word of
          a numbered thing by its chance before that excuse, ``records/Cumulative_Folder_169858.pdf``; never a time
          stamp, a short date or a number alone under a numbered parent), of the ``n`` distinct numbers of that digit
          count in the run's names with ``C(n, m)`` times the product of their chances (each at least the share of
          real IDs around it, :meth:`local_share`: rooms and courses inside a range of sequential staff IDs are
          chance) at most :data:`ANY_GROUP_CHANCE`. A number that equals a real ID by chance among many such numbers
          (2,000 item headers, a folder of report numbers) never adds up."""
        names = [(decoded(t), local, ctx) for t, local, ctx in names if t]
        if not self.namespaces or not names:
            return
        self._index()
        if self._padded_widths:
            self.padded_codes = frozenset(len(d) for t, _, _ in names for d in _DIGIT_RUN.findall(clean_text(t))
                                          if self.padded_code(d))
        best: dict[str, float] = {}
        numbers: set[str] = set()
        for t, local, ctx in names:
            nums = _number_stubs(t)
            if not nums:
                continue
            numbers.update(nums.values())
            for m in self.matches(t, local, ctx):
                c = m.base if m.excuse == "numbered" else m.chance  # A1: a numbered word gives way to evidence too
                if SINGLE_CHANCE < c < 1.0 and m.key.isdigit() and m.excuse != "dated":  # B1: a labelled date is none
                    # never fewer real IDs than around the number: rooms and courses inside a range of sequential IDs
                    c = max(c, self.local_share(m.key))
                    if c < 1.0:
                        best[m.key] = min(best.get(m.key, 1.0), c)
        evident: set[str] = set()
        by_length: dict[int, dict[str, float]] = {}
        for k, c in best.items():
            by_length.setdefault(len(k), {})[k] = c
        for L, got in by_length.items():
            n = sum(1 for k in numbers | got.keys() if len(k) == L)
            if len(got) >= EVIDENT_MIN and _tail_chance(n, list(got.values())) <= ANY_GROUP_CHANCE:
                evident |= got.keys()
        self.evident = frozenset(evident)


def _stamp_time(s: str, start: int, d: str) -> bool:
    """A2: the digit run ``d`` at position ``start`` of ``s`` is the time of a date-time stamp: it reads as a time
    (``HHMMSS``, :func:`natural_number`) right after a date (``YYYYMMDD``, ``YYYYMM``: :func:`date_like`), separators
    only between (``students_20241015_013005``, ``export-20241015T0130``)."""
    if not natural_number(d):
        return False
    m = re.search(r"(?<![0-9])([0-9]{6,8})[\s_.\-T]*$", s[:start])
    return m is not None and date_like(m.group(1))


def _number_stubs(text: str) -> dict[str, str]:
    """Every number of ``text`` that could be an ID (folded stub -> its D8 key): the stub (:class:`IdMatch`) is
    the text with that number replaced by ``#``."""
    s = clean_text(text).strip()
    return {(s[:m.start()] + "#" + s[m.end():]).casefold(): m.group().lstrip("0") for m in _DIGIT_RUN.finditer(s)
            if len(m.group().lstrip("0")) >= MIN_ID_LENGTH and not date_like(m.group())}


def _tail_chance(n: int, chances: list[float]) -> float:
    """The chance that ``len(chances)`` of ``n`` ordinary numbers are distinct real IDs by chance, bounded by
    ``C(n, m)`` times the product of the ``m`` chances (a union bound; in logs, so 2,000 headers do not
    overflow)."""
    if any(c <= 0.0 for c in chances):
        return 0.0
    m = len(chances)
    n = max(n, m)
    log = math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1) + sum(math.log(c) for c in chances)
    return 1.0 if log >= 0 else math.exp(log)


def id_group_flags(ids: IdKeys | None, texts: list[str], local: list[frozenset[str] | None] | None = None,
                   context: bool = False, label: bool = False) -> dict[int, tuple[str, ...]]:
    """Sibling names (one table's headers, one workbook's sheets, one folder's files and folders) that hold a
    real ID, with the ID tokens they hold (PRIVATE): alone (:attr:`IdMatch.strong`; ``local``: the IDs each named
    file, folder or sheet holds), or by repeated evidence: siblings that share the same text around a number
    (``score#``, ``IEP_#``) where ``m`` of the ``n`` numbers are distinct real IDs with a chance (the sum over
    every choice of ``m`` of the ``n``, bounded by ``C(n, m)`` times the product of the ``m`` chances:
    :func:`_tail_chance`) at most :data:`GROUP_CHANCE` (two per-student files ``IEP_387347``, ``IEP_405159``
    beside 50,000 six-digit IDs: 0.056 x 0.056); three distinct IDs or more among all the numbers of the
    siblings with such a chance at most :data:`ANY_GROUP_CHANCE` (2,000 item headers ``Q17_482913`` of which a
    hundred equal a real ID by chance are no evidence); a pivot by person (:func:`pivot_flags`).
    ``context``: the parent name says what its children's numbers are (:func:`numbered_context`). ``label`` (K-01,
    P-05): the parent name holds a person or document label (:func:`person_context`: ``iep/``, ``iep_progress.xlsx``):
    a sibling named by a number alone that is a real ID holds it whatever the density, unless every such sibling
    reads as a date or a time (``student_attendance_oct.xlsx`` with a sheet per school day) or they count up in one
    constant step (a numbered list); a 4-digit year is never an ID. A1: in the shared-text check a numbered word's
    excuse gives way to the siblings' evidence (``Progress_Report_#``); A2: a zero-padded exact ID weighed by density
    because the run uses zero-padded numbers of its width as a code elsewhere is strong among siblings that hold no
    such code (a folder of per-student goal exports)."""
    if not ids:
        return {}
    per = {i: ids.matches(t, local[i] if local else None, context) for i, t in enumerate(texts)}
    per = {i: ms for i, ms in per.items() if ms}
    if ids.padded_codes and any(m.padded and len(m.token) in ids.padded_codes for ms in per.values() for m in ms):
        # A2: zero-padded numbers of a width the run uses as an ordinary code elsewhere (a folder of county-district
        # numbers), but no sibling here holds one that is no real ID: the padding is the IDs' (per-student files)
        sib_codes = {len(d) for t in texts for d in _DIGIT_RUN.findall(clean_text(t)) if ids.padded_code(d)}
        per = {i: [replace(m, chance=0.0, base=0.0, excuse="")
                   if m.padded and m.excuse != "context" and len(m.token) not in sib_codes else m for m in ms]
               for i, ms in per.items()}
    # B1: siblings sharing the text around a number that all read as dates or times (every number of the stub, real ID
    # or not: a folder of daily exports ``MMDDYY_Student_Attendance.csv``) are dated names, whatever a label says
    stub_dates: dict[str, bool] = {}
    if any(m.excuse == "dated" for ms in per.values() for m in ms):
        stub_nums: dict[str, list[str]] = {}
        for t in texts:
            s = clean_text(t).strip()
            for m in _DIGIT_RUN.finditer(s):
                if len(m.group()) >= MIN_ID_LENGTH:
                    stub_nums.setdefault((s[:m.start()] + "#" + s[m.end():]).casefold(), []).append(m.group())
        stub_dates = {stub: len(nums) >= 2 and all(name_date(d) for d in nums) for stub, nums in stub_nums.items()}
    per = {i: [replace(m, chance=1.0) if m.excuse == "dated" and stub_dates.get(m.stub) else m for m in ms]
           for i, ms in per.items()}
    flagged: dict[int, set[str]] = {}
    for i, ms in per.items():
        strong = [m.token for m in ms if m.strong]
        if strong:
            flagged.setdefault(i, set()).update(strong)
    by_stub: dict[str, list[tuple[int, IdMatch]]] = {}
    for i, ms in per.items():
        for m in ms:
            by_stub.setdefault(m.stub, []).append((i, m))
    stub_count: Counter = Counter()
    all_numbers: set[str] = set()
    if per:
        for t in texts:
            nums = _number_stubs(t)
            all_numbers.update(nums.values())
            for stub in nums.keys() & by_stub.keys():
                stub_count[stub] += 1
    for stub, items in by_stub.items():
        # A1: a per-number excuse gives way to its siblings' evidence: numbers after a word of a numbered thing
        # (``Progress_Report_#``, ``Lunch_Account_#``, ``Case_#_Notes``) are weighed by their chance before it, at the
        # stricter limit of :data:`ANY_GROUP_CHANCE` (the word says what they are); so are numbers that read as a time
        # or a short date when not all of them do (one that does not makes none of them a date)
        dated = all(m.excuse in ("natural", "dated") for _, m in items) and (
            stub_dates.get(stub, True) or not any(m.excuse == "dated" for _, m in items))
        best: dict[str, float] = {}
        strict = False
        for _, m in items:
            c = m.chance
            if m.excuse == "numbered":  # at least the share of real IDs around it: rooms inside sequential staff IDs
                c = max(m.base, ids.local_share(m.key))
                strict = True
            elif m.excuse in ("natural", "dated") and not dated:
                c = m.base
            best[m.key] = min(best.get(m.key, 1.0), c)
        n_cands = max(stub_count[stub], len({i for i, _ in items}))
        if len(best) >= 2 and _tail_chance(n_cands, list(best.values())) <= (ANY_GROUP_CHANCE if strict
                                                                              else GROUP_CHANCE):
            for i, m in items:
                flagged.setdefault(i, set()).add(m.token)
    best_any: dict[str, float] = {}
    holders: dict[str, list[tuple[int, IdMatch]]] = {}
    for i, ms in per.items():
        for m in ms:
            if m.chance < 1.0:
                best_any[m.key] = min(best_any.get(m.key, 1.0), m.chance)
                holders.setdefault(m.key, []).append((i, m))
    if len(best_any) >= 3 and _tail_chance(len(all_numbers | best_any.keys()),
                                           list(best_any.values())) <= ANY_GROUP_CHANCE:
        for items in holders.values():
            for i, m in items:
                flagged.setdefault(i, set()).add(m.token)
    for i, toks in pivot_flags(ids, texts, context).items():
        flagged.setdefault(i, set()).update(toks)
    if label and not context:
        for i, tok in _labelled_children(ids, texts).items():
            flagged.setdefault(i, set()).add(tok)
    return {i: tuple(sorted(toks)) for i, toks in flagged.items()}


def _bare_number(text: str) -> str | None:
    """The digits of a name that is a number alone (``1053``, `` 1053 ``), else None."""
    s = clean_text(text).strip()
    return s if s and s.isascii() and s.isdigit() else None


def _schedule_times(numbers: Iterable[str]) -> bool:
    """Every number reads as a clock time ``HHMM`` on a five-minute mark (``1235``, ``1305``, ``1330``: a schedule of
    lunch periods or meetings as columns or sheets): sibling numbers that are such times are times, not people (a set
    of student numbers that all do so is rare: one in eight 4-digit numbers from 1000 to 2359 is such a time)."""
    ds = list(numbers)
    return bool(ds) and all(len(d) == 4 and d.isdigit() and int(d[:2]) <= 23 and int(d[2:]) <= 59
                            and int(d[2:]) % 5 == 0 for d in ds)


#: The fewest numbers that show a constant step (two numbers always differ by one step: no evidence of a list).
SEQUENCE_MIN = 3


def _counts_up(keys: Iterable[str]) -> bool:
    """Three or more numbers counting up in one constant step (``1001,1002,1003``, ``1010,1020,1030``): a numbered
    list (assignment, item, band or room numbers), never a set of people."""
    ints = sorted({int(k) for k in keys})
    return len(ints) >= SEQUENCE_MIN and len({b - a for a, b in zip(ints, ints[1:])}) == 1


def _labelled_children(ids: IdKeys, texts: list[str]) -> dict[int, str]:
    """K-01 (P-05): the siblings named by a number alone that are real IDs, under a parent with a person or document
    label (:func:`id_group_flags` ``label``)."""
    bare = {i: d for i, t in enumerate(texts) if (d := _bare_number(t)) is not None
            and len(d.lstrip("0")) >= MIN_ID_LENGTH}
    if not bare or all(natural_number(d) or date_like(d) for d in bare.values()) \
            or _schedule_times(bare.values()) or _counts_up(d.lstrip("0") for d in bare.values()):
        return {}
    digit_keys = ids._index()[2]
    return {i: d for i, d in bare.items() if d.lstrip("0") in digit_keys and not (len(d) == 4 and date_like(d))}


#: D37: the fewest real IDs of one namespace among sibling names that make a pivot by person (a pair of columns, sheets,
#: files or folders named by two students' IDs), whatever the density of the namespace.
PIVOT_MIN = 2
#: B5, D37: the fewest real IDs among sibling names whose words around the number differ (surnames or subjects:
#: ``Garcia (1004)``, ``Nguyen (1023)``, ``Okafor (1293)``) that make a pivot: two differently worded names holding a
#: real ID each (``IEP_482913``, ``Roster_483000``; county-district numbers ``015907_Northside_ISD_TAPR``,
#: ``015915_Judson_ISD_TAPR``) are no set.
WORDED_PIVOT_MIN = 3


def mostly_real_ids(n_real: int, n_numbers: int, least: int = PIVOT_MIN) -> bool:
    """D37: ``n_real`` real IDs of one namespace among the ``n_numbers`` distinct numbers of a set of sibling names
    make a pivot by person: at least ``least`` of them (:data:`PIVOT_MIN`), and the numbers that are no such ID
    (withdrawn students, a year column) at most one or at most a quarter of the set (``max(1, n // 4)``): 2 of 3, 3 of
    4, 6 of 8, 22 of 26 are a pivot; 2 of 4, 3 of 5, 12 of 20 are not (numbered things that only partly fall in a range
    of IDs)."""
    return n_real >= least and n_numbers - n_real <= max(1, n_numbers // 4)


def pivot_flags(ids: IdKeys | None, texts: list[str], context: bool = False) -> dict[int, set[str]]:
    """K-01, D37: sibling names (one table's headers ``Skill,1180,1349,...``, one workbook's sheets, one folder's files
    and folders) that share the same text around a number (the stub: ``#`` for a number alone; P-03: ``score#`` of
    Stata's ``reshape wide``, ``_#`` of a .dta numeric name, ``#_score``; B5: text that differs only in runs of at most
    :data:`INITIALS_MAX` letters, initials: ``FK (1004)``, ``GB (1023)``) and whose numbers are real IDs of ONE
    namespace or of the run's IDs together (P-04: per-school rosters) are a pivot by person when
    :func:`mostly_real_ids` holds (at least two real IDs; a few numbers that are no such ID, withdrawn students'
    columns, do not excuse the rest), whatever the density of the namespace (D37, user 2026-10-11: a reasonable person
    could identify a student from such names; headers that are IDs are rare, so the file is flagged and skipped;
    ``--keep-name`` overrides, one per name). This replaces the one-in-a-million chance threshold of K-01. Sibling
    names whose text differs in longer words (B5: surnames, ``Garcia (1004)``, ``Nguyen (1023)``) are a set too, of at
    least :data:`WORDED_PIVOT_MIN` real IDs. Still excused:

    * numbers that ALL read as a year, school year, date or clock time (:func:`date_like`, :func:`natural_number`,
      :func:`_schedule_times`: ``2019 ... 2025``, ``2324``, ``1006`` = October 6, lunch periods ``1235,1305,1330``);
      one that does not makes none of them a date (P-05: ``1007`` beside ``1053`` and ``1081`` is a student);
    * numbers counting up in one constant step (three or more: ``1001,1002,1003``, ``1050,1100,1150``: a numbered list
      of assignments, items, bands or rooms; :func:`_counts_up`);
    * numbers after a word of a numbered thing (:data:`NUMBERED_WORDS`: ``Q1014``, ``Item_1135``, ``Room 1104``): they
      never join a set;
    * a numbered parent context (``context``, :func:`numbered_context`: ZIP sheets of ``enrollment_by_zip.xlsx``,
      ``Rooms/``), where each number is weighed by at least the share of the numbers around it that are real IDs
      (:meth:`IdKeys.local_share`) and the set is excused while chance explains it (``C(n, m)`` times the product of
      those chances above :data:`ANY_GROUP_CHANCE`). A3: 26 real IDs spread like the run's 6,000 IDs over every
      six-digit number are a pivot whatever the parent says (``Soc_Studies_Gr8_Gradebook``, ``SOC_Projects/``: ``soc``
      also abbreviates Social Studies).

    D37 drops B4 (numbers ending in two digits too low for a sample of the namespace's IDs: rooms and courses numbered
    within blocks): such numbers that are all real IDs are a pivot unless they count up in one step or a numbered parent
    names them. (index -> numbers as written) of the flagged names."""
    if not ids:
        return {}
    groups: dict[str, list[tuple[int, str]]] = {}
    initials: dict[str, dict[str, list[tuple[int, str]]]] = {}
    worded: dict[str, dict[str, list[tuple[int, str]]]] = {}
    for i, t in enumerate(texts):
        s = clean_text(t).strip()
        for m in _DIGIT_RUN.finditer(s):
            if not numbered_word_before(s, m.start()):
                stub = (s[:m.start()] + "#" + s[m.end():]).casefold()
                item = (i, m.group())
                groups.setdefault(stub, []).append(item)
                # B5: the same text but for its short letter runs (a student's initials around the ID: 'FK (1004)',
                # 'GB (1023)'), or for its words (surnames: 'Garcia (1004)', 'Nguyen (1023)')
                initials.setdefault(_initials_shape(stub), {}).setdefault(stub, []).append(item)
                worded.setdefault(_LETTER_RUN.sub("a", stub), {}).setdefault(stub, []).append(item)
    sets = [(items, PIVOT_MIN) for items in groups.values()]
    for shapes, least in ((initials, PIVOT_MIN), (worded, WORDED_PIVOT_MIN)):
        # a shape of one stub only is that stub's group already
        sets += [([item for items in by_stub.values() for item in items], least)
                 for by_stub in shapes.values() if len(by_stub) > 1]
    candidates: list[tuple[frozenset[str] | set[str], dict[int, float]]] | None = None
    out: dict[int, set[str]] = {}
    for items, least in sets:
        if len(items) < least:
            continue
        keys = {d.lstrip("0") or "0" for _, d in items}
        long_keys = {k for k in keys if len(k) >= MIN_ID_LENGTH}
        if len(long_keys) < least or _counts_up(keys):
            continue  # too few numbers that could be IDs, or one constant step: a numbered list
        if all(natural_number(d) or date_like(d) for _, d in items) or _schedule_times(d for _, d in items):
            continue  # dates, school years and clock times
        if candidates is None:
            _, _, digit_keys, _, density = ids._index()
            candidates = [(digit_keys, density), *ids.namespace_keys()]
        for ns_keys, ns_density in candidates:
            real = long_keys & ns_keys
            if not mostly_real_ids(len(real), len(keys), least):
                continue
            if context and _tail_chance(len(keys), [max(ns_density.get(len(k), 1.0), ids.local_share(k))
                                                    for k in real]) > ANY_GROUP_CHANCE:
                continue  # A3: the parent says what the numbers are, and chance explains them
            for i, d in items:
                if (d.lstrip("0") or "0") in real:
                    out.setdefault(i, set()).add(d)
            break
    return out


#: B5: a run of letters (the text of a stub that a pivot's names may vary: a student's initials).
_LETTER_RUN = re.compile(r"[^\W\d_]+")
#: B5, D37: the most letters of a run that a pivot's names may vary (initials: ``FK``, ``JAS``).
INITIALS_MAX = 3


def _initials_shape(stub: str) -> str:
    """B5: ``stub`` with each letter run of at most :data:`INITIALS_MAX` letters replaced by ``a`` (``fk (#)`` and
    ``gb (#)`` are ``a (#)``; ``iep_#`` and ``roster_#`` stay apart)."""
    return _LETTER_RUN.sub(lambda m: "a" if len(m.group()) <= INITIALS_MAX else m.group(), stub)


def run_id_keys(files: Iterable[FileResult], registry: Any = None) -> IdKeys:
    """:class:`IdKeys` of the run: the namespaces of ``registry`` (the run's ``generate.ids.IdRegistry`` built
    by ``pipeline.plan_ids``, which registers files that had not FAILed), plus the ID columns of every other
    file with tables (a FAILed file's IDs are real IDs too) registered in a scratch registry, plus every column
    that a flag turned from an ID column into something else (R6-02: ``--exclude student_id``, ``--null``,
    ``--force-*``; ``profile.entity.shadow_namespaces``): its real IDs are real IDs of the run, whatever it
    writes."""
    from ..generate.ids import IdRegistry, namespace_is_identity
    from ..profile import entity as entity_mod

    known: set[tuple[str, str | None]] = set()
    spaces: list[Any] = []
    if registry is not None:
        for ns in registry.namespaces.values():
            known.update((m[0], m[1]) for m in ns.members)
            if not ns.identity and ns.raw_values:
                spaces.append(ns)
    scratch = IdRegistry()
    for fr in files:
        for tp in fr.tables:
            if (fr.rel_path, tp.name) in known:
                continue
            for i in tp.entity.id_columns:
                raw = tp.private.id_raw.get(i)
                if raw is None:
                    continue
                key = entity_mod.id_namespace(fr.rel_path, tp.name, tp.columns[i])
                scratch.register(key, fr.rel_path, tp.name, i, tp.columns[i].name, raw)
    spaces += [ns for ns in scratch.namespaces.values() if ns.raw_values and not namespace_is_identity(ns)]
    spaces += entity_mod.shadow_namespaces(files)
    return IdKeys(spaces)


# --------------------------------------------------------------------------- the screen
def url_decoded(text: str) -> str:
    """``text`` URL-decoded (``%20`` and ``+`` as a space: ``Kaden%20Yazzie%20IEP`` is ``Kaden Yazzie IEP``), or
    ``text`` itself when it holds no escape."""
    if "+" not in text and not _URL_ESCAPE.search(text):
        return text
    return urllib.parse.unquote_plus(text)


_URL_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
#: K-10: an OOXML escape of one character (``_x0020_``: Excel, SharePoint and OData write a space in a name so).
_OOXML_ESCAPE = re.compile(r"_[xX]([0-9A-Fa-f]{4})_")
#: K-10: an HTML character reference (``&nbsp;``, ``&#32;``, ``&#x20;``, ``&amp;``), with its closing ``;``.
_HTML_REFERENCE = re.compile(r"&(?:#[0-9]{1,7}|#[xX][0-9A-Fa-f]{1,6}|[A-Za-z][A-Za-z0-9]{1,31});")
#: K-10: digits glued between two letter words (``Ana1Garcia``): a separator. A single letter before or after is no
#: word (``P1Garcia``: a period label, like ``Garcia P1``).
_GLUED_DIGITS = re.compile(r"(?<=[^\W\d_]{2})\d+(?=[^\W\d_]{2})")


def _ooxml_char(m: re.Match) -> str:
    cp = int(m.group(1), 16)
    return m.group() if 0xD800 <= cp <= 0xDFFF else chr(cp)


def decoded(text: str) -> str:
    """``text`` with its escapes decoded (K-10), or ``text`` itself when it holds none: URL escapes (``%20`` and ``+``,
    :func:`url_decoded`), OOXML ``_xHHHH_`` escapes (``Ana_x0020_Garcia`` is ``Ana Garcia``) and HTML character
    references (``Ana&nbsp;Garcia``, ``Ana&#32;Garcia``, ``Ana&#x20;Garcia``)."""
    s = url_decoded(text)
    if "_x" in s or "_X" in s:
        s = _OOXML_ESCAPE.sub(_ooxml_char, s)
    if "&" in s and ";" in s:
        s = _HTML_REFERENCE.sub(lambda m: html.unescape(m.group()), s)
    return s


def digits_as_separators(text: str) -> str:
    """``text`` with every digit run glued between two letter words read as a separator (K-10: ``Ana1Garcia`` is
    ``Ana Garcia``); ``text`` itself when it holds none."""
    return _GLUED_DIGITS.sub(" ", text) if any(ch.isdigit() for ch in text) else text


def _escaped_forms(text: str) -> list[str]:
    """The other forms a flagged name takes (K-10; :meth:`Flag.terms`, :func:`_redaction_forms`): decoded
    (:func:`decoded`) and, when that reads as a person's name, with glued digits read as separators
    (:func:`digits_as_separators`: ``Ana1Garcia`` is ``Ana Garcia``, but ``Student907550Goals`` is never ``Student
    Goals``); none when ``text`` holds no escape."""
    out: list[str] = []
    dec = decoded(text)
    if dec != text and dec.strip():
        out.append(dec)
    for v in (digits_as_separators(text), digits_as_separators(dec)):
        if v != text and v.strip() and v not in out and person_name_hit(v) is not None:
            out.append(v)
    return out


def name_kinds(text: str, ids: IdKeys | None = None) -> frozenset[str]:
    """Kinds of personal data in one header, sheet, file or folder name on its own (module docstring: checks
    1-3; a real ID only when the match alone is strong, :meth:`IdKeys.match`), as written and decoded (URL escapes,
    OOXML ``_xHHHH_`` escapes, HTML character references: :func:`decoded`); a person's name also with digits glued
    between two words read as a separator (``Ana1Garcia``, K-10)."""
    if not text or not text.strip():
        return frozenset()
    dec = decoded(text)
    kinds = _name_kinds(text, ids)
    if dec != text and dec.strip():
        kinds |= _name_kinds(dec, ids)
    if "name" not in kinds:
        for v in {digits_as_separators(text), digits_as_separators(dec)} - {text, dec}:
            if person_name_hit(v) is not None:
                kinds.add("name")
                break
    return frozenset(kinds)


def _name_kinds(text: str, ids: IdKeys | None) -> set[str]:
    s = clean_text(text)
    # R7-03: THE shape test of values (``detect.text.unit_shapes``: every personal-data shape, also a bare North
    # American number, R6-01; an underscore counts as a space), so a name and a value share one shape test. The D34
    # exceptions only ever excuse a bare 10-digit run (:func:`_phone_in_name`), never another shape
    kinds = set(unit_shapes(s)) & _SHAPE_KINDS
    if "phone" in kinds and not _phone_in_name(s):
        kinds.discard("phone")
    if "ssn" in kinds and not _ssn_in_name(s):
        kinds.discard("ssn")
    if _LABELED_SSN.search(s.replace("_", " ")):
        kinds.add("ssn")
    if _labeled_local_phone(s):
        kinds.add("phone")
    if person_name_hit(s) is not None:
        kinds.add("name")
    if ids is not None and ids.match(s):
        kinds.add("id")
    return kinds


#: Words right before 10 bare digits that make them a phone number whatever the digits (``Ph 5551234567``).
_PHONE_LABELS = frozenset({"ph", "phone", "tel", "telephone", "cell", "mobile", "call", "fax", "text", "contact",
                           "home", "work", "emergency", "parent", "guardian", "mom", "dad", "mother", "father",
                           "grandma", "grandpa", "grandmother", "grandfather", "aunt", "uncle", "sibling"})
#: A 9-digit SSN written without separators after its label (``SSN 123456789``, ``SSN_123456789``; K-12: ``SS#
#: 123456789``, ``SS No 123456789``, ``Soc Sec 123456789``, ``W2_123456789``, ``W-2 123456789``). The 3-2-4 layout
#: with any separators is an SSN whatever comes before it (``detect.text.EMB_SSN_RE``, the shape test of values).
_LABELED_SSN = re.compile(
    r"(?<![A-Za-z0-9])(?:ssn|ss(?=\s*#)|ss\s*(?:no|num|nbr)\b|soc(?:ial)?\s*sec(?:urity)?(?:\s*(?:number|num|no|nbr))?"
    r"|w-?2)\W*#?\s*\d{9}(?!\d)", re.I)
#: K-12: a local North American number (exchange 2-9, no ``N11`` service code; 3 + 4 digits joined by a space, dot
#: or dash): a phone number after a phone or family word (``Call 486-1234``, ``Ph 486 1234``, ``Cell_486_1234``).
_LOCAL_PHONE = re.compile(r"(?<![\w.\-\u2010-\u2013])([2-9](?!11)\d\d)[ .\-\u2010-\u2013]{1,3}\d{4}"
                          r"(?![\w\-\u2010-\u2013]|[.,]\d)")


#: K-12: the words that make a local number a phone number: phone-line and family words (``hints.PHONE_LINE_WORDS``,
#: ``Call``, ``Ph``, ``Cell``, ``Mom``), never a bare contact qualifier (``Home``, ``Work``, ``Emergency``: ``Work
#: 250-1000`` is a range) or ``Text`` (``Text Level 200-1000``).
_LOCAL_PHONE_LABELS = (_PHONE_LABELS - hints.CONTACT_QUALIFIERS - {"text"}) | hints.PHONE_LINE_WORDS | {"mob"}


def _labeled_local_phone(s: str) -> bool:
    """K-12: ``s`` holds a local number (:data:`_LOCAL_PHONE`) right after a phone or family word
    (:data:`_LOCAL_PHONE_LABELS`)."""
    if not any(ch.isdigit() for ch in s):
        return False
    for v in {s, s.replace("_", " ")}:
        for m in _LOCAL_PHONE.finditer(v):
            if _words_before(v, m.start()) & _LOCAL_PHONE_LABELS:
                return True
    return False


#: Ten digits that are a valid North American number (``detect.text.nanp_digits``, the one test of values and
#: names): area code and exchange start with 2-9 and are no ``N11`` service code, the area code no ``N9X`` code.
nanp_number = nanp_digits


def _phone_in_name(s: str) -> bool:
    """D34's narrowing of a phone shape (:func:`detect.text.unit_shapes`) found in a name: it stays a phone number
    unless EVERY phone shape of ``s`` is a bare 10-digit run (11 with a leading 1 or 0) that is excused. A separated
    (``312-555-0199``) or international layout is never excused, nor are bare digits after a phone or family word
    (``Ph 5551234567``, ``Parent_5125550199``). A bare run is excused when it is no valid North American number
    (:func:`nanp_number`), when it is a time stamp (``2023101512`` = YYYYMMDDHH; ``1696342522``, a Unix time, is no
    valid number) or when it follows a word of a numbered thing (:data:`_PHONE_NUMBERED_WORDS`: an export, job,
    record or survey number: ``Extract_5839201746``, ``Survey_Responses_...``, ``Qualtrics_SV_...``, ``PO
    4500012345``, ``Invoice 3105551234``). R8-01: a phone number in another national layout (``020 7946 0958``,
    ``Parent_07911_123456``: ``detect.text.phone_runs``) is excused only after a word of a numbered thing (``Invoice
    2023 0001 2345``), like a bare run, or when it reads as a list of numbers (:func:`_number_list`:
    ``Cut_200_250_300``, ``Pct_10_25_50_75_90``, ``05_10_15_20_25``), never after a phone or family word, with a
    ``+`` or with an area code in parentheses. K-19: three groups of 3, 3 and 4 digits joined only by spaces or
    underscores (``Lexile_800_900_1000``, ``Cut_100_200_3000``) are read like a bare run: excused when they are no
    valid North American number (a trunk ``0`` aside), follow a word of a numbered thing or read as a list of numbers;
    never after a phone or family word. Fail-safe (R7-03): a phone shape no excused run accounts for (a whole
    name ``3125550187``, ``3125550187.0``) stays a phone number."""
    excused = False
    for v in {s, s.replace("_", " ")}:
        for m in EMB_PHONE_RE.finditer(v):
            verdict = _spaced_run_excused(v, m.start(), m.group())
            if not verdict:  # a separated North American layout (None), or a spaced one that is a phone number
                return True
            excused = True
        if "+" in v and any(8 <= sum(ch.isdigit() for ch in m.group()) <= 15 for m in EMB_INTL_PHONE_RE.finditer(v)):
            return True
        for m in phone_runs(v):  # R8-01: a phone number in any national layout
            verdict = _spaced_run_excused(v, m.start(), m.group())
            if verdict is not None:
                if not verdict:
                    return True
                excused = True
                continue
            words = _words_before(v, m.start())
            if words & _PHONE_LABELS or m.group("plus") is not None or "(" in m.group():
                return True
            groups = re.findall(r"\d+", m.group())
            if words & _PHONE_NUMBERED_WORDS or _number_list(groups) or _year_and_number(groups):
                # an invoice, order or record number; a list of numbers (``Cut_200_250_300``); B8: a year or a school
                # year beside a number too short for a phone number (``Grades_2024-25_612345``, ``ELA_2025_26_101``)
                excused = True
                continue
            return True
        for m in EMB_BARE_PHONE_RE.finditer(v):
            d = "".join(ch for ch in m.group() if ch.isdigit())
            if len(d) == 11:
                d = d[1:]
            words = _words_before(v, m.start())
            if words & _PHONE_LABELS:
                return True
            if (not nanp_number(d) or (date_like(d[:8]) and int(d[8:]) <= 23)
                    or words & _PHONE_NUMBERED_WORDS):  # no number, a time stamp, an export/job/record number
                excused = True
                continue
            return True
    return not excused


#: B8: an SSN label right before a position (``SSN``, ``SS#``, ``SS No``, ``Soc Sec``, ``Social Security #``, ``W2``).
_SSN_LABEL_BEFORE = re.compile(
    r"(?<![A-Za-z0-9])(?:ssn|ss(?:\s*(?:no|num|nbr))?|soc(?:ial)?\s*sec(?:urity)?(?:\s*(?:number|num|no|nbr))?|w-?2)"
    r"[\s_.:#=\-]*$", re.I)


def _ssn_in_name(s: str) -> bool:
    """B8: D34's narrowing of the SSN shape (``detect.text.EMB_SSN_RE``, any separators) found in a name: it stays an
    SSN unless every such 3-2-4 run is joined by spaces or underscores only, its 2-digit group is a month and its
    4-digit group a year, and no SSN label stands before it (``Bus_Route_214_12_2024``, ``Room_305_09_2024_Attendance``,
    ``Section 101 01 2025 Roster``: a route, room, section or course number, a month and a year). A dashed, dotted or
    slashed layout (``214-12-2024``) and a labelled one (``SSN_214_12_2024``) stay SSNs, and so does any other 3-2-4
    run (fail-safe: a shape no excused run accounts for)."""
    found = False
    for v in {s, s.replace("_", " ")}:
        for m in EMB_SSN_RE.finditer(v):
            found = True
            groups = _DIGIT_RUN.findall(m.group())
            if (_DIGIT_RUN.sub("", m.group()).strip(" ") or not 1 <= int(groups[1]) <= 12
                    or not (len(groups[2]) == 4 and _year(groups[2])) or _SSN_LABEL_BEFORE.search(v[:m.start()])):
                return True
    return not found


def _year_and_number(groups: list[str]) -> bool:
    """B8: digit groups of a phone shape in a name that open or close with a year or a school year (``2024``, ``2024
    25``, ``2024 2025``) and hold fewer digits than a phone number besides (``Grades_2024-25_612345``,
    ``ELA_2025_26_101_Roster``, ``612345_2024-25``): a dated name, no phone number."""
    def year(g: str) -> bool:
        return len(g) == 4 and _year(g)

    def school(a: str, b: str) -> bool:
        return year(a) and ((len(b) == 2 and int(b) == (int(a) + 1) % 100) or (year(b) and int(b) == int(a) + 1))

    n = len(groups)
    lead = 2 if n >= 2 and school(groups[0], groups[1]) else 1 if n and year(groups[0]) else 0
    trail = 2 if n >= 2 and school(groups[-2], groups[-1]) else 1 if n and year(groups[-1]) else 0
    return any(k and rest and sum(map(len, rest)) < PHONE_RUN_MIN
               for k, rest in ((lead, groups[lead:]), (trail, groups[:n - trail])))


#: K-19: three digit groups of 3, 3 and 4 digits joined by spaces only (underscores read as spaces).
_SPACED_334 = re.compile(r"(\d{3}) +(\d{3}) +(\d{4})")


def _spaced_run_excused(v: str, start: int, run: str) -> bool | None:
    """K-19: None when ``run`` (a phone shape found at ``start`` of ``v``) is no 3-3-4 layout joined by spaces or
    underscores alone; otherwise whether it is excused like a bare run of ten digits (:func:`_phone_in_name`): never
    after a phone or family word; excused when it is no valid North American number (a trunk ``0`` keeps it a phone
    number, R9-02), after a word of a numbered thing or when it reads as a list of numbers (:func:`_number_list`)."""
    m = _SPACED_334.fullmatch(run)
    if m is None:
        return None
    words = _words_before(v, start)
    if words & _PHONE_LABELS:
        return False
    d = "".join(m.groups())
    return bool((d[0] != "0" and not nanp_number(d)) or words & _PHONE_NUMBERED_WORDS or _number_list(list(m.groups())))


def _number_list(groups: list[str]) -> bool:
    """R8-01 in a name: digit groups that read as a list of numbers rather than one phone number (a name's separators
    are underscores, so the value test's dash-based rules for ranges and ZIP+4 codes do not see them): groups of at
    most three digits without a trunk ``0`` (``Cut_200_250_300``, ``Pct_10_25_50_75_90``), a range of two numbers of
    one length (``10000_20000``), a ZIP+4 code (``78701_1234``) or three or more groups counting up in one round step
    (at most 10, or a multiple of 5: ``05_10_15_20_25``, ``06_12_18_24_30``)."""
    if not trunk_zero(groups) and max(map(len, groups)) <= 3:
        return True
    nums = [int(g) for g in groups]
    if len(groups) == 2 and ((len(groups[0]), len(groups[1])) == (5, 4)
                             or (len(groups[0]) == len(groups[1]) and nums[0] < nums[1])):
        return True
    steps = {b - a for a, b in zip(nums, nums[1:])}
    step = steps.pop() if len(steps) == 1 else 0
    return len(nums) >= 3 and step > 0 and (step <= 10 or step % 5 == 0)


def _words_before(v: str, start: int) -> set[str]:
    """The word right before position ``start`` of ``v`` and its last camelCase piece, case-folded (``PO``, ``po``;
    ``ParentPhone`` -> ``parentphone``, ``phone``)."""
    label = _LABEL_BEFORE.search(v[:start])
    if label is None:
        return set()
    pieces = _CAMEL_RE.split(label.group(1))
    return {label.group(1).casefold(), pieces[-1].casefold()} if pieces else set()


def kinds_text(kinds: Iterable[str]) -> str:
    ks = set(kinds)
    return ", ".join(KIND_TEXT[k] for k in _KIND_ORDER if k in ks)


def split_name(name: str) -> tuple[str, str]:
    """(stem, extension(s)) of a file name: ``Smith_John_IEP.csv.gz`` -> (``Smith_John_IEP``, ``.csv.gz``),
    ``Garcia_Ana.parquet`` -> (``Garcia_Ana``, ``.parquet``). Only a known extension
    (``io.registry.known_extension``) splits: ``Ana.Garcia`` is one name (``('Ana.Garcia', '')``)."""
    from ..io import registry

    suffixes = PurePosixPath(name).suffixes
    if not suffixes or not registry.known_extension(suffixes[-1]):
        return name, ""
    n = 1
    if len(suffixes) >= 2 and suffixes[-1].lower() in registry.COMPRESSION_SUFFIXES \
            and suffixes[-2].lower() in registry.EXTENSION_FORMATS:
        n = 2
    ext_len = sum(len(x) for x in suffixes[-n:])
    stem, ext = name[:-ext_len], name[-ext_len:]
    if not stem or any(ch.isspace() for ch in ext):
        return name, ""
    return stem, ext


def _with_given_name(text: str, words: tuple[str, ...]) -> tuple[str, ...]:
    """B3: the name words ``words`` of a flagged person's name (:class:`NameHit`) with the given name before its middle
    initials that the layout check reads past (``Mark A. Patel``, ``Summer J. Davis``: an everyday word is no evidence
    of a name on its own, but before the initials of a flagged name it is that person's given name): the capitalized
    word of two letters or more right before the initials that precede the first name word, spaces or underscores only
    between it and them; never a title, a place or an institution word (``Mr. A. Patel``, ``Room B. Patel``)."""
    if not words:
        return words
    s = clean_text(text)
    toks = _tokens(s)
    i = next((k for k, t in enumerate(toks) if t.key == words[0]), None)
    if i is None:
        return words
    j = i - 1
    while j >= 0 and len(toks[j].key) == 1 and toks[j].raw.isalpha() and toks[j].raw.isupper():
        j -= 1
    if j < 0 or j == i - 1:
        return words
    g = toks[j]
    if (g.raw.isalpha() and len(g.key) >= 2 and g.raw[:1].isupper() and g.key not in words
            and g.key not in _TITLES | _ROLE_TITLES | INSTITUTION_WORDS | _PLACE_BEFORE
            and s[g.end:toks[j + 1].start].strip(" _") == ""):
        return (g.key, *words)
    return words


def _given_surname_pairs(text: str, words: tuple[str, ...]) -> list[tuple[str, str]]:
    """B3: a flagged person's name of three words or more (``words``, in written order) as its first given name beside
    each surname word, in either order, middle names left out (``Davis, Summer Joy``: ``summer davis``, ``davis
    summer``; ``Ana Rose Garcia``: ``ana garcia``; ``Garcia Lopez, Maria``: ``maria garcia``, ``maria lopez``): the
    surnames are the name words before a comma or semicolon (``Surname, Given Middle``), else the last name word."""
    if len(set(words)) < 3:
        return []
    s = clean_text(text)
    cut = min((p for p in (s.find(","), s.find(";")) if p >= 0), default=-1)
    k = 0
    if cut >= 0:
        names = set(words)
        k = sum(1 for t in _tokens(s[:cut]) if t.key in names)
    surnames, given = (words[:k], words[k:]) if 0 < k < len(words) else (words[-1:], words[:-1])
    out: list[tuple[str, str]] = []
    for sur in surnames:
        for pair in ((given[0], sur), (sur, given[0])):
            if pair[0] != pair[1] and pair not in out:
                out.append(pair)
    return out


@dataclass
class Flag:
    """One flagged name. ``text`` is real data: PRIVATE, never serialised or printed."""

    text: str
    what: str  # 'folder' | 'file' | 'sheet' | 'column' | 'title' | 'path' | 'setting' | 'message'
    kinds: frozenset[str]
    shown: str  # the placeholder printed instead of the text
    where: str  # report-safe location
    written: bool  # the output would hold it (FAIL) or it is only printed (redacted)
    words: tuple[str, ...] = ()  # PRIVATE: name words of a person name (run-wide real-name index)
    number: int | None = None  # column number (1-based) of a header
    file: str | None = None  # shown (redacted) relative path of the (first) file it belongs to
    id_tokens: tuple[str, ...] = ()  # PRIVATE: the real IDs it holds (as written)
    #: flagged only beside another flagged name (a typed glob naming one word of a flagged person's name,
    #: ``*Okafor*``): one everyday-looking word, so the leak scanner indexes it only within its whole name
    group_only: bool = False

    def terms(self) -> tuple[list[str], list[tuple[str, ...]], list[str]]:
        """What the leak scanner indexes for this name (``privacy.leaks``; PRIVATE): (single folded words, token
        sequences, exact strings). A name of 2-6 word tokens is one sequence, a person's name words another;
        a one-token name is a word; the whole name (>= 4 characters) and, for a real ID, the letter-bearing real IDs
        it holds are exact strings. Never its other parts (``Reading`` of ``S482901_Reading_Score``): a clean name that
        shares one is no leak. A name flagged only with its siblings gives no single word. P-10: a real ID that is a
        plain number (``1375`` of a pivot, ``612587`` of ``Student 612587``), and a name that is such a number alone and
        holds nothing else, are no never-excused text: they are real IDs, which the scanner already indexes with the
        run's ID columns and judges by its ID rules (an exact maximum or a row count that equals a dense 4-digit ID is
        a statistic, no leak)."""
        from .leaks import NGRAM, WORD_RE, fold  # lazy: leaks imports this module

        toks = WORD_RE.findall(fold(self.text))
        bare_id = self.kinds == frozenset({"id"}) and _bare_number(self.text) is not None
        words: list[str] = []
        seqs: list[tuple[str, ...]] = []
        if 2 <= len(toks) <= NGRAM:
            seqs.append(tuple(toks))
        elif len(toks) == 1 and len(toks[0]) >= 2 and not self.group_only and not bare_id:
            words.append(toks[0])
        if len(set(self.words)) >= 2:
            # A4: a person's name words in written order and in each other order that keeps them together (``Patel,
            # Mark`` is also ``Mark Patel``; ``Garcia Lopez, Maria`` also ``Maria Garcia Lopez``): one person's name;
            # B3: also the first given name beside each surname, middle names left out (``Davis, Summer Joy`` is also
            # ``Summer Davis``)
            rots = [tuple(self.words[k:]) + tuple(self.words[:k]) for k in range(len(self.words))]
            for seq in [*rots, *_given_surname_pairs(self.text, self.words)]:
                if seq not in seqs:
                    seqs.append(seq)
        exact: list[str] = []
        if len(self.text.strip()) >= MIN_ID_LENGTH and not (self.group_only and len(toks) == 1) and not bare_id:
            exact.append(self.text.strip())
        # 'Kaden%20Yazzie%20IEP' is also indexed as 'Kaden Yazzie IEP'; K-10: 'Ana_x0020_Garcia', 'Ana&nbsp;Garcia' and
        # 'Ana1Garcia' as 'Ana Garcia'
        for form in _escaped_forms(self.text):
            dtoks = WORD_RE.findall(fold(form))
            if 2 <= len(dtoks) <= NGRAM and tuple(dtoks) not in seqs:
                seqs.append(tuple(dtoks))
            if len(form.strip()) >= MIN_ID_LENGTH:
                exact.append(" ".join(form.split()))
        exact += [t for t in self.id_tokens if len(t) >= MIN_ID_LENGTH and not t.isdigit()]
        return words, seqs, sorted(set(exact))


#: K-16: the joiners inside a leak-scanner word token (``Smith-Jones``, ``O'Brien``).
_TOKEN_JOINERS = re.compile(r"['\u2019\-\u2010\u2011]")


def _scan_sequences(text: str) -> list[tuple[str, ...]]:
    """K-16: the token sequences of ``text`` as the leak scanner reads it (``privacy.leaks.WORD_RE`` over
    ``leaks.fold``: split at underscores and every other separator, joined at hyphens and apostrophes) and split finer
    (each token also at its hyphens and apostrophes and at camelCase: ``Jack-Yates-HS`` and ``JackYatesHS`` are ``jack
    yates hs``; never between letters and digits, so ``Q1014`` holds no ``1014``)."""
    from .leaks import WORD_RE, fold  # lazy: leaks imports this module

    coarse = tuple(WORD_RE.findall(fold(text)))
    fine: list[str] = []
    for tok in WORD_RE.findall(text):
        for part in _CAMEL_RE.split(tok):
            fine += [p.casefold() for p in _TOKEN_JOINERS.split(part) if p]
    return [coarse] if tuple(fine) == coarse else [coarse, tuple(fine)]


@lru_cache(maxsize=65536)
def _redaction_forms(text: str) -> frozenset[str]:
    """Forms a name takes in printed text: as is, whitespace collapsed, Markdown-table escaped, inside repr()
    and JSON strings, URL-encoded (report links) and console-escaped (ASCII backslash escapes), and without
    its invisible format characters; an escaped name (``Kaden%20Yazzie``; K-10: ``Ana_x0020_Garcia``,
    ``Ana&nbsp;Garcia``, ``Ana1Garcia``: :func:`_escaped_forms`) also in its decoded forms."""
    forms = {text, text.strip(), " ".join(text.split())}
    plain = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    forms |= {plain, " ".join(plain.split())}
    variants = [text, unicodedata.normalize("NFC", text), unicodedata.normalize("NFD", text)]
    for form in _escaped_forms(text):
        forms |= {form.strip(), " ".join(form.split())}
        variants.append(form)
    for t in variants:
        forms.add(t)
        forms.add(repr(t)[1:-1])
        forms.add(json.dumps(t, ensure_ascii=False)[1:-1])
        forms.add(json.dumps(t)[1:-1])
        forms.add(t.replace("|", "\\|").replace("\n", " ").replace("\r", " "))
        forms.add(urllib.parse.quote(t, safe="/"))
        forms.add(urllib.parse.quote(t, safe=""))
        forms.add(t.encode("ascii", "backslashreplace").decode("ascii"))
    return frozenset(f for f in forms if f.strip())


class NameScreen:
    """The D34 result of a run (PRIVATE, in memory only): the flags, the redactor and the path screen."""

    def __init__(self, ids: IdKeys | None = None, keep: Iterable[str] = ()) -> None:
        self.ids = ids
        self.keep = frozenset(keep)
        self.flags: list[Flag] = []
        self._shown: dict[str, str] = {}  # flagged text -> placeholder
        self._pattern: re.Pattern[str] | None = None
        self._replace: dict[str, str] = {}
        # R7-07: derived indexes, rebuilt only when a flag or a placeholder is added (they were rebuilt for every
        # header :meth:`texts_in` looked at: quadratic in a run with thousands of flagged headers)
        self._by_shown: dict[str, str] = {}  # placeholder -> flagged text (with the pattern)
        self._standalone: frozenset[str] | None = None
        self._kinds_of: dict[str, frozenset[str]] | None = None
        #: K-16: first token -> (token sequence, flagged text) of the standalone flags' leak-scanner terms
        self._seq_index: dict[str, list[tuple[tuple[str, ...], str]]] | None = None
        self._seq_seen: set[tuple[tuple[str, ...], str]] = set()
        #: P-10: the real IDs of the flags that hold nothing else (one namespace) and the flagged texts of each D8 key
        self._id_flags: tuple[IdKeys, dict[str, set[str]]] | None = None
        self._public_numbers: frozenset[str] = frozenset()

    # ---------------------------------------------------------------- flags
    def kept(self, text: str, ext: str = "") -> bool:
        return text in self.keep or (bool(ext) and text + ext in self.keep)

    def kinds(self, text: str, ext: str = "") -> frozenset[str]:
        if self.kept(text, ext):
            return frozenset()
        return name_kinds(text, self.ids)

    def group_kinds(self, items: list[tuple[str, str]], local: list[frozenset[str] | None] | None = None,
                    parent: str | None = None, headers: bool = False
                    ) -> list[tuple[frozenset[str], tuple[str, ...], bool]]:
        """(kinds, real ID tokens, flagged only as a sibling) of sibling names ``(text, extension)`` screened
        together (module docstring: repeated real IDs); ``local``: the IDs each named file,
        folder or sheet holds (:meth:`IdKeys.matches`); ``parent``: the name of the folder, workbook or table they
        belong to (:func:`numbered_context`; for files, folders and sheets, ``headers`` False, also
        :func:`person_context`)."""
        texts = [t for t, _ in items]
        kept = [self.kept(t, e) for t, e in items]
        out: list[tuple[set[str], set[str], bool]] = []
        context = numbered_context(parent)
        for (t, e), k in zip(items, kept):
            kinds = set() if k else set(self.kinds(t, e))
            if decoded(t) == t:
                # a real ID is the sibling check's (id_group_flags flags a name that holds one alone too), which weighs
                # the siblings together (B1: a labelled date among dated siblings is a date)
                kinds.discard("id")
            if context and kinds == {"id"} and t.strip().isdigit():  # the parent says what the number is
                kinds = set()
            out.append((kinds, set(), False))
        for i, toks in id_group_flags(self.ids, [decoded(t) for t in texts], local, context,
                                      not headers and person_context(parent)).items():
            if not kept[i]:
                out[i][0].add("id")
                out[i][1].update(toks)
        return [(frozenset(k), tuple(sorted(t)), g) for k, t, g in out]

    def placeholder(self, text: str) -> str:
        shown = self._shown.get(text)
        if shown is None:
            shown = self._shown[text] = PLACEHOLDER.format(len(self._shown) + 1)
            self._pattern = None
            self._standalone = None
        return shown

    def add(self, text: str, what: str, kinds: frozenset[str], where: str, written: bool,
            number: int | None = None, id_tokens: tuple[str, ...] = (), group_only: bool = False,
            words: tuple[str, ...] = ()) -> Flag:
        hit = person_name_hit(url_decoded(text)) if "name" in kinds and not words else None
        if hit is None and "name" in kinds and not words:
            hit = person_name_hit(text)
        for form in _escaped_forms(text) if hit is None and "name" in kinds and not words else ():
            hit = person_name_hit(form)  # K-10: 'Ana_x0020_Garcia', 'Ana&nbsp;Garcia', 'Ana1Garcia'
            if hit is not None:
                break
        if text in self._shown and _bare_number(text) is not None and kinds != self.kinds_of(text):
            self._pattern = None  # P-10: how a number alone is redacted depends on its kinds
        if hit is not None and not words:  # B3: 'Mark A. Patel' is Mark Patel's name
            words = next((w for form in (url_decoded(text), text, *_escaped_forms(text))
                          if (w := _with_given_name(form, hit.words)) != hit.words), hit.words)
        flag = Flag(text, what, kinds, self.placeholder(text), where, written, words, number, id_tokens=id_tokens,
                    group_only=group_only)
        self.flags.append(flag)
        self._standalone = None
        self._kinds_of = None
        self._id_flags = None
        self._index_sequences(flag)
        return flag

    def kinds_of(self, text: str) -> frozenset[str]:
        """The kinds every flag of ``text`` holds (empty when ``text`` is not flagged)."""
        if self._kinds_of is None:
            index: dict[str, set[str]] = {}
            for f in self.flags:
                index.setdefault(f.text, set()).update(f.kinds)
            self._kinds_of = {t: frozenset(k) for t, k in index.items()}
        return self._kinds_of.get(text, frozenset())

    @property
    def texts(self) -> frozenset[str]:
        return frozenset(self._shown)

    def __bool__(self) -> bool:
        return bool(self.flags)

    def name_word_lists(self) -> list[tuple[str, ...]]:
        """Name words of every flagged person name (``generate.pii.plan_names``)."""
        return [f.words for f in self.flags if len(set(f.words)) >= 2]

    def standalone_texts(self) -> frozenset[str]:
        """Flagged texts that hold personal data on their own (not only as one of several siblings)."""
        if self._standalone is None:
            group = {f.text for f in self.flags if f.group_only}
            alone = {f.text for f in self.flags if not f.group_only}
            self._standalone = frozenset(alone | (set(self._shown) - group))
        return self._standalone

    # ---------------------------------------------------------------- paths
    def display_path(self, rel: str) -> str:
        """``rel`` (posix, relative) with every flagged folder or file name shown as its placeholder; a part not
        screened yet (an out-dir path from an earlier run) is screened now (nothing is recorded)."""
        parts = rel.split("/")
        out = []
        for k, part in enumerate(parts):
            if not part:
                out.append(part)
                continue
            is_file = k == len(parts) - 1
            stem, ext = split_name(part) if is_file else (part, "")
            if stem in self._shown:
                out.append(self._shown[stem] + ext)
            elif part in self._shown:
                out.append(self._shown[part])
            elif self.kinds(stem, ext):
                out.append("[redacted]" + ext)
            else:
                out.append(part)
        return "/".join(out)

    # ---------------------------------------------------------------- redaction
    @property
    def public_numbers(self) -> frozenset[str]:
        """P-10: numbers alone that the run shows as unflagged names (:func:`_public_numbers`)."""
        return self._public_numbers

    @public_numbers.setter
    def public_numbers(self, value: Iterable[str]) -> None:
        self._public_numbers = frozenset(value)
        self._pattern = None

    def _bare_id(self, text: str) -> str | None:
        """P-10: the digits of a flagged name that is a number alone and holds a real ID only (a pivot's ``1375``, a
        folder's ``iep/1053.csv``), else None."""
        d = _bare_number(text)
        return d if d is not None and self.kinds_of(text) == frozenset({"id"}) else None

    def _compiled(self) -> re.Pattern[str] | None:
        if self._pattern is None and self._shown:
            repl: dict[str, str] = {}
            numbers: set[str] = set()
            for text, shown in self._shown.items():
                d = self._bare_id(text)
                if d is not None:  # P-10: redacted where it is a name, never as a statistic (below)
                    if d not in self._public_numbers:  # nor where the run shows it as another, unflagged name
                        numbers.add(d)
                        repl.setdefault(d, shown)
                    continue
                for form in _redaction_forms(text):
                    repl.setdefault(form.casefold(), shown)
            self._replace = repl
            self._by_shown = {v: k for k, v in self._shown.items()}
            alts = [re.escape(a) for a in sorted(set(repl) - numbers, key=len, reverse=True)]
            if numbers:
                # P-10: a real ID that is a flagged name alone is redacted where it is a name: in a path (``iep/1053``,
                # ``1053/``, ``1053.csv``) or quoted (``'1053'``, ``"1053"``); an exact maximum, a row count or a
                # rank that equals it is a statistic of another file (the leak scanner judges it by its ID rules)
                nums = "|".join(re.escape(d) for d in sorted(numbers, key=len, reverse=True))
                alts.append(rf"(?<=[/\\'\"`])(?:{nums})|(?:{nums})(?=[/\\'\"`]|\.[A-Za-z])")
            self._pattern = re.compile(r"(?<![0-9A-Za-z])(?:" + ("|".join(alts) or "(?!)") + r")(?![0-9A-Za-z])",
                                       re.I)
        return self._pattern

    def redact_text(self, text: str) -> str:
        pat = self._compiled()
        if pat is None or not text:
            return text
        return pat.sub(lambda m: self._replace.get(m.group().casefold(), PLACEHOLDER.format("")), text)

    def holds(self, text: str) -> bool:
        """``text`` holds a flagged name (any of its printed forms)."""
        pat = self._compiled()
        return bool(pat is not None and text and pat.search(text))

    def texts_in(self, text: str, standalone: bool = False) -> set[str]:
        """The flagged names ``text`` holds (any printed form; PRIVATE); ``standalone``: only names that hold
        personal data on their own (:meth:`standalone_texts`)."""
        pat = self._compiled()
        if pat is None or not text:
            return set()
        by_shown = self._by_shown
        found = {by_shown[self._replace[m.group().casefold()]] for m in pat.finditer(text)
                 if m.group().casefold() in self._replace}
        return found & self.standalone_texts() if standalone else found

    def _sequences(self) -> dict[str, list[tuple[tuple[str, ...], str]]]:
        """K-16: the token sequences the leak scanner indexes for each standalone flag (:meth:`Flag.terms`: its single
        word, its token sequences, its exact strings and real IDs), each as the scanner reads it and split finer
        (:func:`_scan_sequences`), by first token."""
        if self._seq_index is None:
            self._seq_index = {}
            self._seq_seen = set()
            for f in self.flags:
                self._index_sequences(f)
        return self._seq_index

    def _index_sequences(self, f: Flag) -> None:
        """Add the leak-scanner sequences of flag ``f`` to :meth:`_sequences` (a flag of its own, never one flagged
        only beside another: its text is then a standalone text)."""
        if f.group_only or self._seq_index is None or f.kinds == frozenset({"id"}):
            return  # a flag that holds a real ID only is matched by that ID (:meth:`_id_repeats`)
        words, seqs, exact = f.terms()
        for src in [*words, *(" ".join(q) for q in seqs), *exact]:
            for seq in _scan_sequences(src):
                if (len(seq) > 1 or (seq and len(seq[0]) >= 2)) and (seq, f.text) not in self._seq_seen:
                    self._seq_seen.add((seq, f.text))
                    self._seq_index.setdefault(seq[0], []).append((seq, f.text))

    def repeats(self, text: str, context: bool = False) -> set[str]:
        """The flagged names (standalone: :meth:`standalone_texts`) a written header, sheet, file or folder name not
        flagged on its own holds (step 3 of :func:`screen_run`, and the leak scanner's backstop on re-read output
        names, ``privacy.leaks``; one matcher for both): in any printed form (:meth:`texts_in`) or, K-16, by the token
        sequences the leak scanner indexes for them, as it reads them (split at underscores and other separators) and
        split finer (at hyphens, apostrophes and camelCase: :func:`_scan_sequences`): ``Jack_Yates_HS_Enrollment``,
        ``Jack-Yates-HS``, ``JackYatesHS`` hold ``Jack Yates``. ``context`` (K-18): the name's own parent says what
        its numbers are (:func:`numbered_context`); a name that is a number alone then takes no flag that holds only
        a real ID (``enrollment_by_zip.xlsx`` sheet ``30303`` beside another workbook's flagged sheet ``30303``).
        K-18, P-10: a flag that holds a real ID only is matched by that ID, weighed like any real ID in a name
        (:meth:`_id_repeats`), never by its text."""
        found = self.texts_in(text, standalone=True)
        index = self._sequences()
        if index:
            for toks in _scan_sequences(text):
                for k, tok in enumerate(toks):
                    for seq, flagged in index.get(tok, ()):
                        if flagged not in found and toks[k:k + len(seq)] == seq:
                            found.add(flagged)
        found = {t for t in found if self.kinds_of(t) != frozenset({"id"})} | self._id_repeats(text, context)
        if found and context and clean_text(text).strip().isdigit():
            found = {t for t in found if self.kinds_of(t) - {"id"}}
        return found

    def _flagged_ids(self) -> tuple[IdKeys, dict[str, set[str]]]:
        """P-10: the real IDs that the standalone flags holding a real ID only hold, as one namespace, and the flagged
        texts of each D8 key."""
        if self._id_flags is None:
            from types import SimpleNamespace

            raw: set[str] = set()
            owners: dict[str, set[str]] = {}
            standalone = self.standalone_texts()
            for f in self.flags:
                if f.text not in standalone or self.kinds_of(f.text) != frozenset({"id"}):
                    continue
                for tok in f.id_tokens:
                    raw.add(tok)
                    key = (tok.lstrip("0") or "0") if tok.isdigit() else normalize_id(tok)
                    owners.setdefault(key, set()).add(f.text)
            self._id_flags = (IdKeys([SimpleNamespace(raw_values=raw)] if raw else []), owners)
        return self._id_flags

    def _id_repeats(self, text: str, context: bool = False) -> set[str]:
        """K-18, P-10: the flags holding a real ID only whose ID ``text`` holds as strong evidence, weighed against
        the flagged IDs alone (:meth:`IdKeys.matches` over :meth:`_flagged_ids`): a letter-prefixed or labelled ID,
        or a plain number of which few are flagged (a labelled 6-digit ``Student 612587`` repeated in
        ``612587_Notes``); never a number of a dense pivot (25 of 9,000 four-digit numbers are flagged: a sheet
        ``1375`` of another workbook is chance), a date or time (an attendance sheet ``1007``), a numbered thing or a
        number alone under a numbered parent (``context``)."""
        keys, owners = self._flagged_ids()
        if not owners:
            return set()
        out: set[str] = set()
        for m in keys.matches(decoded(text), context=context):
            if m.strong:
                out |= owners.get(m.key, set())
        return out

    def redact_obj(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {self.redact_text(k) if isinstance(k, str) else k: self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_obj(v) for v in obj)
        return obj

    def redact_bytes(self, name: str, data: bytes) -> bytes:
        if self._compiled() is None:
            return data
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            return data
        if not self.holds(text):
            return data
        if name.lower().endswith(".json"):
            try:
                obj = json.loads(text)
            except ValueError:
                obj = None
            if obj is not None:
                return (json.dumps(self.redact_obj(obj), indent=2, ensure_ascii=False, default=str)
                        + "\n").encode("utf-8")
        return self.redact_text(text).encode("utf-8")

    def redact_payload(self, payload: dict[str, bytes]) -> dict[str, bytes]:
        return {self.redact_text(k): self.redact_bytes(k, v) for k, v in payload.items()}


@dataclass
class NameScreenMessage(Message):
    """The D34 run-level note. Carries the PRIVATE :class:`NameScreen` (never serialised: :meth:`to_public` is
    inherited), whose redactor the report builder, the console and the out-dir docs apply."""

    screen: Any = field(default=None, repr=False, compare=False)

    def redact_payload(self, payload: dict[str, bytes]) -> dict[str, bytes]:
        return self.screen.redact_payload(payload) if self.screen is not None else payload

    def redact_text(self, text: str) -> str:
        return self.screen.redact_text(text) if self.screen is not None else text

    def redact_obj(self, obj: Any) -> Any:
        return self.screen.redact_obj(obj) if self.screen is not None else obj


def screen_of(messages: Iterable[Message]) -> NameScreen | None:
    """The run's :class:`NameScreen` from its messages (None when nothing was flagged)."""
    return next((m.screen for m in messages if isinstance(m, NameScreenMessage) and m.screen is not None), None)


# --------------------------------------------------------------------------- names in printed text
#: A Python-repr quoted string in a message (``'Okonkwo, Chinedu'``, ``"O'Brien, Pat"``): read and usage
#: messages quote sheet, column and range names this way.
_QUOTED = re.compile(r"'(?:[^'\\\n]|\\.){1,200}'|\"(?:[^\"\\\n]|\\.){1,200}\"")


def quoted_names(text: str) -> list[str]:
    """The names a message quotes (repr-quoted strings, unescaped)."""
    out: list[str] = []
    for m in _QUOTED.finditer(text or ""):
        q = m.group()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # an invalid escape in a quoted Windows path
                v = ast.literal_eval(q)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            v = q[1:-1]
        if isinstance(v, str) and v.strip():
            out.append(v)
    return out


def redact_message(text: str, keep: Iterable[str] = (), typed: Iterable[str] = ()) -> str:
    """``text`` (a usage or read error printed before any run exists) with every quoted name and every part of a
    quoted reference (``book.xlsx::Okonkwo, Chinedu``, ``Okonkwo!A5:H200``) that holds personal data on its own
    replaced by ``[redacted]``; ``typed``: the command-line arguments, whose values and their parts (``COL`` of
    ``COL=TYPE``, the folders of a path) are redacted the same way wherever the message repeats them unquoted
    (``--force-type Garcia_Ana=bogus: unknown type``). The folders and file of a typed path are screened on their
    own, so the message may print the path in another form (another separator, ``./``, a trailing separator, the
    resolved absolute path), and so are the folders of every path the message prints (a name between two path
    separators)."""
    if not text:
        return text
    screen = NameScreen(keep=keep)
    hits: set[str] = set()
    cands = quoted_names(text) + _PRINTED_FOLDER.findall(text)
    for t in typed:
        if not isinstance(t, str) or not t.strip():
            continue
        if t in text:
            cands.append(t)
        for part in {x for ref in _ref_parts(t) for x in _path_parts(ref)}:
            if any(form in text for form in _redaction_forms(part)):
                cands.append(part)
    for q in cands:
        parts = _ref_parts(q)
        found = {p for p in parts[1:] + _path_parts(q) if len(p) >= 2 and screen.kinds(p)}
        if not found and parts and screen.kinds(parts[0]):
            found.add(parts[0])
        hits |= found
    for h in sorted(hits, key=len, reverse=True):
        for form in _redaction_forms(h):
            text = text.replace(form, "[redacted]")
    return text


#: A folder name of a path printed in a message: the text between two path separators.
_PRINTED_FOLDER = re.compile(r"(?<=[\\/])[^\\/\n'\"]{2,200}(?=[\\/])")


def _path_parts(text: str) -> list[str]:
    """The folder and file names of a path typed as an argument (none for text without a path separator)."""
    if "/" not in text and "\\" not in text:
        return []
    out = []
    for part in re.split(r"[\\/]+", text):
        part = part.strip()
        if part and part not in (".", "..") and not part.endswith(":"):
            out.append(part)
            stem = split_name(part)[0]
            if stem != part:
                out.append(stem)
    return out


_REF_SPLIT = re.compile(r"::|!|=")


def _ref_parts(text: str) -> list[str]:
    """A typed reference and its parts: ``file::sheet::column``, ``Sheet!A5:H200``, ``COL=TYPE``."""
    parts = [text] + [p for p in _REF_SPLIT.split(text) if p.strip() and p != text]
    return [p.strip().strip("'\"") for p in parts if p.strip().strip("'\"")]


def _folded_tokens(text: str) -> tuple[str, ...]:
    return tuple(t.casefold() for t in _WORD_RE.findall(clean_text(text)))


def _holds_tokens(toks: tuple[str, ...], seq: tuple[str, ...]) -> bool:
    n = len(seq)
    return n > 0 and any(toks[i:i + n] == seq for i in range(len(toks) - n + 1))


# --------------------------------------------------------------------------- locations
def _sorted_names(names: Iterable[str]) -> list[str]:
    return sorted(names, key=lambda x: (x.casefold(), x))


def _listdir(folder: Path) -> tuple[list[str], list[str]]:
    """(sub-folder names, file names) of a folder on disk (empty when it cannot be listed)."""
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return [], []
    dirs, files = [], []
    for e in entries:
        try:
            (dirs if e.is_dir() else files).append(e.name)
        except OSError:
            continue
    return dirs, files


def _in_folder(parent_shown: str) -> str:
    return f"folder {parent_shown}/" if parent_shown else "the input folder"


def _ranges(nums: list[int], limit: int = 12) -> str:
    """``1-3, 7, 9-12`` (1-based numbers, sorted); long lists end with ``...``."""
    nums = sorted(set(nums))
    parts: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ", ".join(parts[:limit]) + (", ..." if len(parts) > limit else "")


def _kind_counts(flags: list[Flag]) -> str:
    c: Counter = Counter()
    for f in flags:
        for k in f.kinds:
            c[k] += 1
    return ", ".join(f"{KIND_TEXT[k]}: {c[k]}" for k in _KIND_ORDER if c[k])


def _single_word(text: str) -> str | None:
    """The folded word when ``text`` is one letter word (a folder ``Garcia``), else None."""
    toks = _tokens(clean_text(text).strip())
    return toks[0].key if len(toks) == 1 and toks[0].raw.isalpha() and len(toks[0].key) >= 2 else None


def _nested_names(p: Any) -> list[str]:
    """Names of the child fields of an Arrow struct/union/map column type, at any depth (PRIVATE): a column
    kept with ``--null`` keeps its type, so the output holds them."""
    native = getattr(p, "native", None)
    t = getattr(native.get("field"), "type", None) if isinstance(native, dict) else None
    out: list[str] = []

    def walk(t: Any, depth: int) -> None:
        if t is None or depth > 20:
            return
        n = getattr(t, "num_fields", 0) or 0
        for i in range(n):
            try:
                f = t.field(i)
            except (AttributeError, IndexError, TypeError):
                break
            name = getattr(f, "name", None)
            if isinstance(name, str) and name not in ("item", "element", "key", "value", "entries"):
                out.append(name)
            walk(getattr(f, "type", None), depth + 1)
        for attr in ("value_type", "key_type", "item_type"):
            child = getattr(t, attr, None)
            if child is not None and child is not t and not n:
                walk(child, depth + 1)

    walk(t, 0)
    return out


def _table_id_values(tp: Any) -> set[Any]:
    """The distinct raw values of a table's ID columns and shadow IDs (R6-02, ``profile.entity.privacy_id_raw``:
    an excluded, nulled or flagged ID column holds real IDs too), PRIVATE."""
    from ..profile import entity as entity_mod

    out: set[Any] = set()
    for raw in entity_mod.privacy_id_raw(tp).values():
        if raw is None:
            continue
        try:
            out.update(v for v in set(raw.tolist() if hasattr(raw, "tolist") else raw) if v is not None)
        except TypeError:  # unhashable values: never an ID here
            continue
    return out


def _file_id_values(fr: FileResult) -> set[Any]:
    """The distinct raw values of every ID column of a file's tables (PRIVATE)."""
    out: set[Any] = set()
    for tp in fr.tables:
        out |= _table_id_values(tp)
    return out


#: An Office owner file (``~$Garcia_Ana_IEP.xlsx``, Word: ``~$rcia_Ana_IEP.docx``) or a LibreOffice lock file
#: (``.~lock.Garcia_Ana_IEP.xlsx#``) of an open document, or a macOS AppleDouble file (``._Garcia_Ana_IEP.csv``).
_LOCK_RE = re.compile(r"^(?:~\$(?P<o>.+)|\.~lock\.(?P<l>.+)#|\._(?P<a>.+))$")


def _holds_name_word(text: str) -> bool:
    """``text`` holds a distinctive given name or surname that is no everyday or header word (``Okafor`` of
    ``Okafor Summer``)."""
    toks = _tokens(clean_text(text))
    kinds = _token_kinds(toks)
    return any(kd == "name" and len(t.key) >= 3 and _distinctive(t.key) and t.key not in _EVERYDAY_SURNAMES
               for t, kd in zip(toks, kinds))


def lock_file_rest(name: str) -> str | None:
    """The document name (without its extension) an Office owner or lock file or a macOS AppleDouble file
    (``._Garcia_Ana.csv``) ``name`` holds, or None."""
    m = _LOCK_RE.match(name)
    if not m:
        return None
    return split_name(m.group("o") or m.group("l") or m.group("a"))[0]


def _verbatim_texts(fr: FileResult) -> list[tuple[str, str]]:
    """``--keep-titles``: (text, location) of every title, notes, footer and side-note text a workbook or an
    HTML-disguised .xls would write verbatim (PRIVATE; ``privacy.texts.layout_texts``, which the run's real-text
    registry reads too, R8-04)."""
    from .texts import layout_texts

    return layout_texts(fr)


_SEGMENT_SPLIT = re.compile(r"[:;()\[\]|\n\r\t]+|\s+-\s+|\s+(?:by|for|of|from|to|with)\s+", re.I)


def _verbatim_kinds(screen: NameScreen, text: str) -> frozenset[str]:
    """Personal data in a title/notes text written verbatim: a shape, or a person name anywhere in the line
    (``Gradebook for Okonkwo, Chinedu (teacher)``) or as one of its clauses (``Teacher: Ana Garcia``), each
    judged in its own context (``Jefferson Davis High School - Roster`` names a school)."""
    kinds = set(screen.kinds(text))
    if "name" not in kinds:
        for seg in _SEGMENT_SPLIT.split(clean_text(text)):
            seg = seg.strip(" .,")
            if seg and person_name_hit(seg) is not None:
                kinds.add("name")
                break
    return frozenset(kinds)


# --------------------------------------------------------------------------- the run step
def screen_run(run: Any, settings: Any, root: Path, registry: Any = None) -> list[Message]:
    """D34 for a run (pipeline, after the run-level planning of IDs; ``inspect`` too): screen every name the
    run would write, FAIL the files holding a flagged one (withholding their details), show redacted paths,
    screen the names it only prints, and return the run-level :class:`NameScreenMessage` (or [] when nothing was
    flagged).

    ``root`` is the input root folder (discovery), ``registry`` the run's ID registry (None in ``inspect``:
    a scratch registry of every file's ID columns is used).
    """
    screen = NameScreen(run_id_keys(run.files, registry), getattr(settings, "keep_name", ()) or ())
    entries: list[tuple[str, Any]] = [(fr.rel_path, fr) for fr in run.files]
    entries += [(s.rel_path, s) for s in run.skipped]
    by_path: dict[str, list[Flag]] = {}  # rel path -> flags of its folders and file name

    # 1. folders and file names, screened per parent folder (siblings together), numbered in sorted path order
    folders: dict[str, None] = {}
    files_in: dict[str, list[tuple[str, str]]] = {}  # parent -> (rel, name) of files
    for rel, _ in entries:
        parts = rel.rstrip("/").split("/")
        n_dirs = len(parts) if rel.endswith("/") else len(parts) - 1
        for k in range(1, n_dirs + 1):
            folders.setdefault("/".join(parts[:k]), None)
        if not rel.endswith("/"):
            parent, _, name = rel.rpartition("/")
            files_in.setdefault(parent, []).append((rel, name))
    children: dict[str, list[tuple[str, str, str]]] = {}  # parent -> (kind, key, name)
    for folder in folders:
        parent, _, name = folder.rpartition("/")
        children.setdefault(parent, []).append(("folder", folder, name))
    for parent, items in files_in.items():
        for rel, name in items:
            children.setdefault(parent, []).append(("file", rel, name))
    # the IDs each file and folder holds: a name equal to one of its own few IDs is a real ID by itself
    file_local: dict[str, frozenset[str]] = {}
    folder_local: dict[str, set[str]] = {}
    if screen.ids:
        for fr in run.files:
            keys = screen.ids.local_keys(_file_id_values(fr))
            if keys:
                file_local[fr.rel_path] = keys
                parts = fr.rel_path.split("/")
                for k in range(1, len(parts)):
                    folder_local.setdefault("/".join(parts[:k]), set()).update(keys)
        # A2, A8: run-wide evidence (zero-padded codes, real IDs the run's names hold too often for chance)
        screen.ids.note_run_names(_run_names(run, children, file_local, folder_local))
    path_kinds: dict[str, tuple[frozenset[str], tuple[str, ...], bool, tuple[str, ...]]] = {}
    for parent, items in children.items():
        texts = [(name, "") if kind == "folder" else split_name(name) for kind, _, name in items]
        local = [frozenset(folder_local.get(key, ())) if kind == "folder" else file_local.get(key)
                 for kind, key, _ in items]
        pname = parent.rpartition("/")[2] if parent else Path(str(getattr(run, "input", "") or "")).name
        got = screen.group_kinds(texts, local, pname)
        for (kind, key, _), (kinds, toks, group) in zip(items, got):
            if kinds:
                path_kinds[key] = (kinds, toks, group, ())
    # an Office owner or lock file of a flagged document holds (part of) its name ('~$rcia_Ana_IEP.docx' of
    # 'Garcia_Ana_IEP.docx'): flagged with it
    for parent, items in children.items():
        stems = {key: (name if kind == "folder" else split_name(name)[0]) for kind, key, name in items}
        for kind, key, name in items:
            rest = lock_file_rest(name) if kind == "file" else None
            if rest is None or key in path_kinds or len(rest) < 4:
                continue
            for other, stem in stems.items():
                if other != key and other in path_kinds and stem != name and (
                        stem == rest or (stem.endswith(rest) and len(stem) - len(rest) <= 2)):
                    path_kinds[key] = (path_kinds[other][0], (), False, ())
                    break
            else:  # never written, only listed as skipped: any distinctive name word is enough to redact it
                kinds = screen.kinds(rest) or (frozenset({"name"}) if _holds_name_word(rest) else frozenset())
                if kinds and not screen.kept(rest):
                    path_kinds[key] = (kinds, (), False, ())
    folder_flags: dict[str, Flag] = {}
    file_flags: dict[str, Flag] = {}
    entry_of = dict(entries)
    single = Path(run.input).is_file()
    # R7-07: each folder is listed and sorted once, and the files under each folder counted once (they were
    # listed, sorted and counted again for every flagged name: quadratic in a folder of per-student files)
    files_under: Counter = Counter()
    for rel, _ in entries:
        if not rel.endswith("/"):
            parts = rel.split("/")
            for k in range(1, len(parts)):
                files_under["/".join(parts[:k])] += 1
    orders: dict[tuple[str, int], tuple[dict[str, int], list[tuple[str, str]]]] = {}

    def ordinal(parent: str, which: int, name: str) -> tuple[int, int]:
        """(place, count) of ``name`` among the sub-folders (``which`` 0) or files (1) of ``parent`` on disk,
        itself included, sorted by name ignoring case (:func:`_sorted_names`)."""
        got = orders.get((parent, which))
        if got is None:
            names = (_listdir(root / Path(*parent.split("/"))) if parent else _listdir(root))[which]
            ordered = _sorted_names(set(names))
            got = ({n: i for i, n in enumerate(ordered)}, [(n.casefold(), n) for n in ordered])
            orders[(parent, which)] = got
        place, keys = got
        if name in place:
            return place[name] + 1, len(keys)
        return bisect.bisect_left(keys, (name.casefold(), name)) + 1, len(keys) + 1

    def add_folder(folder: str, kinds: frozenset[str], toks: tuple[str, ...], group: bool,
                   words: tuple[str, ...]) -> None:
        name = folder.rsplit("/", 1)[-1]
        parent = folder.rsplit("/", 1)[0] if "/" in folder else ""
        k, m = ordinal(parent, 0, name)
        n_files = files_under[folder]
        where = (f"folder {k} of {m} in {_in_folder(screen.display_path(parent) if parent else '')} (sorted by "
                 f"name, ignoring case), with {n_files} file(s) under it")
        folder_flags[folder] = screen.add(name, "folder", kinds, where, written=True, id_tokens=toks,
                                          group_only=group, words=words)

    def add_file(rel: str, kinds: frozenset[str], toks: tuple[str, ...], group: bool,
                 words: tuple[str, ...]) -> None:
        parent, _, name = rel.rpartition("/")
        stem, ext = split_name(name)
        try:
            size = f"{(root / Path(*rel.split('/'))).stat().st_size:,} bytes"
        except OSError:
            size = "size unknown"
        if single:
            where = f"the input file; {ext or 'no extension'}, {size}"
        else:
            k, m = ordinal(parent, 1, name)
            where = (f"file {k} of {m} in {_in_folder(screen.display_path(parent) if parent else '')} (sorted by "
                     f"name, ignoring case); {ext or 'no extension'}, {size}")
        file_flags[rel] = screen.add(stem, "file", kinds, where, written=isinstance(entry_of[rel], FileResult),
                                     id_tokens=toks, group_only=group, words=words)

    file_rels = [rel for rel, _ in sorted(entries, key=lambda e: (e[0].casefold(), e[0])) if not rel.endswith("/")]
    for folder in _sorted_names(folders):
        if folder in path_kinds:
            add_folder(folder, *path_kinds[folder])
    for rel in file_rels:
        if rel in path_kinds:
            add_file(rel, *path_kinds[rel])

    # 2. sheet names and headers, file by file (discovery order), siblings together
    table_flags: dict[int, list[tuple[int, Flag]]] = {}
    keep_titles = bool(getattr(settings, "keep_titles", False))
    for f_i, fr in enumerate(run.files):
        n_tables = len(fr.tables)
        multi = n_tables > 1 or any(tp.name for tp in fr.tables)
        table_names = {tp.name for tp in fr.tables}
        sheets = ((fr.layout.extra or {}).get("sheets") or []) if fr.layout is not None else []
        other_sheets = [(s_i, sh.get("name")) for s_i, sh in enumerate(sheets)
                        if isinstance(sh, dict) and isinstance(sh.get("name"), str) and sh.get("name")
                        and sh.get("name") not in table_names]
        sheet_items = [(t_i, tp.name) for t_i, tp in enumerate(fr.tables) if tp.name]
        sheet_local = [screen.ids.local_keys(_table_id_values(fr.tables[t_i])) if screen.ids else None
                       for t_i, _ in sheet_items]
        stem = split_name(fr.rel_path.rpartition("/")[2])[0]
        sheet_kinds = screen.group_kinds([(name, "") for _, name in sheet_items]
                                         + [(name, "") for _, name in other_sheets],
                                         sheet_local + [None] * len(other_sheets), stem)
        for (t_i, name), (kinds, toks, group) in zip(sheet_items, sheet_kinds):
            if kinds:
                sheet_where = f"sheet {t_i + 1} of {n_tables}" if multi else ""
                table_flags.setdefault(f_i, []).append(
                    (t_i, screen.add(name, "sheet", kinds, f"the name of {sheet_where}", written=True,
                                     id_tokens=toks, group_only=group)))
        for (s_i, name), (kinds, toks, group) in zip(other_sheets, sheet_kinds[len(sheet_items):]):
            if kinds:  # sheets of the workbook that hold no table are named in the layout only: redacted
                screen.add(name, "sheet", kinds, f"the name of workbook sheet {s_i + 1}", written=False,
                           id_tokens=toks, group_only=group)
        for t_i, tp in enumerate(fr.tables):
            sheet_where = f"sheet {t_i + 1} of {n_tables}" if multi else ""
            n_cols = len(tp.columns)
            primary = [tp.header[p.index] if p.index < len(tp.header) else p.name for p in tp.columns]
            col_kinds = screen.group_kinds([(t, "") for t in primary], None, tp.name or stem, headers=True)
            for p, text, (kinds, toks, group) in zip(tp.columns, primary, col_kinds):
                where = f"column {p.index + 1} of {n_cols}" + (f" of {sheet_where}" if sheet_where else "")
                found = [(text, kinds, toks, group)]
                if p.name != text:
                    found.append((p.name, screen.kinds(p.name), (), False))
                for child in _nested_names(p):  # Arrow struct/union field names (kept with --null)
                    if child not in (text, p.name):
                        found.append((child, screen.kinds(child), (), False))
                for t, kinds2, toks2, group2 in found:
                    if kinds2:
                        table_flags.setdefault(f_i, []).append(
                            (t_i, screen.add(t, "column", kinds2, where, written=p.role != "excluded",
                                             number=p.index + 1, id_tokens=toks2, group_only=group2)))
        if keep_titles:  # title/notes rows written verbatim are names the output holds
            for text, where in _verbatim_texts(fr):
                kinds = _verbatim_kinds(screen, text)
                if kinds:
                    table_flags.setdefault(f_i, []).append(
                        (0, screen.add(text, "title", kinds, f"{where} (written verbatim with --keep-titles)",
                                       written=True)))

    # 3. a written name that holds a name flagged on its own elsewhere is flagged too (one name, one verdict), by the
    # same matcher as the leak scanner's (K-16: NameScreen.repeats); never an ID flag on a number alone that its own
    # parent names (K-18: numbered_context)
    input_name = Path(str(getattr(run, "input", "") or "")).name
    _screen_printed_paths(run, settings, screen)  # A5: a printed folder's name is a name of its own too

    def parent_name(path: str) -> str:
        parent = path.rpartition("/")[0]
        return parent.rpartition("/")[2] if parent else input_name

    if screen.standalone_texts():
        _flag_repeats(run, screen, table_flags)
        for folder in _sorted_names(folders):
            name = folder.rsplit("/", 1)[-1]
            if folder not in folder_flags and not screen.kept(name):
                found = screen.repeats(name, numbered_context(parent_name(folder)))
                if found:
                    add_folder(folder, _repeat_kinds(screen, found), (), False, ())
        for rel in file_rels:
            stem, ext = split_name(rel.rpartition("/")[2])
            if rel not in file_flags and not screen.kept(stem, ext):
                found = screen.repeats(stem, numbered_context(parent_name(rel)))
                if found:
                    add_file(rel, _repeat_kinds(screen, found), (), False, ())
    for rel, _ in entries:
        parts = rel.rstrip("/").split("/")
        got = [folder_flags[f] for f in ("/".join(parts[:k]) for k in range(1, len(parts) + 1)) if f in folder_flags]
        if rel in file_flags:
            got.append(file_flags[rel])
        if got:
            by_path[rel] = got

    # 4. names the run only prints: typed references, quoted read messages (the folders of its paths: before step 3)
    _screen_printed(run, settings, screen)
    if not screen:
        return []

    # 5. FAIL the files that would write a flagged name; show redacted paths
    n_failed = 0
    for f_i, fr in enumerate(run.files):
        path_flags = by_path.get(fr.rel_path, [])
        tflags = [(t_i, f) for t_i, f in table_flags.get(f_i, []) if f.written]
        if path_flags or tflags:
            reason, fix = _reason(fr, path_flags, tflags, sharing=getattr(settings, "mode", "") == "sharing")
            fr.fail(reason, fix, "name_holds_personal_data")
            fr.withheld = True
            n_failed += 1
        if path_flags:
            if fr.out_rel_real is None:
                fr.out_rel_real = fr.out_rel_path
            fr.rel_path = screen.display_path(fr.rel_path)
            fr.out_rel_path = screen.display_path(fr.out_rel_path)
        for f in path_flags + [f for _, f in table_flags.get(f_i, [])]:
            if f.file is None:
                f.file = fr.rel_path
    for s in run.skipped:
        flags = by_path.get(s.rel_path)
        if flags:
            s.rel_path = screen.display_path(s.rel_path)
            for f in flags:
                if f.file is None:
                    f.file = s.rel_path
    for f in screen.flags:  # a sheet that holds no table, a folder of skipped entries only, a printed name ...
        if f.file is None:
            f.file = "(input)"
    screen.public_numbers = _public_numbers(run, table_flags, by_path)
    n_written = sum(1 for f in screen.flags if f.written)
    n_texts = len(screen.texts)
    shown = PLACEHOLDER.format(1) if n_texts == 1 else f"{PLACEHOLDER.format(1)} ... {PLACEHOLDER.format(n_texts)}"
    n_other = len(screen.flags) - n_written
    text = (f"{D34_PREFIX}{len(screen.flags)} header, sheet, file, folder or other printed name(s) hold personal "
            f"data ({_kind_counts(screen.flags)}). "
            + (f"{n_failed} file(s) FAILed because their output would hold {n_written} of them"
               if n_written else "None of them is written")
            + (f"; {n_other} {'is' if n_other == 1 else 'are'} never written (an excluded column, a skipped or "
               "excluded file, a sheet without a table, a path or setting the report prints)"
               if n_written and n_other else "")
            + f". The names are never printed: the report, the console and the out-dir docs show {shown}")
    return [NameScreenMessage("info", "personal_names_redacted", text, screen=screen)]


def _run_names(run: Any, children: dict[str, list[tuple[str, str, str]]], file_local: dict[str, frozenset[str]],
               folder_local: dict[str, set[str]]) -> list[tuple[str, frozenset[str] | None, bool]]:
    """Every name the run would write, as ``(text, local IDs, numbered parent)`` (:meth:`IdKeys.note_run_names`): the
    folder and file names of every parent folder (``children`` of :func:`screen_run` step 1), each sheet name and
    each header (also a column's other name and its nested field names)."""
    input_name = Path(str(getattr(run, "input", "") or "")).name
    out: list[tuple[str, frozenset[str] | None, bool]] = []
    for parent, items in children.items():
        context = numbered_context(parent.rpartition("/")[2] if parent else input_name)
        for kind, key, name in items:
            if kind == "folder":
                out.append((name, frozenset(folder_local.get(key, ())), context))
            else:
                out.append((split_name(name)[0], file_local.get(key), context))
    for fr in run.files:
        stem = split_name(fr.rel_path.rpartition("/")[2])[0]
        book = numbered_context(stem)
        sheets = ((fr.layout.extra or {}).get("sheets") or []) if fr.layout is not None else []
        for sh in sheets:
            if isinstance(sh, dict) and isinstance(sh.get("name"), str):
                out.append((sh["name"], None, book))
        for tp in fr.tables:
            if tp.name:
                out.append((tp.name, None, book))
            context = numbered_context(tp.name or stem)
            for p in tp.columns:
                text = tp.header[p.index] if p.index < len(tp.header) else p.name
                for t in {text, p.name, *_nested_names(p)}:
                    if isinstance(t, str):
                        out.append((t, None, context))
    return out


def _flag_repeats(run: Any, screen: NameScreen, table_flags: dict[int, list[tuple[int, Flag]]]) -> None:
    """A header or sheet name not flagged in its own file that holds a flagged name (on its own, never one flagged
    only beside another: a clean ``Okafor`` column next to a typed glob ``*Okafor*`` stays clean) or the real ID of
    a flagged name is flagged too: the same person's name or ID FAILs every file that would write it. The match is
    :meth:`NameScreen.repeats` (K-16: the leak scanner's token sequences too), the context of a sheet its workbook's
    name and of a header its table's (K-18: :func:`numbered_context`)."""
    for f_i, fr in enumerate(run.files):
        done = table_flags.get(f_i, [])
        sheets_done = {t_i for t_i, f in done if f.what == "sheet"}
        cols_done = {(t_i, f.number) for t_i, f in done if f.what == "column"}
        multi = len(fr.tables) > 1 or any(tp.name for tp in fr.tables)
        stem = split_name(fr.rel_path.rpartition("/")[2])[0]
        book_context = numbered_context(stem)
        for t_i, tp in enumerate(fr.tables):
            sheet_where = f"sheet {t_i + 1} of {len(fr.tables)}" if multi else ""
            if tp.name and t_i not in sheets_done and not screen.kept(tp.name):
                found = screen.repeats(tp.name, book_context)
                if found:
                    table_flags.setdefault(f_i, []).append(
                        (t_i, screen.add(tp.name, "sheet", _repeat_kinds(screen, found), f"the name of {sheet_where}",
                                         written=True)))
            table_context = numbered_context(tp.name or stem)
            for p in tp.columns:
                text = tp.header[p.index] if p.index < len(tp.header) else p.name
                if (t_i, p.index + 1) in cols_done or screen.kept(text):
                    continue
                found = screen.repeats(text, table_context)
                if not found:
                    continue
                where = f"column {p.index + 1} of {len(tp.columns)}" + (f" of {sheet_where}" if sheet_where else "")
                table_flags.setdefault(f_i, []).append(
                    (t_i, screen.add(text, "column", _repeat_kinds(screen, found), where,
                                     written=p.role != "excluded", number=p.index + 1)))


def _public_numbers(run: Any, table_flags: dict[int, list[tuple[int, Flag]]],
                    by_path: dict[str, list[Flag]]) -> frozenset[str]:
    """P-10: the numbers alone that the run shows as names it did not flag (a header, column, sheet, file or folder
    name of a file whose details are shown: an attendance workbook's daily sheet ``1007``): a flagged name equal to one
    is no secret of the run's printed text, so the redactor leaves that number alone (:meth:`NameScreen._compiled`)
    and the docs of the file that writes it stay right."""
    out: set[str] = set()
    flagged: set[str] = set()

    def note(name: Any) -> None:
        if isinstance(name, str) and name not in flagged and (d := _bare_number(name)) is not None:
            out.add(d)

    for f_i, fr in enumerate(run.files):
        if fr.withheld:
            continue
        # the names of this file flagged but never written (an excluded column, a sheet without a table) stay secret
        flagged = {f.text for _, f in table_flags.get(f_i, [])} | {f.text for f in by_path.get(fr.rel_path, [])}
        parts = fr.rel_path.split("/")
        for part in parts[:-1]:
            note(part)
        note(split_name(parts[-1])[0])
        for sh in ((fr.layout.extra or {}).get("sheets") or []) if fr.layout is not None else []:
            if isinstance(sh, dict):
                note(sh.get("name"))
        for tp in fr.tables:
            note(tp.name)
            for h in tp.header:
                note(h)
            for c in tp.columns:
                note(c.name)
    return frozenset(out)


def _repeat_kinds(screen: NameScreen, found: Iterable[str]) -> frozenset[str]:
    """The kinds of the flagged names ``found`` (:meth:`NameScreen.repeats`) a repeated name holds."""
    kinds: set[str] = set()
    for t in found:
        kinds |= screen.kinds_of(t)
    return frozenset(kinds or {"name"})


def _screen_printed_paths(run: Any, settings: Any, screen: NameScreen) -> None:
    """The folders of the input, out-dir, report-dir and ``--config`` paths as given (the input root folder, the
    parents of a single-file input): printed in the manifest, the report and the console, never written; redacted
    when flagged. A5 (K-16): screened before step 3 of :func:`screen_run`, so a written name that holds one (a teacher
    folder ``Davis, Summer`` given as the input, holding ``Davis_Summer_Reading_Log.csv``) gets the D34 verdict."""
    for label, p in (("input", getattr(run, "input", None)), ("out dir", getattr(run, "out_dir", None)),
                     ("report dir", getattr(run, "report_dir", None)),
                     ("config", getattr(settings, "config_path", None))):
        if p is None:
            continue
        pp = Path(p)
        parts = [x for x in pp.parts if x not in (pp.anchor, "", ".", "..")]
        if label in ("input", "config") and Path(p).is_file():
            parts = parts[:-1] + [split_name(parts[-1])[0]] if parts else parts
        for part in parts:
            if part in screen.texts or screen.kept(part):
                continue
            kinds = screen.kinds(part)
            if kinds:
                screen.add(part, "path", kinds, f"a folder of the {label} path", written=False)


def _screen_printed(run: Any, settings: Any, screen: NameScreen) -> None:
    """Names the run prints but never writes (module docstring): redacted when flagged, nothing FAILs (the folders of
    the run's paths: :func:`_screen_printed_paths`, before step 3)."""
    # typed references (columns, sheets, ranges, globs): flagged on their own, or a near miss of a flagged name
    typed: list[str] = []
    pub = settings.to_public() if hasattr(settings, "to_public") else {}
    for key, v in pub.items():
        if key in ("keep_name", "config_path"):
            continue
        for item in (v if isinstance(v, list) else [v]):
            if isinstance(item, str) and item.strip():
                typed.append(item)
    seqs = [(_folded_tokens(t), t) for t in screen.texts]
    seqs = [(s, t) for s, t in seqs if len(s) >= 2 or (len(s) == 1 and len(s[0]) >= MIN_ID_LENGTH)]
    parts = [part for item in typed for part in _ref_parts(item)]
    for part in parts:  # flagged on their own or holding a flagged name
        if part in screen.texts or screen.kept(part):
            continue
        kinds = set(screen.kinds(part))
        toks = _folded_tokens(part)
        for seq, t in seqs:
            if _holds_tokens(toks, seq):
                kinds |= screen.kinds_of(t)
        if kinds:
            screen.add(part, "setting", frozenset(kinds), "a column, sheet or path reference of the settings",
                       written=False)
    person_words = {w for f in screen.flags for w in f.words if len(w) >= 3}
    for part in parts:  # a glob or reference naming part of a flagged person's name ('Quixote*/*', '*Okafor*')
        if part in screen.texts or screen.kept(part):
            continue
        if any(name_lists.fold_word(t) in person_words for t in _folded_tokens(part)):
            screen.add(part, "setting", frozenset({"name"}), "a column, sheet or path reference of the settings",
                       written=False, group_only=True)
    # names quoted in messages: read errors and notes of files that hold no table (their names were never
    # screened as headers or sheets), run notes (an unused column reference), skip reasons, tracebacks
    texts: list[str] = []
    for fr in getattr(run, "files", []):
        if fr.tables:
            continue
        texts += list(fr.reasons) + list(fr.fixes)
        for m in fr.all_messages():
            texts += [m.text, m.fix or ""]
    for s in getattr(run, "skipped", []):
        texts.append(s.reason)
    for m in getattr(run, "messages", []):
        texts += [m.text, m.fix or ""]
    texts += list(getattr(run, "debug", []))
    for t in texts:
        for q in quoted_names(t):
            for part in _ref_parts(q):
                if part in screen.texts or screen.kept(part):
                    continue
                kinds = screen.kinds(part)
                if kinds:
                    screen.add(part, "message", kinds, "a name quoted in a message", written=False)


def _reason(fr: FileResult, path_flags: list[Flag], tflags: list[tuple[int, Flag]],
            sharing: bool = False) -> tuple[str, str]:
    """The FAIL reason (locations and kinds, never a name) and the fix of one file."""
    parts: list[str] = []
    for f in path_flags:
        if f.what == "folder":
            parts.append(f"its folder {f.shown} ({kinds_text(f.kinds)}): {f.where}")
        else:
            parts.append(f"its file name {f.shown} ({kinds_text(f.kinds)}): {f.where}")
    sheets = [f for _, f in tflags if f.what == "sheet"]
    for f in sheets:
        parts.append(f"{f.where} ({kinds_text(f.kinds)})")
    titles = [f for _, f in tflags if f.what == "title"]
    for f in titles:
        parts.append(f"{f.where} ({kinds_text(f.kinds)})")
    cols: dict[int, list[Flag]] = {}
    for t_i, f in tflags:
        if f.what == "column":
            cols.setdefault(t_i, []).append(f)
    for t_i, flags in cols.items():
        tp = fr.tables[t_i]
        nums = sorted({f.number for f in flags if f.number is not None})
        sheet = f" of sheet {t_i + 1} of {len(fr.tables)}" if len(fr.tables) > 1 or tp.name else ""
        if len(nums) == 1:
            kinds = frozenset().union(*(f.kinds for f in flags))
            parts.append(f"the header of {flags[0].where} ({kinds_text(kinds)})")
        else:
            per_column: dict[int | None, set[str]] = {}
            for f in flags:  # a column whose raw header and name differ is one column
                per_column.setdefault(f.number, set()).update(f.kinds)
            counts = Counter(k for ks in per_column.values() for k in ks)
            parts.append(f"the headers of {len(nums)} of {len(tp.columns)} columns{sheet} (columns "
                         f"{_ranges(nums)}; "
                         + ", ".join(f"{KIND_TEXT[k]}: {counts[k]}" for k in _KIND_ORDER if counts[k]) + ")")
    reason = (f"{D34_PREFIX}personal data in names this file would write: {'; '.join(parts)}. The names are never "
              "printed (shown as [redacted-N]); the file is not written")
    what = []
    if any(f.what == "folder" for f in path_flags):
        what.append("the folder (or move the files out of it)")
    if any(f.what == "file" for f in path_flags):
        what.append("the file")
    if sheets:
        what.append("the sheet")
    if cols:
        what.append("the column(s)")
    fix = (f"rename {' and '.join(what)} in the source" if what else "")
    if titles:
        fix = (fix + "; " if fix else "") + "drop --keep-titles (title/notes rows are then written as placeholders)"
    fix += ((" or reshape the table to one row per person" if cols else "")
            + "; or skip the file with --exclude-path GLOB")
    if sharing:  # sharing mode refuses --keep-name
        return reason, fix
    fix += "; if a flagged name is not personal data, --keep-name TEXT keeps it as it is"
    if len({f.text for f in path_flags} | {f.text for _, f in tflags}) > 1:
        # D37: a set of sibling names flagged together (a pivot) is kept name by name
        fix += " (one --keep-name per flagged name; a keep_name list in --config keeps a whole set)"
    return reason, fix


__all__ = ["screen_run", "name_kinds", "person_name_kind", "person_name_hit", "NameScreen", "NameScreenMessage",
           "IdKeys", "IdMatch", "run_id_keys", "screen_of", "id_candidates", "date_like", "natural_number",
           "split_name", "D34_PREFIX", "INSTITUTION_WORDS", "CHURCH_TITLES", "DATA_WORDS", "header_words",
           "id_group_flags", "quoted_names", "redact_message", "clean_text", "url_decoded", "nanp_number",
           "person_id_shape", "NUMBERED_WORDS", "decoded", "digits_as_separators", "person_label_before",
           "pivot_flags", "mostly_real_ids", "PERSON_LABEL_WORDS", "numbered_context", "person_label_after", "person_label",
           "person_context", "numbered_word_before"]

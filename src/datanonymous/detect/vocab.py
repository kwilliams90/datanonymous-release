"""The built-in vocabularies whose words quote nobody (design D35).

D35 writes a real text value verbatim only when it is provably safe. One of the ways a value is safe is that it
is a word of a vocabulary every school system shares: :func:`is_vocabulary` is that test, whatever the value's
count. It covers

* the **standard education vocabulary** (:data:`STANDARD_EDUCATION_VOCABULARY` plus the generated grade, term,
  level and group forms of :func:`_generated`): race/ethnicity (federal categories and common SIS spellings and
  codes), gender/sex, grade levels (``PK``, ``JK``, ``K``, ``KG``, ``1``-``12``, ``01``-``12``, ``1st``-``12th``,
  ``Grade 9`` ...), English-learner statuses (ELL/EL/EB/LEP/FEP ...), special education (SPED/IEP/504, the
  disability categories and settings), free/reduced lunch and economic disadvantage (``Free``, ``Reduced``,
  ``Paid``, ``F``/``R``/``P``), yes/no words, terms (``Fall``, ``S1``, ``Q1``-``Q4``, ``T1``-``T3`` ...),
  attendance codes, enrollment and withdrawal statuses, common program codes (``Title I``, ``GT``, ``AP``,
  ``IB``, ``Dual Enrollment``, ``CTE``, ``Gifted``, homeless night-time residences, military-connected
  statuses ...), subjects, home languages, regions and locales (``Northeast``, ``Eastern``, ``Coast``, ``Rural``,
  ``City: Large``), standard NA/placeholder words and suppression tokens;
* the tool's other vocabularies: boolean words (``hints.BOOLEAN_PAIRS``), the ordinal scales
  (``hints.ORDINAL_VOCABS``: grade levels, letter grades, Likert ...), school years, the default NA tokens and
  suppression-like tokens (``detect.numeric.is_suppression_token``: ``*``, ``<10``, ``n<10`` ...).

Matching (:func:`vocab_key`) ignores case, runs of spaces, the spaces around ``/`` and the difference between a
hyphen, an underscore and a space (``Free / Reduced`` = ``free/reduced``, ``Pre-K`` = ``pre k``); joined forms
are listed where they are used (``PreK``), and one word may spell a phrase of the vocabulary without its spaces
(``EconDis`` = ``Econ Dis``, ``Grade9``) when the joined word has a digit or at least six letters (``Daya``,
``Tiera`` are names, not ``Day A`` and ``Tier A``). Letters are never joined across a space or a dot (``Ma Th`` is
not ``Math``, ``E.L.`` is not ``EL``). Three safeguards keep names out:

* an abbreviation of two to four capitals (``GT``, ``ID``, ``GATE``) matches only text written in one case
  (``GT``, ``gt``): ``An``, ``Tu`` and ``Na`` are names, not codes;
* a word that is also a common given name or surname (``White``, ``Black``, ``Summer``, ``Major``, ``Parent``,
  ``English``, ``Lao``, ``Holiday``, ``Sunday``, ``Karen`` ...; and the codes that are surnames written in capitals
  in an all-caps export: ``AN``, ``AU``, ``TU``, ``IP``, ``MENA``, ``ED`` ...: the entries marked ``~``) is
  vocabulary only in a column that also holds an unambiguous word of the same section (``domains`` of
  :func:`is_vocabulary`, from :func:`context_domains`): ``White`` beside ``Asian`` and ``Hispanic`` is a race,
  ``White`` among teacher surnames is a name and follows the D35 count rule;
* only a word with at least two letters vouches for its section (:func:`context_domains`): room numbers
  (``1``-``12``), single letters (``B``, ``H``) and codes such as ``Q1`` or ``P3`` are too common outside the
  vocabulary to make ``Senior`` a grade or ``White`` a race. Words that are mostly given names are left out
  entirely (``Ed`` for economically disadvantaged, ``April``/``May``/``June``/``August``, ``Jan``). A header that
  names a section (:func:`header_domains`: ``Attendance_Code``, ``Race``, ``Home_Language``; never one naming a
  person) and the value describing a code in a one-to-one partner column (``ED`` beside ``Early Dismissal``:
  ``detect.text.partner_domains``) vouch for it too (hardening pass 3).

An Ed-Fi descriptor URI (``uri://ed-fi.org/RaceDescriptor#White``) is judged by its code value
(:func:`descriptor_code`).

Entries of the form ``#race`` (``#`` and lowercase letters) are section markers, not words.

The vocabulary is DATA, not text the tool writes: it is one literal collection of more than 50 entries, so the
leak scanner (``privacy.leaks.tool_vocabulary``) never treats its words as the tool's own text, and a real name
that happens to be a vocabulary word (``White``) is still matched in the report when it comes from a personal
column.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

#: Standard education vocabulary (see the module docstring). ONE literal collection of more than 50 entries
#: (``privacy.leaks.literal_texts`` skips such collections: their words are data, not the tool's text).
STANDARD_EDUCATION_VOCABULARY: tuple[str, ...] = (
    # race / ethnicity: federal categories, common SIS spellings and codes
    "#race",
    "American Indian or Alaska Native", "American Indian/Alaska Native", "American Indian/Alaskan Native",
    "American Indian", "Alaska Native", "Alaskan Native", "Native American", "Indigenous", "First Nations",
    "Asian", "Asian American", "Black or African American", "Black/African American", "African American",
    "~Black", "Native Hawaiian or Other Pacific Islander", "Native Hawaiian/Other Pacific Islander",
    "Native Hawaiian/Pacific Islander", "Native Hawaiian", "Pacific Islander", "Other Pacific Islander",
    "Asian/Pacific Islander", "Asian or Pacific Islander", "Asian Pacific Islander", "~White", "Caucasian",
    "White, Not Hispanic", "Black, Not Hispanic", "Hispanic or Latino", "Hispanic/Latino", "Hispanic/Latinx",
    "Hispanic", "Latino", "Latina", "Latinx", "Latine", "Hispanic or Latino of Any Race", "Not Hispanic or Latino",
    "Not Hispanic", "Non-Hispanic", "Non Hispanic", "Two or More Races", "Two or More", "Two+ Races",
    "Multiracial", "Multi-Racial", "Multiple Races", "Multiple", "Mixed", "Mixed Race", "Biracial",
    "Middle Eastern or North African", "Middle Eastern", "~MENA", "Filipino", "Race Unknown", "Ethnicity Unknown",
    "American Indian or Alaskan Native", "Native Hawaiian or Pacific Islander", "Native Hawaiian or Other Pacific",
    "Native Hawaiian/Pacific Islander", "Hawaiian/Pacific Islander", "Hispanic/Latino of Any Race",
    "Hispanic or Latino of any race", "Hispanic/Latino, Any Race", "Two or More Races, Non-Hispanic",
    "Two or More Races, Not Hispanic", "Multiracial, Non-Hispanic", "Declined to Respond", "Decline to Respond",
    "Declined to Answer", "Declined to Specify", "Not Reported/Declined", "Unknown/Declined",
    "~AI", "AIAN", "AI/AN", "~AN", "AS", "AA", "BL", "BAA", "HI", "HL", "HS", "~PI", "NHPI", "NH/PI", "API", "WH",
    "MR", "TM", "TW", "~MU", "B", "H", "I", "P", "W", "M",
    # gender / sex
    "#gender",
    "Male", "Female", "~Man", "Woman", "Boy", "Girl", "Boys", "Girls", "Men", "Women", "M", "F", "X", "U", "NB",
    "Non-binary", "Nonbinary", "Non Binary", "Gender Nonconforming", "Genderqueer", "Gender Fluid",
    "Transgender", "Trans", "Two-Spirit", "Not Specified", "Unspecified", "Prefer Not to Say",
    "Prefer Not to Answer", "Decline to State", "Declined to State", "Declined", "Not Disclosed",
    "Not Listed", "Another Gender", "Other Gender",
    # grade levels (more forms are generated)
    "#grade",
    "PK", "PK3", "PK4", "PK-3", "PK-4", "Pre-K", "PreK", "Pre-K3", "Pre-K4", "Pre-Kindergarten",
    "Prekindergarten", "Pre-School", "Preschool", "PS", "TK", "Transitional Kindergarten", "K", "KG", "KN",
    "KF", "KH", "Kindergarten", "~Kinder", "Full-Day Kindergarten", "Half-Day Kindergarten", "Infant", "Toddler",
    "IT", "UG", "Ungraded", "Adult", "Adult Education", "Post-Secondary", "Postsecondary", "PG",
    "Post Graduate", "Freshman", "Sophomore", "~Junior", "~Senior", "Freshmen", "Sophomores", "Juniors", "Seniors",
    "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth", "Eleventh",
    "Twelfth", "Elementary", "Middle", "Mid", "High", "Elementary School", "Middle School", "Junior High",
    "Junior High School", "High School", "Intermediate School", "Primary", "Primary School", "Secondary",
    "Secondary School", "K-2", "K-3", "K-5", "K-6", "K-8", "K-12", "PK-5", "PK-8", "PK-12", "3-5", "6-8", "7-8",
    "9-12", "10-12",
    "JK", "SK", "3K", "4K", "K3", "K4", "K5", "EE", "Early Education", "Junior Kindergarten", "Senior Kindergarten",
    # English learners
    "#ell",
    "ELL", "EL", "ELs", "ELLs", "LEP", "FEP", "IFEP", "RFEP", "R-FEP", "I-FEP", "EO", "TBD", "English Only",
    "English Learner", "English Learners", "English Language Learner", "English Language Learners",
    "Limited English Proficient", "Limited English Proficiency", "Fluent English Proficient",
    "Initially Fluent English Proficient", "Reclassified Fluent English Proficient", "Reclassified",
    "Redesignated", "Monitor", "Monitored", "Monitoring", "M1", "M2", "M3", "M4", "Exited", "Exited EL",
    "Former EL", "Former ELL", "Former LEP", "Ever EL", "Ever ELL", "Never EL", "Never ELL", "Non-ELL", "Non-EL",
    "Non-LEP", "Not ELL", "Not EL", "Not LEP", "ESL", "ESOL", "ELD", "ENL", "Bilingual", "~Newcomer",
    "Long-Term EL", "Long-Term English Learner", "LTEL", "At-Risk EL", "Parent Waived", "Waived", "Refused",
    "Opt Out", "Opted Out", "Entering", "Emerging", "Developing", "Expanding", "Bridging", "Reaching",
    "Beginning", "Beginner", "Early Intermediate", "Intermediate", "Early Advanced", "Advanced", "Fluent",
    "Proficient", "Not Proficient", "Nearly Proficient", "Partially Proficient", "Limited", "Moderate",
    "EB", "Emergent Bilingual", "Emergent Bilinguals", "Non-EB", "Not EB", "Former EB", "EB/EL", "Monitored Year 1",
    "Monitored Year 2", "Monitored Year 3", "Monitored Year 4", "Monitor Year 1", "Monitor Year 2",
    # special education / IEP / 504
    "#sped",
    "SPED", "SpEd", "Special Education", "Special Ed", "Sp Ed", "IEP", "Has IEP", "No IEP", "504", "504 Plan",
    "Section 504", "Has 504", "No 504", "Gen Ed", "GenEd", "General Education", "Regular Education", "Reg Ed",
    "SWD", "Students with Disabilities", "Student with Disability", "Student with a Disability", "Not SWD",
    "Non-SWD", "Disability", "No Disability", "Disabled", "Not Disabled", "Gifted", "Gifted and Talented",
    "Twice Exceptional", "2e", "Autism", "AUT", "Autism Spectrum Disorder", "ASD", "SLD",
    "Specific Learning Disability", "Learning Disability", "LD", "SLI", "Speech or Language Impairment",
    "Speech/Language Impairment", "Speech Impairment", "Language Impairment", "Speech", "OHI",
    "Other Health Impairment", "Other Health Impaired", "EBD", "EBD/ED", "E/BD", "Emotional Disturbance",
    "Emotional Disability", "Emotional/Behavioral Disability", "Emotionally Disturbed", "~ED", "EMD", "ID",
    "Intellectual Disability", "Intellectual Disabilities", "MD", "Multiple Disabilities", "HI/D",
    "Hearing Impairment", "Hearing Impaired", "Deaf", "Deafness", "Hard of Hearing", "DHH", "~VI",
    "Visual Impairment", "Visual Impairment Including Blindness", "Visually Impaired", "Blind", "Blindness",
    "DB", "Deaf-Blindness", "Deaf-Blind", "OI", "Orthopedic Impairment", "Orthopedically Impaired", "TBI",
    "Traumatic Brain Injury", "DD", "Developmental Delay", "Developmentally Delayed", "Inclusion",
    "Deaf/Hard of Hearing", "Deaf or Hard of Hearing", "Speech Language Impairment", "Speech Language Impaired",
    "Speech Impaired", "Language Impaired", "Speech/Language Impaired", "Learning Disabled", "Learning Disabilities",
    "Specific Learning Disabilities", "Cognitive Disability", "Cognitive Impairment", "Cognitively Impaired",
    "Emotional Behavioral Disability", "Emotional Behavioral Disorder", "Emotional/Behavioral Disorder",
    "Emotional Impairment", "Emotionally Impaired", "Significant Developmental Delay", "Developmental Delays",
    "Multiple Disability", "Other Health Impairments", "Hearing Impairments", "Visual Impairments",
    "Orthopedic Impairments", "Deaf/Blind", "Dyslexia",
    "Resource", "Resource Room", "Self-Contained", "Co-Taught", "Push-In", "Pull-Out", "Consult",
    "Separate School", "Separate Class", "Residential Facility", "Regular Class", "Homebound/Hospital",
    "Regular Class 80%+", "Regular Class 40-79%", "Regular Class <40%", "Inside Regular Class 80% or More",
    "Inside Regular Class 40-79%", "Inside Regular Class Less Than 40%", "Correctional Facility",
    "Parentally Placed in Private School",
    # free / reduced lunch, economic disadvantage
    "#frl",
    "Free", "Reduced", "Paid", "Full Pay", "Full Price", "Full", "Free Lunch", "Reduced Lunch",
    "Reduced Price", "Reduced-Price", "Reduced Price Lunch", "F/R", "F/R/P", "Free/Reduced", "Free / Reduced",
    "Free-Reduced", "Free and Reduced", "Free or Reduced", "Free & Reduced", "Free/Reduced Price",
    "Free/Reduced Lunch", "Free or Reduced Price", "Free and Reduced Price", "FRL", "FRPL", "FRM", "Non-FRL",
    "Not FRL", "NSLP", "CEP", "Direct Certification", "Direct Cert", "Directly Certified", "Eligible",
    "Not Eligible", "Ineligible", "Econ Dis", "Econ Disadv", "Economically Disadvantaged",
    "Not Economically Disadvantaged", "Non-Economically Disadvantaged", "Non Economically Disadvantaged",
    "Economic Disadvantage", "ECD", "ECON", "Non-ED", "Not ED", "Low Income", "Not Low Income", "Low-Income",
    "Standard", "Standard Lunch",
    "R", "FDC", "RDC", "Free-Direct", "Reduced-Direct", "Free Direct", "Reduced Direct",
    "Not Applied", "No Application", "Application Not Returned", "Free Meals", "Reduced Meals", "Paid Meals",
    "Free Meal", "Reduced Meal", "Reduced-Price Meals", "Free - CEP", "Free CEP", "CEP Free", "Direct Certified",
    "Free - Direct Cert", "Free - Application", "Reduced - Application",
    # yes / no style
    "#yesno",
    "Y", "N", "Yes", "No", "True", "False", "T", "Y/N", "Checked", "Unchecked", "On", "Off", "Pass", "Fail",
    "Passed", "Failed", "Approved", "Denied", "Accepted", "Rejected", "Agree", "Disagree", "Always", "Never",
    "Sometimes", "Often", "Rarely", "Usually", "Seldom",
    # terms and periods (more forms are generated)
    "#terms",
    "Fall", "Spring", "~Summer", "~Winter", "~Autumn", "Fall Semester", "Spring Semester", "Summer Session",
    "Summer School", "Intersession", "Full Year", "Full-Year", "Year", "Yearlong", "Year-Long", "Annual", "FY",
    "YR", "Semester", "Quarter", "Trimester", "Term", "Session", "First Semester", "Second Semester",
    "First Quarter", "Second Quarter", "Third Quarter", "Fourth Quarter", "First Trimester", "Second Trimester",
    "Third Trimester", "BOY", "~MOY", "EOY", "Beginning of Year", "Middle of Year", "End of Year", "Fall BOY",
    "Winter MOY", "Spring EOY", "Midterm", "Final", "Finals", "Final Exam", "Progress Report", "Report Card",
    "#calendar",
    "~Monday", "~Tuesday", "~Wednesday", "Thursday", "~Friday", "Saturday", "~Sunday", "Mon", "Tue", "Tues", "Wed",
    "Thur", "Thurs", "Fri", "Sat", "Weekday", "Weekend", "~January", "February", "~March", "July",
    "September", "October", "November", "December", "Feb", "Mar", "Apr", "Jul", "Aug", "Sep", "Sept",
    "Oct", "Nov", "Dec", "AM", "PM", "Morning", "Afternoon", "Evening", "Before School", "After School",
    # attendance codes
    "#attendance",
    "Present", "Absent", "Tardy", "Late", "Excused", "Unexcused", "Excused Absence", "Unexcused Absence",
    "Excused Tardy", "Unexcused Tardy", "Tardy Excused", "Tardy Unexcused", "Absent Excused",
    "Absent Unexcused", "Early Dismissal", "Early Release", "Left Early", "Early Checkout", "Late Arrival",
    "Truant", "Truancy", "Chronic Absence", "Chronically Absent", "In-School Suspension",
    "Out-of-School Suspension", "In School Suspension", "Out of School Suspension", "ISS", "OSS", "Suspended",
    "Suspension", "Expelled", "Expulsion", "Medical", "Illness", "Ill", "Sick", "~Doctor", "Medical Appointment",
    "Appointment", "Field Trip", "School Activity", "School Business", "School Sponsored", "Religious",
    "Religious Holiday", "Religious Observance", "Bereavement", "Funeral", "Family Emergency", "Emergency",
    "~Court", "Legal", "Quarantine", "Remote", "Virtual", "Distance Learning", "Online", "Hybrid", "In Person",
    "In-Person", "Homebound", "Hospital", "Hospitalized", "Partial Day", "Half Day", "Full Day", "AE", "~AU",
    "UA", "EA", "~TE", "~TU", "UT", "ET", "EX", "UX", "UE", "EXC", "UNX", "UNEX", "ABS", "TDY", "TAR", "SUS",
    "MED", "FT", "~SA", "HB", "Unverified", "Verified", "No School", "~Holiday", "Snow Day", "Weather",
    "Not Enrolled", "Non-Attendance",
    "L", "V", "~ED", "~LE", "DR", "CT", "Court Appearance", "Doctor Appointment", "Dental", "Dental Appointment",
    # enrollment / withdrawal statuses
    "#enrollment",
    "Active", "Inactive", "Enrolled", "Pre-Registered", "Preregistered", "Pre-Enrolled", "Registered",
    "Unenrolled", "Withdrawn", "Withdrawal", "Withdrew", "Transferred", "Transfer", "Transfer In",
    "Transfer Out", "Transferred In", "Transferred Out", "Graduated", "Graduate", "Graduates", "Completer",
    "Completed", "Not Completed", "Non-Completer", "Dropout", "Dropped Out", "Dropped", "Drop", "Promoted",
    "Retained", "Retention", "Demoted", "Continuing", "Continuing Student", "New", "New Student", "Returning",
    "Returning Student", "Re-entry", "Reentry", "Re-enrolled", "Reenrolled", "Readmitted", "No Show", "No-Show",
    "Deceased", "Moved", "Moved Out of State", "Moved Out of District", "Moved Out of Country",
    "Moved Within District", "Private School", "Home School", "Homeschool", "Home Schooled", "Charter School",
    "~Charter", "GED", "HSE", "Certificate", "Certificate of Completion", "Diploma", "Standard Diploma",
    "Advanced Diploma", "Special Diploma", "Aged Out", "Incarcerated", "Pending", "Admitted", "Waitlisted",
    "Foundation", "Foundation Plan", "FHSP", "FHSP-E", "FHSP-DLA", "DLA", "Distinguished Level of Achievement",
    "RHSP", "DAP", "MHSP", "Endorsement", "Recommended", "Distinguished",
    "Wait List", "Current", "Former", "Concurrent", "~Main", "Full Time", "Full-Time", "Part Time", "Part-Time",
    "FTE", "Entry", "Exit", "Entered", "Exited", "Cohort", "On Track", "Off Track", "At Risk", "At-Risk",
    "Not At Risk", "Resident", "Non-Resident", "Nonresident", "In District", "Out of District",
    "Out-of-District", "Inter-District", "Intra-District", "Open Enrollment", "Choice", "School Choice",
    "Tuition", "Tuition Paying", "Foreign Exchange", "Exchange Student",
    "GED Program", "Transfer to Private", "Transfer to Private School", "Transfer to Public School",
    # programs
    "#programs",
    "Title I", "Title 1", "Title I-A", "Title IA", "Title I Schoolwide", "Title I Targeted", "Title III",
    "Title 3", "Title IX", "Not Title I", "Non-Title I", "Migrant", "Migrant Education", "Homeless",
    "McKinney-Vento", "McKinney Vento", "Doubled Up", "Doubled-Up", "Shelter", "Hotel/Motel", "Unsheltered",
    "Unaccompanied Youth", "~Foster", "Foster Care", "In Foster Care", "Military", "Military Connected",
    "Military Family", "Active Duty", "Active Duty Military", "National Guard", "Reserve", "Reserves",
    "Reserve Component", "Guard/Reserve", "Veteran", "Not Military Connected", "Immigrant", "Refugee",
    "GT", "G/T", "GATE", "TAG", "Talented and Gifted", "Highly Gifted",
    "AP", "Advanced Placement", "IB", "International Baccalaureate", "Pre-AP", "PreAP", "Pre-IB", "Honors",
    "Accelerated", "Dual Enrollment", "Dual Credit", "Concurrent Enrollment", "Early College",
    "Early College High School", "CTE", "Career and Technical Education", "Career & Technical Education",
    "Career Technical Education", "Career Tech", "Vocational", "Vocational Education", "JROTC", "ROTC", "AVID",
    "Magnet", "STEM", "STEAM", "Dual Language", "Dual Language Immersion", "Immersion", "Two-Way Immersion",
    "~Montessori", "Head Start", "Early Head Start", "Early Childhood", "Early Childhood Special Education",
    "ECSE", "After School Program", "Extended Day", "Extended School Year", "ESY", "Credit Recovery",
    "Alternative", "Alternative Education", "Alternative School", "Continuation", "Independent Study",
    "Virtual School", "Online School", "RTI", "MTSS", "PBIS", "Tier 1", "Tier 2", "Tier 3", "Tier I", "Tier II",
    "Tier III", "Intervention", "Tutoring", "Enrichment", "Remediation", "Remedial", "Speech Therapy",
    "Occupational Therapy", "Physical Therapy", "OT", "PT", "Counseling", "Social Work", "Case Management",
    "General", "Regular", "Core", "Elective", "Required", "Optional", "None", "Other", "Not Applicable",
    "Shelters", "Sheltered", "Hotels/Motels", "Hotel", "Motel", "Hotels", "Motels", "Transitional Housing",
    "Emergency Shelter",
    "Title I Part A", "Title I, Part A", "Title 1 Part A", "Title 1, Part A", "Title III Part A", "Title III, Part A",
    "Neglected or Delinquent", "Neglected", "Delinquent", "N or D", "Hospital/Homebound", "Hospital Homebound",
    "Gifted/Talented", "Bilingual/ESL", "ESL Program", "Bilingual Program", "Section 504 Plan", "Career Pathway",
    # assessment and performance labels
    "#assessment",
    "Exceeds", "Exceeded", "Exceeds Expectations", "Exceeds Standard", "Meets", "Met", "Meets Expectations",
    "Meets Standard", "Approaches", "Approaching", "Approaches Expectations", "Partially Meets",
    "Partially Met", "Does Not Meet", "Did Not Meet", "Not Met", "Nearly Met", "~Masters", "Mastered", "Mastery",
    "Not Mastered", "Below Basic", "Basic", "Above Basic", "Below Standard", "At Standard", "Above Standard",
    "Below Grade Level", "On Grade Level", "Above Grade Level", "Below Benchmark", "At Benchmark",
    "Above Benchmark", "Well Below", "Below", "Above", "Well Above", "At", "~Low", "Medium", "High", "Very Low",
    "Very High", "Low Average", "Average", "High Average", "Satisfactory", "Unsatisfactory",
    "Needs Improvement", "Not Tested", "Not Assessed", "Exempt", "Incomplete", "In Progress", "Not Started",
    "Withdrawn Passing", "Withdrawn Failing", "NP", "WP", "WF", "AUD", "CR", "NC", "NM", "~IP", "Credit",
    "No Credit", "S", "O", "E", "G", "Excellent", "~Good", "~Fair", "~Poor", "Outstanding",
    "Masters Grade Level", "Meets Grade Level", "Approaches Grade Level", "Did Not Meet Grade Level",
    "Approaches Grade Level or Above", "Meets Grade Level or Above", "LoAvg", "Avg", "HiAvg", "Lo Avg",
    "Hi Avg", "Low Avg", "High Avg", "Novice", "Apprentice", "Practitioner", "Expert", "Exemplary",
    "Highly Proficient", "Not Yet Proficient", "Minimally Proficient", "Beginning Learner", "Developing Learner",
    "Proficient Learner", "Distinguished Learner", "Emerging Learner", "Not Yet Met", "Did Not Yet Meet",
    "Partially Meets Expectations", "Does Not Meet Expectations", "Approached Expectations", "Met Expectations",
    "Exceeded Expectations", "Did Not Meet Expectations", "Partially Met Expectations",
    # discipline
    "#discipline",
    "Detention", "Warning", "Verbal Warning", "Written Warning", "Referral", "Office Referral", "Conference",
    "Parent Contact", "Parent Conference", "Saturday School", "Lunch Detention", "Alternative Placement",
    "Restorative", "Fighting", "Disruption", "Disruptive Behavior", "Defiance", "Insubordination", "Bullying",
    "Harassment", "Vandalism", "Theft", "Drugs", "Alcohol", "Tobacco", "Vaping", "Weapon", "Weapons",
    "Dress Code", "Cell Phone", "Profanity", "Cheating", "Skipping", "Skipping Class", "Assault", "Threat",
    "~Minor", "~Major", "Level 1 Offense", "Level 2 Offense", "Level 3 Offense", "DAEP", "DAEP Placement",
    "JJAEP", "JJAEP Placement", "Expulsion Placement", "Tardiness", "Truancy Referral", "Arson", "Trespassing",
    "Gang Activity", "Robbery", "Burglary", "Battery", "Physical Aggression", "Verbal Aggression", "Cyberbullying",
    "Hazing", "Horseplay", "Forgery", "Gambling", "Extortion", "Drug Possession", "Alcohol Possession",
    "Weapon Possession", "Tobacco Possession", "Leaving Campus", "Inappropriate Language", "Electronic Device",
    "Defiance/Insubordination", "Insubordination/Defiance", "Dress Code Violation", "Cell Phone Violation",
    "Electronic Device Violation", "Disrespect", "Disrespectful Behavior", "Disruptive Conduct", "Class Disruption",
    "Physical Altercation", "Verbal Altercation", "Tardies", "Excessive Tardies", "Bus Misconduct", "Misconduct",
    "Persistent Misbehavior", "Other Misconduct", "Mutual Combat", "Possession", "Vape", "E-Cigarette",
    # transportation
    "#transportation",
    "Bus", "Bus Rider", "Walk", "Car", "Car Rider", "Parent Pickup", "Parent Drop-Off", "~Parent",
    "Daycare", "Day Care", "After Care", "Aftercare", "Bike", "Bicycle", "~Van", "Drive", "~Driver", "Self",
    "Public Transit", "Public Transportation", "Not Eligible for Transportation",
    # subjects
    "#subjects",
    "Math", "Maths", "Mathematics", "Algebra", "Algebra I", "Algebra II", "Algebra 1", "Algebra 2", "Geometry",
    "Precalculus", "Pre-Calculus", "Calculus", "Statistics", "Biology", "Chemistry", "Physics", "Science",
    "Earth Science", "Life Science", "Physical Science", "Environmental Science", "Social Studies", "History",
    "US History", "U.S. History", "World History", "Government", "Civics", "Economics", "Geography",
    "Psychology", "~English", "ELA", "English Language Arts", "Language Arts", "~Reading", "Writing",
    "Literature", "Literacy", "Phonics", "~Art", "Arts", "Visual Arts", "Music", "Band", "Choir", "Orchestra",
    "Drama", "Theater", "Theatre", "~Dance", "PE", "P.E.", "Physical Education", "Health", "Health Education",
    "Computer Science", "Computers", "Technology", "Engineering", "Foreign Language", "World Language",
    "World Languages", "Homeroom", "Advisory", "Study Hall", "Lunch", "Breakfast", "Recess", "Library",
    "Assessment", "Test", "Exam", "Quiz", "Homework", "Project", "Participation", "Attendance", "Behavior",
    # home languages
    "#languages",
    "Spanish", "Vietnamese", "Arabic", "Chinese", "Mandarin", "Cantonese", "Somali", "Hmong", "Korean",
    "Russian", "Tagalog", "Haitian Creole", "Creole", "Portuguese", "~French", "Japanese", "Hindi", "Urdu",
    "Punjabi", "Bengali", "Farsi", "Persian", "Ukrainian", "Swahili", "Amharic", "Nepali", "Burmese",
    "Kinyarwanda", "Pashto", "~Dari", "Polish", "~German", "Italian", "Greek", "Navajo", "Gujarati",
    "Telugu", "~Tamil", "~Thai", "~Lao", "Khmer", "Cambodian", "Turkish", "Hebrew", "Yiddish", "Marshallese",
    "Chuukese", "Samoan", "Tongan", "Ilocano", "Sign Language", "ASL", "American Sign Language", "~Karen",
    "Mam", "Oromo", "Afaan Oromo", "Tigrinya", "Tigrigna", "Kirundi", "Lingala", "Mixteco", "Mixtec", "K'iche'",
    "Q'anjob'al", "Rohingya", "Dinka", "Nuer", "Yoruba", "Igbo", "Hausa", "Wolof", "Twi", "Fulani", "Kurdish",
    "Uzbek", "Albanian", "Bosnian", "Serbian", "Croatian", "Romanian", "Hungarian", "Dutch", "Armenian",
    "Malayalam", "Marathi", "Kannada", "Sinhala", "Indonesian", "Malay", "~Chamorro", "Hawaiian", "Ojibwe",
    "Cherokee", "Lakota", "Yupik", "Quechua", "Haitian", "Patois", "Jamaican Creole", "Cape Verdean Creole",
    "Other Language", "English and Other", "Language Other Than English",
    # regions and locales: compass regions, Census regions, NCES urban-centric locales
    "#region",
    "Northern", "Southern", "Eastern", "Western", "Central", "North Central", "South Central", "Northeast",
    "Northwest", "Southeast", "Southwest", "Northeastern", "Northwestern", "Southeastern", "Southwestern",
    "North Eastern", "North Western", "South Eastern", "South Western", "Midwest", "Midwestern",
    "New England", "Mid-Atlantic", "~Mountain", "Pacific", "Coast", "Coastal", "Gulf Coast", "East Coast",
    "West Coast", "Inland", "Upstate", "Downstate", "~Metro", "Metropolitan", "Non-Metro", "Nonmetropolitan",
    "Region", "~Urban", "Suburban", "Rural", "Town", "Suburb", "City", "City: Large", "City: Midsize",
    "City: Small", "Suburb: Large", "Suburb: Midsize", "Suburb: Small", "Town: Fringe", "Town: Distant",
    "Town: Remote", "Rural: Fringe", "Rural: Distant", "Rural: Remote", "Large City", "Midsize City",
    "Small City", "Fringe", "Distant", "Remote Rural", "~North", "~South", "~East", "~West",
    # missing, NA and placeholder words
    "#missing",
    "NA", "N/A", "N.A.", "n/a", "NULL", "Null", "None Reported", "NaN", "#N/A", "-", "--", "---", ".", "?",
    "*", "**", "***", "Missing", "Unknown", "Unk", "Not Known", "~Blank", "Empty", "Not Available",
    "Not Reported", "Not Provided", "Not Recorded", "Not Entered", "Not Collected", "No Data", "No Response",
    "No Answer", "No Value", "Undefined", "Undetermined", "To Be Determined", "TBA", "Not Applicable (N/A)",
    "Does Not Apply", "Not Answered", "Unanswered", "Refused to Answer", "Other/Unknown", "Unknown/Other",
    "Other (Specify)", "Other - Specify", "All", "Some", "Most", "Few", "Several", "Mild", "Severe", "Group",
    "Total", "Subtotal", "District", "School", "State", "Statewide", "Districtwide", "Schoolwide", "Campus",
)

#: Spelled ordinals for the generated grade forms ("Ninth Grade").
_ORDINAL_WORDS = ("First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth",
                  "Eleventh", "Twelfth")
#: Joiners that match each other and a space in :func:`vocab_key` (``Pre-K`` = ``Pre K`` = ``pre_k``).
_JOINERS = re.compile("[-_‐‑–]+")
_SPACES = re.compile(r"\s+")
_SLASH = re.compile(r"\s*/\s*")
#: A section marker of :data:`STANDARD_EDUCATION_VOCABULARY` (``#race``), not a word.
_MARKER = re.compile("#[a-z]+")
#: Fewest letters of a word that vouches for its section (:func:`context_domains`).
CONTEXT_MIN_LETTERS = 2
#: Fewest letters of a phrase written as one word (``EconDis``) unless it has a digit (``Grade9``, ``Tier1``).
JOINED_MIN_LETTERS = 6


def vocab_key(text: str) -> str:
    """Case- and spacing-insensitive form of a vocabulary word (module docstring): ``Free / Reduced`` ->
    ``free/reduced``, ``Pre-K`` -> ``pre k``, ``Ma Th`` -> ``ma th``."""
    s = str(text).replace("’", "'").casefold()
    s = _SLASH.sub("/", _JOINERS.sub(" ", s))
    return _SPACES.sub(" ", s).strip()


def _suffix(n: int) -> str:
    return "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _generated() -> list[tuple[str, str]]:
    """Grade, term, level and group forms that follow a pattern (``Grade 9``, ``9th Grade``, ``Q3``, ``Level 4``),
    each with its section."""
    out: list[tuple[str, str]] = []
    for g in range(0, 14):
        ordn = f"{g}{_suffix(g)}"
        words = [str(g), f"{g:02d}", ordn, f"Grade {g}", f"Grade {g:02d}", f"Gr {g}", f"Gr. {g}", f"Grd {g}",
                 f"G{g}", f"{ordn} Grade", f"Grade {ordn}", f"Year {g}", f"Yr {g}"]
        if 1 <= g <= 12:
            words += [f"{_ORDINAL_WORDS[g - 1]} Grade", f"Grade {_ORDINAL_WORDS[g - 1]}"]
        out += [(w, "#grade") for w in words]
    for k in ("K", "KG", "PK", "TK", "JK", "SK", "Pre-K", "Kindergarten", "Pre-Kindergarten"):
        out += [(f"Grade {k}", "#grade"), (f"Gr {k}", "#grade")]
    for i in range(1, 7):
        for stem in ("Q", "Quarter ", "Qtr ", "S", "Sem ", "Semester ", "T", "Tri ", "Trimester ", "Term ", "MP",
                     "Marking Period ", "Period ", "P", "Block ", "H", "Half ", "Y"):
            out.append((f"{stem}{i}", "#terms"))
        for stem in ("Level ", "Tier ", "Cycle ", "Unit ", "Round ", "Window "):
            out.append((f"{stem}{i}", "#assessment"))
    for i in range(1, 7):
        out += [(f"{_ORDINAL_WORDS[i - 1]} {w}", "#terms") for w in ("Period", "Block", "Term", "Semester", "Quarter")]
    for letter in "ABCDEFGH":
        out += [(f"Group {letter}", "#assessment"), (f"Level {letter}", "#assessment"),
                (f"Tier {letter}", "#assessment"), (f"Block {letter}", "#terms"), (f"Day {letter}", "#terms")]
    # hardening pass 3: a term with its year (``Fall 2023``, ``Spring 2024``, ``FY 2024``) and the race categories
    # with their ethnicity (``White, Non-Hispanic``, ``Non-Hispanic Black``). The words come from the sections of
    # STANDARD_EDUCATION_VOCABULARY (data), so no short literal list makes them the tool's own text
    terms = [w for w in _section_words("#terms") if w.isalpha() or _abbreviation(w)]
    out += [(f"{t} {y}", "#terms") for t in terms for y in range(1990, 2051)]
    races = [w for w in _section_words("#race")
             if not _abbreviation(w) and len(w) > 1 and not any(x in w.casefold() for x in ("hispanic", "latin",
                                                                                           "unknown"))]
    for race in races:
        for form in ("{r}, Non-Hispanic", "{r}, Not Hispanic", "{r}, Not Hispanic or Latino", "Non-Hispanic {r}",
                     "{r} (Non-Hispanic)", "{r}, Hispanic", "Hispanic {r}"):
            out.append((form.format(r=race), "#race"))
    return out


def _section_words(section: str) -> list[str]:
    """The words of one section of :data:`STANDARD_EDUCATION_VOCABULARY` (``~`` marks removed)."""
    out: list[str] = []
    current = ""
    for w in STANDARD_EDUCATION_VOCABULARY:
        if _MARKER.fullmatch(w):
            current = w
        elif current == section:
            out.append(w.removeprefix("~"))
    return out


def _abbreviation(word: str) -> bool:
    """Two to four capitals, digits or ``/`` with at least one capital: ``AN``, ``GATE``, ``M1``, ``F/R``."""
    return (2 <= len(word) <= 4 and any(ch.isalpha() for ch in word)
            and all(ch.isupper() or ch.isdigit() or ch == "/" for ch in word))


class _Index:
    """The parsed vocabulary: the keys matched in any case, the keys of abbreviations (matched in one case only),
    the sections of each key and the keys that are also common names (entries marked ``~``)."""

    def __init__(self) -> None:
        self.any_case: set[str] = set()
        self.one_case: set[str] = set()
        self.sections: dict[str, set[str]] = {}
        self.name_like: set[str] = set()
        section = ""
        entries = [(w, None) for w in STANDARD_EDUCATION_VOCABULARY] + list(_generated())
        for word, generated in entries:
            if generated is None and _MARKER.fullmatch(word):
                section = word
                continue
            name_like = word.startswith("~")
            word = word.removeprefix("~")
            key = vocab_key(word)
            if not key:
                continue
            (self.one_case if _abbreviation(word) else self.any_case).add(key)
            self.sections.setdefault(key, set()).add(generated or section)
            if name_like:
                self.name_like.add(key)
        self.one_case -= self.any_case  # a key also listed in another spelling matches in any case
        #: a phrase of the vocabulary written as one word (``EconDis`` = ``Econ Dis``): its key without spaces, when
        #: that word has a digit or at least JOINED_MIN_LETTERS letters (``Daya`` is a name, not ``Day A``)
        self.joined = {j: k for k in self.any_case if ' ' in k
                       for j in (k.replace(' ', ''),)
                       if any(ch.isdigit() for ch in j) or sum(ch.isalpha() for ch in j) >= JOINED_MIN_LETTERS}


@lru_cache(maxsize=1)
def _index() -> _Index:
    return _Index()


def standard_keys() -> frozenset[str]:
    """The :func:`vocab_key` of every word of the standard education vocabulary (literal and generated)."""
    ix = _index()
    return frozenset(ix.any_case | ix.one_case)


def _mixed_case(text: str) -> bool:
    return any(ch.islower() for ch in text) and any(ch.isupper() for ch in text)


#: An Ed-Fi descriptor URI (``uri://ed-fi.org/RaceDescriptor#White``, a state's ``uri://tea.texas.gov/...``): its
#: code value after ``#`` is the word judged (hardening pass 3); the namespace is the vendor's or the state's.
_DESCRIPTOR_RE = re.compile(r"uri://[^\s#/]+(?:/[^\s#]*)?Descriptor#(?P<code>\S.*)", re.I)


def descriptor_code(text: str) -> str:
    """The code value of an Ed-Fi descriptor URI (``uri://ed-fi.org/SexDescriptor#Female`` -> ``Female``), else
    ``text`` itself."""
    if "#" not in text or "descriptor#" not in text.casefold():
        return text
    m = _DESCRIPTOR_RE.fullmatch(text.strip())
    return m.group("code").strip() if m else text


def _standard_key(text: str) -> str | None:
    """The key of ``text`` when it is written as a word of the standard education vocabulary, else None."""
    text = descriptor_code(text)
    key = vocab_key(text)
    ix = _index()
    if key in ix.any_case or (key in ix.one_case and not _mixed_case(text)):
        return key
    if " " not in key and key in ix.joined:  # one word spelling a phrase (``EconDis``, ``Grade9``)
        return ix.joined[key]
    return None


def is_standard_word(text: str) -> bool:
    """``text`` is a word of the standard education vocabulary (case- and spacing-insensitive; an abbreviation in
    one case only). A name-like word counts here; :func:`is_vocabulary` also asks for its section."""
    return _standard_key(text) is not None


def is_name_like(text: object) -> bool:
    """``text`` is a vocabulary word that is also a common given name or surname (``White``, ``Summer``)."""
    return text is not None and vocab_key(descriptor_code(str(text).strip())) in _index().name_like


def context_domains(texts: Iterable[object]) -> frozenset[str]:
    """The vocabulary sections of the unambiguous standard words among ``texts`` (a column's units): the sections
    in which a name-like word of the same column is read as vocabulary (:func:`is_vocabulary`). Only a word with at
    least :data:`CONTEXT_MIN_LETTERS` letters vouches for its section: numbers (room numbers ``1``-``12``), single
    letters (``B``, ``H``) and one-letter codes (``Q1``, ``P3``) are written in too many columns to make ``Senior`` a
    grade or ``White`` a race (hardening pass 2)."""
    ix = _index()
    out: set[str] = set()
    for t in texts:
        if t is None:
            continue
        s = descriptor_code(str(t).strip())
        if sum(ch.isalpha() for ch in s) < CONTEXT_MIN_LETTERS:
            continue
        key = _standard_key(s)
        if key is not None and key not in ix.name_like:
            out |= ix.sections.get(key, set())
    return frozenset(out)


#: Header words that name a section of the vocabulary (hardening pass 3): under ``Attendance_Code``, ``Race``,
#: ``Home_Language`` or ``Term`` a name-like word of that section is read in it (``ED`` early dismissal, ``White``,
#: ``Karen``, ``Summer``), as beside an unambiguous word of the section. A header with a person word names no section
#: (``Teacher_Language``: the column may hold names). ONE literal mapping of more than 50 entries: data, not text the
#: tool writes.
HEADER_SECTIONS: dict[str, str] = {
    "race": "#race", "races": "#race", "ethnicity": "#race", "ethnicities": "#race", "ethnic": "#race",
    "raceeth": "#race", "raceethnicity": "#race", "gender": "#gender", "sex": "#gender", "grade": "#grade",
    "grades": "#grade", "grd": "#grade", "gradelevel": "#grade", "ell": "#ell", "lep": "#ell", "esl": "#ell",
    "elstatus": "#ell", "ellstatus": "#ell", "sped": "#sped", "disability": "#sped", "disabilities": "#sped",
    "iep": "#sped", "exceptionality": "#sped", "lunch": "#frl", "frl": "#frl", "frpl": "#frl", "meal": "#frl",
    "meals": "#frl", "term": "#terms", "terms": "#terms", "semester": "#terms", "season": "#terms",
    "quarter": "#terms", "trimester": "#terms", "session": "#terms", "day": "#calendar", "weekday": "#calendar",
    "dow": "#calendar", "month": "#calendar", "attendance": "#attendance", "att": "#attendance",
    "attend": "#attendance", "absence": "#attendance", "absences": "#attendance", "abs": "#attendance",
    "tardy": "#attendance", "enrollment": "#enrollment", "enrolment": "#enrollment", "withdrawal": "#enrollment",
    "program": "#programs", "programs": "#programs", "performance": "#assessment", "proficiency": "#assessment",
    "achievement": "#assessment", "quintile": "#assessment", "discipline": "#discipline", "incident": "#discipline",
    "offense": "#discipline", "offence": "#discipline", "infraction": "#discipline", "consequence": "#discipline",
    "transportation": "#transportation", "transport": "#transportation", "dismissal": "#transportation",
    "subject": "#subjects", "subjects": "#subjects", "language": "#languages", "languages": "#languages",
    "lang": "#languages", "region": "#region", "locale": "#region", "urbanicity": "#region",
}


def header_domains(name: object) -> frozenset[str]:
    """The vocabulary sections a column header names (:data:`HEADER_SECTIONS`), or none when the header also names a
    person (a role word, an actor ``<verb>_By``)."""
    from ..hints import PERSON_ACTOR_VERBS, is_role_word, tokens

    toks = tokens(str(name or ""))
    if any(is_role_word(t) or t == "by" or t in PERSON_ACTOR_VERBS for t in toks):
        return frozenset()
    return frozenset(HEADER_SECTIONS[t] for t in toks if t in HEADER_SECTIONS)


def is_placeholder_word(text: object) -> bool:
    """``text`` stands for no value: blank, a default NA token, a missing/unknown word of the vocabulary, a
    boolean word or a suppression-like token. Under a personal header these are the only words kept (D12, D35)."""
    from ..hints import BOOLEAN_PAIRS, DEFAULT_NA_TOKENS
    from .numeric import is_suppression_token

    if text is None:
        return False
    s = descriptor_code(str(text).strip())
    if not s or s in DEFAULT_NA_TOKENS:
        return True
    f = s.casefold()
    if any(f in pair for pair in BOOLEAN_PAIRS) or is_suppression_token(s):
        return True
    key = _standard_key(s)
    ix = _index()
    return key is not None and key not in ix.name_like and "#missing" in ix.sections.get(key, set())


def is_vocabulary(text: object, domains: Iterable[str] | None = None) -> bool:
    """D35: ``text`` is a word of one of the tool's vocabularies (module docstring), so it quotes nobody: blank,
    a default NA token, a boolean word, a level of an ordinal scale, a school year, a suppression-like token or a
    word of the standard education vocabulary.

    A name-like word (:func:`is_name_like`) is vocabulary only when ``domains`` (:func:`context_domains` of the
    column's units) holds one of its sections; without ``domains`` it never is.
    """
    from ..hints import BOOLEAN_PAIRS, DEFAULT_NA_TOKENS, ORDINAL_VOCABS, school_year_start
    from .numeric import is_suppression_token

    if text is None:
        return False
    s = descriptor_code(str(text).strip())
    if not s or s in DEFAULT_NA_TOKENS:
        return True
    ix = _index()
    key = vocab_key(s)
    if key in ix.name_like:
        return _standard_key(s) is not None and bool(ix.sections.get(key, set()) & set(domains or ()))
    f = s.casefold()
    return (any(f in pair for pair in BOOLEAN_PAIRS) or any(f in v.ranks for v in ORDINAL_VOCABS)
            or school_year_start(s) is not None or is_suppression_token(s) or is_standard_word(s))


__all__ = ["HEADER_SECTIONS", "STANDARD_EDUCATION_VOCABULARY", "context_domains", "descriptor_code", "header_domains",
           "is_name_like", "is_placeholder_word", "is_standard_word", "is_vocabulary", "standard_keys", "vocab_key"]

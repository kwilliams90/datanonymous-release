"""D34 vocabulary: given names and surnames of US school populations that Faker's lists miss (AREA G).

The screen of header, sheet, file and folder names (``privacy.headers``) reads a name pair from the name lists of
``detect.names`` (Faker's given-name and surname lists of the person locales of US school populations). Those
lists hold few names of newcomer and refugee families: Somali, Arabic, Persian and Afghan, West and East African,
South Asian and Nepali, Burmese and Karen, Khmer, Lao and Hmong, Filipino, Chinese pinyin, Korean, Japanese,
Ukrainian, Russian, Polish and Bosnian, Haitian and Central American names. A pair of two such names
(``Hodan Warsame``, ``Dmitri Volkov``, ``Warsame, Hodan``) holds no listed word, so it is invisible to the pair
rules. These curated lists add them, for the D34 screen only (value detection keeps ``detect.names``).

Entries are folded (lower-case ASCII letters, hyphens and spaces removed: ``Seo-yeon`` is ``seoyeon``). Words that
are also everyday English or header words (``Win``, ``Say``, ``Paw``, ``Nada``, ``Lama``, ``Shin``, ``Muse``,
``Jun``) are left out: a header word is never name evidence. Each list is one literal collection of more than
50 entries, so the leak scanner treats it as data (``privacy.leaks.literal_texts``), never as text the tool
writes. The lists are curated, not exhaustive: a name pair of two unlisted words is the accepted residual risk.

Also here: :data:`ASIAN_SURNAMES` (two-letter surnames such as ``Le`` and ``Li``, and Korean and Vietnamese
surnames written first), and the names the screen reads in scripts it does not transliterate: :data:`ARABIC_NAMES`,
:data:`HAN_SURNAMES` and :data:`HAN_GIVEN` (Chinese), :data:`HANGUL_SURNAMES` and :data:`HANGUL_GIVEN` (Korean).
Cyrillic and Greek names are transliterated (``privacy.headers.clean_text``) and read with the Latin lists.
"""

from __future__ import annotations

#: Given names (romanized), by community.
GIVEN_NAMES: frozenset[str] = frozenset({
    # Somali
    "hodan", "abdi", "abdirahman", "abdullahi", "abdiaziz", "abdikadir", "abdiwali", "ayan", "hibaq", "fartun",
    "hamdi", "ifrah", "iman", "khadra", "muna", "nasra", "najma", "sahra", "sagal", "sumaya", "ubah", "ubax",
    "zamzam", "faisa", "filsan", "hawa", "ikran", "istahil", "kowsar", "libaan", "mahad", "bashir", "farhan",
    "guled", "mohamud", "abshir", "deqa", "fadumo", "halima", "ilhan", "ladan", "maryan", "qamar", "shukri",
    "suad", "hamse", "anisa", "nimco", "hodo", "idil",
    # Arabic
    "abdulrahman", "abdelrahman", "abdullah", "alaa", "amal", "anas", "ayah", "bilal", "dalia", "duaa", "eman",
    "fadi", "ghada", "hamza", "haneen", "heba", "hiba", "imad", "kareem", "khaled", "khalid", "maha", "mahmoud",
    "majd", "malak", "marwa", "mazen", "mohammad", "mustafa", "rami", "rania", "rawan", "reem", "rima", "saad",
    "saif", "tariq", "wael", "walid", "yahya", "yousef", "zainab", "zaid", "ziad", "aaisha", "safiya",
    "sumayyah", "habiba", "hafsa",
    # West African
    "oluwaseun", "oluwatobi", "oluwadamilare", "olumide", "oluwafemi", "adebayo", "adeola", "adewale",
    "ayodele", "babajide", "chidi", "chidinma", "chiamaka", "chinedu", "chinonso", "chukwuemeka", "chioma",
    "ebuka", "emeka", "ifeanyi", "ifeoma", "ikechukwu", "kelechi", "ngozi", "nnamdi", "obinna", "uchenna",
    "adaeze", "funmilayo", "folake", "tunde", "temitope", "titilayo", "yetunde", "kwame", "kofi", "kwabena",
    "kwaku", "kojo", "akosua", "abena", "efua", "adwoa", "afia", "fatou", "mamadou", "moussa", "ousmane",
    "aminata", "mariama", "ibrahima", "abdoulaye", "seydou", "aissatou", "binta", "kadiatou", "oumar",
    # East African
    "abebe", "almaz", "bereket", "dawit", "eyerusalem", "fikru", "habtamu", "hiwot", "kidist", "meron",
    "mesfin", "selam", "selamawit", "tesfaye", "tigist", "yonas", "yohannes", "mekdes", "biruk", "filmon",
    "senait", "semhar", "yemane", "haben", "baraka", "amani", "neema", "zawadi", "furaha", "imani", "jabari",
    "juma", "rehema", "bahati",
    # Afghan and Persian
    "ahmadullah", "habibullah", "hamida", "farida", "fereshta", "gulalai", "hasibullah", "khalida", "mahbooba",
    "mansoor", "massoud", "mursal", "nasrin", "navid", "parwana", "parisa", "roya", "sadia", "shabnam",
    "shekiba", "sohrab", "wahid", "zahra", "zarghona", "ziaullah", "arash", "babak", "dariush", "farhad",
    "mehrdad", "nima", "omid", "pouya", "reza", "shirin", "yasaman",
    # South Asian and Nepali
    "aarav", "ananya", "arjun", "ayaan", "ishaan", "kavya", "saanvi", "vihaan", "vivaan", "aanya", "advik",
    "harpreet", "gurpreet", "manpreet", "jaspreet", "simran", "navdeep", "rajesh", "suresh", "ramesh",
    "mahesh", "deepak", "sanjay", "vijay", "anjali", "kiran", "sunita", "raj", "bishal", "sujan", "prakash",
    "sushma", "bikash", "anish", "aayush", "rajan", "nabin", "sanjeev", "arun", "karthik", "vignesh",
    "swathi", "tanvir", "farzana", "nusrat", "tahmina", "shakil", "rakib", "sabbir", "priyanka",
    # Burmese, Karen, Khmer, Lao, Hmong, Thai
    "aung", "kyaw", "myint", "thant", "zaw", "htet", "khin", "nyein", "phyo", "thiri", "thandar", "thazin",
    "yadanar", "naing", "thein", "maung", "htoo", "hser", "sokha", "sopheap", "vannak", "chanthy", "sreymom",
    "phalla", "sovann", "pheakdey", "bounmy", "khamla", "souksavanh", "vilay", "somchai", "somsak", "malee",
    "choua", "zoua", "neng",
    # Filipino
    "jhon", "jaypee", "jomar", "arnel", "rodel", "mylene", "jovelyn", "marivic",
    # Chinese (pinyin), Korean, Japanese
    "zihan", "yuxuan", "haoran", "zixuan", "yichen", "xinyi", "yutong", "ruoxi", "jiayi", "mingyu", "junjie",
    "haoyu", "zhiyuan", "jiahui", "siyu", "yuting", "wenjie", "xiaoming", "xiaoling", "seoyeon", "minjun",
    "jiwoo", "haeun", "doyun", "jiho", "yejin", "jimin", "hyunwoo", "seunghoon", "jaewon", "eunji", "subin",
    "jihoon", "junseo", "minseo", "hyejin", "sungmin", "takeshi", "haruto", "yuto", "sota", "riku", "hinata",
    "sakura", "yuna", "kaito", "daiki", "kenta", "yusuke", "takumi", "ayaka", "misaki", "akiko", "hiroko",
    "tomoko", "keiko", "satoshi", "takashi", "masato", "kazuki", "ryota", "tatsuya", "koki",
    # Ukrainian, Russian, Polish, Bosnian
    "dmitri", "dmitry", "dmytro", "oleksandr", "oleksiy", "olena", "oksana", "yulia", "yuliya", "iryna",
    "tetiana", "volodymyr", "serhiy", "mykola", "mykhailo", "andriy", "bohdan", "taras", "yaroslav", "vasyl",
    "kateryna", "yevgeniya", "yevgeny", "artem", "artyom", "maksim", "kirill", "denys", "vitaliy", "ihor",
    "anatoliy", "valentyna", "halyna", "lyudmila", "darya", "sofiya", "kseniya", "krzysztof", "wojciech",
    "grzegorz", "agnieszka", "malgorzata", "katarzyna", "pawel", "piotr", "tomasz", "lukasz", "kacper",
    "jakub", "zofia", "zuzanna", "wiktoria", "amra", "amela", "edin", "haris", "lejla", "dzenana", "senad",
    "elvir", "nermin", "sanela", "almir",
    # Haitian and Central American
    "wilner", "guerline", "dieuseul", "dieunel", "fritznel", "nadege", "mirlande", "rosemene", "widline",
    "wislande", "woodline", "sherlie", "djimy", "kervens", "yamileth", "yatziri", "xochitl", "maynor",
    "selvin", "eduin",
    # US given names of recent birth years (SSA lists since 2005) that Faker's older lists miss
    "zyaire", "jaxtyn", "kairo", "lennox", "zayn", "kylo", "dariel", "everly", "emersyn", "kehlani", "remi",
    "daleyza", "zuri", "zariyah", "briella", "emberly", "everleigh", "haisley", "journee", "oaklynn", "saylor",
    "wrenley", "jaxxon", "kaison", "khalani", "aziel", "zamir", "azael", "kenzo", "ameer", "jayceon", "kashton",
    "kayce", "jakari", "zakai", "aziyah", "khaleesi", "paislee", "tinsley", "londynn", "layan",
    # Korean and Vietnamese given names (romanized) the lists above miss
    "minho", "minjae", "jisoo", "jiyeon", "hyunjin", "seojun", "yejun", "dohyun", "siwoo", "hajun", "jiwon",
    "sumin", "chaewon", "yerin", "soyeon", "eunwoo", "taeyang", "hieu", "khanh", "uyen", "thinh", "vinh",
    "trinh", "nhi",
})

#: Romanized East and Southeast Asian names the lists treat as no name: two-letter surnames (``Le Minh``, ``Wei
#: Li``, ``Vo Thi Mai``: ``short``, a name only beside a known given name; ``word``: also an everyday English word,
#: ``An``, ``Do``, ``Go``, ``He``, ``Oh``, never a name word) and common Korean and Vietnamese surnames written
#: before the given name (``Lee Minho``, ``Nguyen Van Minh``: ``first``, read by the name lists). One dict of more
#: than 50 entries (data).
ASIAN_SURNAMES: dict[str, str] = {
    "le": "short", "ly": "short", "vo": "short", "vu": "short", "do": "word", "ho": "word", "li": "short",
    "wu": "short", "xu": "short", "ma": "word", "hu": "short", "yu": "short", "lu": "short", "he": "word",
    "ng": "short", "oh": "word", "ko": "short", "go": "word", "ha": "word", "yi": "short", "an": "word",
    "im": "word", "ra": "word", "bo": "word", "la": "word", "mo": "word",
    "kim": "first", "lee": "first", "park": "first", "choi": "first", "jung": "first", "chung": "first",
    "kang": "first", "cho": "first", "yoon": "first", "jang": "first", "lim": "first", "han": "first",
    "shin": "first", "seo": "first", "kwon": "first", "hwang": "first", "ahn": "first", "song": "first",
    "ryu": "first", "jeon": "first", "hong": "first", "moon": "first", "yang": "first", "son": "first",
    "bae": "first", "baek": "first", "nam": "first", "noh": "first", "kwak": "first", "nguyen": "first",
    "tran": "first", "pham": "first", "hoang": "first", "huynh": "first", "phan": "first", "dang": "first",
    "bui": "first", "dinh": "first", "duong": "first", "truong": "first", "luong": "first", "trinh": "first",
}

#: Common name words in Arabic script (given names, ``Abd`` names, family names), normalized as
#: :func:`privacy.headers.clean_text` reads them (no vowel marks; alef, taa marbuta and alef maqsura forms folded):
#: two of them in one name are a person's name (``محمد علي``). One literal collection of more than 50 entries.
ARABIC_NAMES: frozenset[str] = frozenset({
    "محمد", "احمد", "محمود", "مصطفي", "علي", "حسن", "حسين", "عمر", "عثمان", "خالد", "يوسف", "ابراهيم",
    "اسماعيل", "عبدالله", "عبدالرحمن", "عبدالعزيز", "عبدالكريم", "عبدالقادر", "عبد", "الله", "الرحمن",
    "فاطمه", "عائشه", "عايشه", "مريم", "زينب", "خديجه", "نور", "ساره", "ليلي", "هدي", "سلمي", "رنا", "ياسمين",
    "امنه", "حمزه", "بلال", "طارق", "سعيد", "سعد", "كريم", "جمال", "ناصر", "صالح", "منصور", "قاسم", "جابر",
    "عوض", "حداد", "خوري", "درويش", "يحيي", "زيد", "زياد", "رامي", "وليد", "عادل", "سامي", "هاني", "ماهر",
    "نبيل", "فيصل", "سلطان", "راشد", "حمد", "عيسي", "موسي", "هارون", "داود", "سليمان", "ايوب", "يونس",
    "ادم", "اياد", "انس", "معاذ", "رحمه", "دعاء", "ايمان", "مروه", "رهف", "جني", "تالا", "حسام", "هشام",
    "عصام", "مجدي", "رضا", "مهدي", "عباس", "جعفر", "صادق", "باقر", "زهراء", "رقيه", "سكينه", "حليمه",
    "شريف", "منير", "نادر", "ناديه", "سميره", "امل", "هند", "لينا", "دينا", "منى", "هبه", "اسماء", "سمير",
})

#: Common Chinese surnames (simplified and traditional characters). One literal collection of more than 50
#: entries (data).
HAN_SURNAMES: frozenset[str] = frozenset({
    "王", "李", "张", "刘", "陈", "杨", "黄", "赵", "吴", "周", "徐", "孙", "马", "朱", "胡", "郭", "何", "林",
    "罗", "高", "郑", "梁", "谢", "宋", "唐", "许", "韩", "冯", "邓", "曹", "彭", "曾", "肖", "田", "董", "袁",
    "潘", "于", "蒋", "蔡", "余", "杜", "叶", "程", "苏", "魏", "吕", "丁", "任", "沈", "姚", "卢", "姜", "崔",
    "钟", "谭", "陆", "汪", "范", "金", "石", "廖", "贾", "夏", "韦", "付", "方", "白", "邹", "孟", "熊", "秦",
    "邱", "江", "尹", "薛", "闫", "段", "雷", "侯", "龙", "史", "陶", "黎", "贺", "顾", "毛", "郝", "龚", "邵",
    "万", "钱", "严", "覃", "武", "戴", "莫", "孔", "向", "汤", "張", "劉", "陳", "楊", "黃", "趙", "吳", "許",
    "韓", "馮", "鄧", "蕭", "葉", "蘇", "呂", "盧", "鍾", "譚", "陸", "範", "賈", "鄒", "龍", "賀", "顧", "龔",
    "錢", "嚴", "湯", "羅", "鄭", "謝", "蔣", "韋", "萬",
})

#: Characters of Chinese given names that ordinary words rarely pair with a surname character (``伟``, ``芳``,
#: ``婷``; not ``一``, ``国``, ``文``, ``明``: ``周一``, ``王国``). One literal collection of more than 50 entries.
HAN_GIVEN: frozenset[str] = frozenset({
    "伟", "芳", "娜", "敏", "静", "丽", "磊", "军", "洋", "勇", "艳", "杰", "娟", "涛", "秀", "霞", "刚", "桂",
    "玲", "鹏", "辉", "兰", "燕", "婷", "浩", "宇", "欣", "怡", "涵", "轩", "萱", "睿", "彬", "斌", "鑫", "琳",
    "颖", "慧", "婧", "倩", "晶", "佳", "嘉", "梓", "诺", "俊", "峰", "健", "志", "宁", "凯", "坤", "毅", "翔",
    "宏", "昊", "泽", "铭", "瑞", "皓", "晗", "妍", "琪", "瑶", "璐", "蕾", "薇", "菲", "莉", "琴", "梅", "凤",
    "萍", "珍", "娇", "婉", "悦", "彤", "晴", "宸", "煜", "逸", "辰", "楠", "偉", "麗", "軍", "艷", "傑", "濤",
    "鵬", "輝", "蘭", "飛", "軒", "穎", "曉", "誌", "寧", "凱", "銘", "瑤", "雲", "鳳", "悅", "飞", "雯", "霖",
})

#: Common Korean surnames (Hangul syllables). One literal collection of more than 50 entries (data).
HANGUL_SURNAMES: frozenset[str] = frozenset({
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임", "한", "오", "서", "신", "권", "황", "안", "송",
    "류", "유", "전", "홍", "고", "문", "양", "손", "배", "백", "허", "남", "심", "노", "하", "곽", "성", "차",
    "주", "우", "구", "민", "진", "나", "지", "엄", "원", "채", "천", "방", "공", "현", "함", "변", "염", "여",
    "추", "도", "소", "석", "선", "설", "마", "길", "연", "위", "표", "명", "기", "반", "라", "왕", "금", "옥",
})

#: Common Korean given names (two Hangul syllables). One literal collection of more than 50 entries (data).
HANGUL_GIVEN: frozenset[str] = frozenset({
    "민준", "서준", "도윤", "예준", "시우", "하준", "주원", "지호", "지후", "준우", "준서", "건우", "도현", "현우",
    "지훈", "우진", "선우", "서진", "민재", "현준", "연우", "유준", "정우", "승우", "승현", "시윤", "준혁", "은우",
    "지환", "승민", "유찬", "윤우", "민성", "준영", "시후", "진우", "수현", "재윤", "시현", "동현", "수호", "태윤",
    "민규", "재원", "한결", "지성", "윤호", "서연", "서윤", "지우", "서현", "민서", "하은", "하윤", "윤서", "지유",
    "지민", "채원", "수아", "지아", "지윤", "은서", "다은", "예은", "수빈", "소율", "예린", "지원", "소윤", "지안",
    "하린", "시은", "유나", "채은", "가은", "윤아", "민지", "예서", "수민", "지현", "서영", "예원", "다인", "은지",
    "수연", "지은", "혜원", "영희", "미영", "정희", "영숙", "경희", "철수", "영수", "영호", "정훈", "민호", "성민",
})

#: Surnames (romanized), by community.
SURNAMES: frozenset[str] = frozenset({
    # Somali and Arabic
    "warsame", "mohamud", "abdullahi", "elmi", "dahir", "guled", "hersi", "samatar",
    "abdirahman", "haddad", "khoury", "nasser", "saleh", "hamdan", "mansour", "qasim", "jaber", "awad",
    "darwish",
    # West African
    "adeyemi", "adebayo", "okafor", "okonkwo", "nwosu", "okeke", "ogunleye", "olawale", "oyelaran",
    "babatunde", "afolabi", "akinola", "balogun", "bankole", "nwachukwu", "mensah", "owusu", "boateng",
    "asante", "appiah", "agyeman", "danso", "diallo", "camara", "cisse", "coulibaly", "diop", "fofana",
    "keita", "konate", "kone", "ndiaye", "sylla", "toure", "traore",
    # East African
    "tesfaye", "haile", "bekele", "girma", "tadesse", "alemu", "mekonnen", "gebre", "gebremedhin", "kebede",
    "wolde", "desta", "assefa", "abebe", "mengistu", "berhane", "tekle", "mwangi", "kamau", "otieno",
    "odhiambo", "wanjiru", "njoroge", "kimani", "mutua",
    # Afghan and Persian
    "ahmadzai", "popal", "stanikzai", "noori", "rahimi", "sultani", "karimi", "hashimi", "hakimi", "safi",
    "wardak", "durrani", "barakzai", "rezaei", "mohammadi", "hosseini", "ahmadi", "jafari", "sadeghi", "moradi",
    # South Asian and Nepali
    "akter", "uddin", "hossain", "thapa", "gurung", "tamang", "limbu", "shrestha", "adhikari", "bhattarai",
    "khadka", "poudel", "sapkota", "dahal", "ghimire", "karki",
    # Burmese, Hmong
    "aung", "htun", "kyaw", "myint", "naing", "thein", "hlaing", "lwin", "maung", "thao", "moua", "khang",
    "kue",
    # Ukrainian, Russian, Polish, Bosnian
    "volkov", "ivanov", "petrov", "smirnov", "kuznetsov", "popov", "lebedev", "kozlov", "novikov", "morozov",
    "pavlov", "semenov", "golubev", "vinogradov", "bogdanov", "vorobyov", "fedorov", "mikhailov", "belyaev",
    "tarasov", "belov", "komarov", "orlov", "kiselev", "makarov", "andreev", "kovalenko", "shevchenko",
    "bondarenko", "tkachenko", "kravchenko", "melnyk", "boyko", "kovalchuk", "oliynyk", "marchenko", "moroz",
    "lysenko", "rudenko", "savchenko", "petrenko", "kowalski", "nowak", "wisniewski", "wojcik", "kaminski",
    "lewandowski", "zielinski", "szymanski", "wozniak", "dabrowski", "hodzic", "begic", "hadzic", "mujic",
    "delic", "kovacevic", "hasanovic", "huskic", "salihovic",
    # Haitian
    "baptiste", "celestin", "desir", "dorvil", "exantus", "laguerre", "metellus", "valcourt",
    # Korean
    "ahn", "baek", "heo", "hwang", "jang", "jeon", "jung", "kwon", "yoo", "yoon",
    # Arabic given names used as family names
    "abdullah", "muhammad", "khalil", "mustafa",
    # Navajo and other Native American, Cajun, Polish, Bengali and Ghanaian surnames the lists miss
    "begay", "yazzie", "tsosie", "benally", "etsitty", "bitsui", "yellowhair", "tsinajinnie", "nakai",
    "manygoats", "harjo", "thibodeaux", "boudreaux", "broussard", "guidry", "arceneaux", "fontenot", "comeaux",
    "theriot", "breaux", "robichaux", "trahan", "kowalczyk", "kaczmarek", "krawczyk", "piotrowski", "grabowski",
    "pawlowski", "michalski", "nowakowski", "adamczyk", "dudek", "zajac", "wieczorek", "jablonski", "olszewski",
    "jaworski", "malinowski", "bhattacharya", "quarshie", "quaye", "tetteh", "amoah", "addo", "adjei",
})

__all__ = ["GIVEN_NAMES", "SURNAMES", "ASIAN_SURNAMES", "ARABIC_NAMES", "HAN_SURNAMES", "HAN_GIVEN",
           "HANGUL_SURNAMES", "HANGUL_GIVEN"]

"""D34 vocabulary: public people, characters, programs and brands whose names label K-12 data (AREA G).

A folder, file, sheet or column named after a school, charter network, program, award, curriculum unit or
vendor (``Rosa_Parks/``, ``Sam_Houston/``, ``Jose_Marti_MAST_Grades.csv``, ``Cal_Grant_Eligibility.csv``, a
gradebook column ``Atticus Finch Essay``, ``Harry Potter Ch 1-5``, ``Charles's Law Lab``, ``Lewis Dot
Structures``, a budget sheet ``Carl D. Perkins CTE``, a vendor ``Sherwin Williams``) holds a public name, not a
student's or teacher's. The screen of names (``privacy.headers``) takes none of these words for a person's name:

* :data:`NAMESAKES`: notable people and fictional characters US schools, programs, awards and curriculum units
  are named after (presidents and founders, civil-rights and community leaders, authors and their characters,
  scientists, explorers, artists, athletes, federal programs). A run of name tokens that spells one, in order
  (extra initials, particles and suffixes in between are fine; an initial the entry holds must be there:
  ``Robert E. Lee`` but not ``Robert Lee``) is no person, unless another name word sits right beside it
  (``Jose Marti Garcia`` is a person);
* :data:`PROGRAM_WORDS`: single words of programs, networks, assessments and products (``KIPP``, ``AVID``);
* :data:`PROGRAM_PHRASES`: multi-word programs, products, eponyms of science and history units and vendors
  (``Green Dot``, ``Lewis Dot``, ``Lincoln Douglas``, ``John Deere``): never name words wherever they occur;
* :data:`BRANDS`: vendor and brand names; a brand that is also a very common person's name (``Sara Lee``,
  ``Edward Jones``) counts only beside a business word or a number (``Sara Lee Bread``, ``Edward Jones 403b``);
* :data:`EPONYM_WORDS`: a name right before one of these words names a law, model, event or era
  (``Charles's Law``, ``Hardy-Weinberg Principle``, ``John Brown's Raid``, ``Ming Dynasty``).

The lists are curated, not exhaustive: a public name missing here is kept with ``--keep-name TEXT``. Since D34
was frozen as a best-effort layer (user, 2026-10-06), the authors, characters and historical figures of K-12
gradebooks are listed even when their names are also ordinary names (``Harper Lee``, ``Andrew Johnson``, ``John
Henry``, ``Charlie Gordon``): a student with exactly such a name passes, the accepted residual risk. A few of the
most common names of ordinary people stay out (``Michael Anderson``, ``John Lewis``, ``Ben Davis``; ``Robert Lee``
without the ``E.``): a folder of a student with that name still FAILs. Each list
is one literal collection of more than 50 entries, so the leak scanner treats it as data
(``privacy.leaks.literal_texts``), never as text the tool writes. Entries are folded: lower-case ASCII words
without punctuation (a possessive ``'s`` dropped, hyphens as spaces).
"""

from __future__ import annotations

from functools import lru_cache

#: Notable namesakes of US schools, programs, awards, holidays and curriculum units.
NAMESAKES: frozenset[str] = frozenset({
    # presidents, first ladies, founders, statesmen
    "george washington", "john adams", "thomas jefferson", "james madison", "james monroe", "john quincy adams",
    "andrew jackson", "martin van buren", "william henry harrison", "james polk", "james k polk", "zachary taylor",
    "millard fillmore", "franklin pierce", "james buchanan", "abraham lincoln", "ulysses grant", "ulysses s grant",
    "rutherford hayes", "rutherford b hayes", "james garfield", "chester arthur", "grover cleveland",
    "benjamin harrison", "william mckinley", "theodore roosevelt", "teddy roosevelt", "william howard taft",
    "woodrow wilson", "warren harding", "calvin coolidge", "herbert hoover", "franklin roosevelt",
    "franklin d roosevelt", "franklin delano roosevelt", "eleanor roosevelt", "harry truman", "harry s truman",
    "dwight eisenhower", "dwight d eisenhower", "john kennedy", "john f kennedy", "john fitzgerald kennedy",
    "robert f kennedy", "lyndon johnson", "lyndon b johnson", "lyndon baines johnson", "lady bird johnson",
    "jimmy carter", "rosalynn carter", "ronald reagan", "nancy reagan", "george h w bush", "george w bush",
    "barbara bush", "laura bush", "barack obama", "michelle obama", "richard nixon", "gerald ford",
    "benjamin franklin", "alexander hamilton", "patrick henry", "paul revere", "samuel adams", "john hancock",
    "nathan hale", "betsy ross", "francis scott key", "thomas paine", "daniel webster", "meriwether lewis",
    "daniel boone", "davy crockett", "kit carson", "john muir", "johnny appleseed", "noah webster", "horace mann",
    "john dewey", "william penn", "john marshall", "john jay", "henry clay", "john c calhoun", "william seward",
    "charles sumner", "robert e lee", "stonewall jackson", "thomas stonewall jackson", "jefferson davis",
    "william tecumseh sherman", "philip sheridan", "george custer", "george armstrong custer", "douglas macarthur",
    "george patton", "john j pershing", "chester w nimitz", "dred scott", "john wilkes booth",
    # civil rights, labour and community leaders
    "martin luther king", "coretta scott king", "rosa parks", "frederick douglass", "harriet tubman",
    "sojourner truth", "booker washington", "booker t washington", "george washington carver", "w e b du bois",
    "web du bois", "malcolm x", "medgar evers", "myrlie evers", "thurgood marshall", "jackie robinson",
    "jesse owens", "mary mcleod bethune", "mary bethune", "ida b wells", "fannie lou hamer", "ralph abernathy",
    "whitney young", "bayard rustin", "a philip randolph", "dorothy height", "shirley chisholm", "barbara jordan",
    "mickey leland", "colin powell", "condoleezza rice", "carter woodson", "carter g woodson", "crispus attucks",
    "nat turner", "ralph bunche", "charles drew", "benjamin banneker", "madam c j walker", "madam cj walker",
    "ruby bridges", "paul robeson", "harriet beecher stowe", "susan b anthony", "elizabeth cady stanton",
    "lucretia mott", "clara barton", "florence nightingale", "jane addams", "dorothea dix", "helen keller",
    "marcus garvey", "harvey milk", "cesar chavez", "cesar e chavez", "dolores huerta", "larry itliong",
    "fred korematsu", "sylvia mendez", "gonzalo mendez", "felicitas mendez", "sonia sotomayor",
    "sandra day oconnor", "ruth bader ginsburg", "malala yousafzai", "oprah winfrey", "nelson mandela",
    "mahatma gandhi", "mother teresa", "albert schweitzer", "anne frank", "elie wiesel", "martin luther",
    # writers and poets
    "langston hughes", "maya angelou", "paul laurence dunbar", "phillis wheatley", "zora neale hurston",
    "james weldon johnson", "countee cullen", "gwendolyn brooks", "toni morrison", "walt whitman", "mark twain",
    "edgar allan poe", "robert frost", "emily dickinson", "henry wadsworth longfellow", "ralph waldo emerson",
    "henry david thoreau", "louisa may alcott", "laura ingalls wilder", "john steinbeck", "pearl s buck",
    "theodor geisel", "beverly cleary", "jack london", "ernest hemingway", "f scott fitzgerald",
    "william shakespeare", "geoffrey chaucer", "charles dickens", "jane austen", "charlotte bronte", "emily bronte",
    "bram stoker", "oscar wilde", "nathaniel hawthorne", "herman melville", "washington irving",
    "james fenimore cooper", "sylvia plath", "carl sandburg", "e e cummings", "william carlos williams",
    "robert louis stevenson", "rudyard kipling", "jules verne", "victor hugo", "leo tolstoy", "fyodor dostoevsky",
    "gabriel garcia marquez", "isabel allende", "julia alvarez", "khaled hosseini", "chinua achebe",
    "ralph ellison", "octavia butler", "kurt vonnegut", "ray bradbury", "george orwell", "aldous huxley",
    "william golding", "j d salinger", "jd salinger", "lorraine hansberry", "tennessee williams", "markus zusak",
    "orson scott card", "ursula le guin", "lois lowry", "roald dahl", "judy blume", "kate dicamillo",
    "jerry spinelli", "walter dean myers", "kwame alexander", "jacqueline woodson", "christopher paul curtis",
    "pam munoz ryan", "eric carle", "mo willems", "kevin henkes", "patricia polacco", "cynthia rylant",
    "arnold lobel", "dav pilkey", "jeff kinney", "rick riordan", "j k rowling", "jk rowling", "lemony snicket",
    "shel silverstein", "maurice sendak", "ezra jack keats", "tomie depaola", "chris van allsburg", "e b white",
    "eb white", "l frank baum", "lewis carroll", "c s lewis", "cs lewis", "j r r tolkien", "jrr tolkien",
    "madeleine lengle", "gary paulsen", "louis sachar", "r j palacio", "rj palacio", "ann m martin",
    "mary pope osborne", "peggy parish", "margaret wise brown", "laura numeroff", "jon scieszka", "jon klassen",
    "mercer mayer", "norman bridwell", "william steig", "leo lionni", "robert munsch", "jane yolen", "eve bunting",
    "faith ringgold", "jerry pinkney", "matt de la pena", "linda sue park", "jewell parker rhodes", "s e hinton",
    "jan brett", "andrew clements", "jack prelutsky", "doreen cronin", "gloria anzaldua",
    "se hinton", "katherine applegate", "wilson rawls", "esther forbes", "elizabeth george speare", "scott odell",
    "jean craighead george", "natalie babbitt", "e l konigsburg",
    # fictional characters of K-12 reading lists
    "atticus finch", "scout finch", "jem finch", "boo radley", "jay gatsby", "holden caulfield", "huck finn",
    "huckleberry finn", "tom sawyer", "oliver twist", "jane eyre", "ebenezer scrooge", "tiny tim",
    "stanley yelnats", "stuart little", "flat stanley", "junie b jones", "amelia bedelia", "cam jansen",
    "ramona quimby", "beezus quimby", "mr popper", "mrs frisby", "ms frizzle", "mr putter", "mrs piggle wiggle",
    "greg heffley", "percy jackson", "harry potter", "hermione granger", "ron weasley", "albus dumbledore",
    "charlie bucket", "willy wonka", "matilda wormwood", "james henry trotter", "lennie small", "romeo montague",
    "juliet capulet", "julius caesar", "augustus caesar", "mark antony", "marc antony", "king lear", "lady macbeth",
    "jean valjean", "sherlock holmes", "captain ahab", "hester prynne", "victor frankenstein", "dorian gray",
    "katniss everdeen", "peeta mellark", "frodo baggins", "bilbo baggins", "peter pan", "encyclopedia brown",
    "curious george", "paddington bear", "winnie the pooh", "stanley lambchop", "pippi longstocking",
    "auggie pullman", "august pullman", "johnny tremain", "ponyboy curtis", "miss honey", "mrs twit",
    "jack and annie", "rosie the riveter", "uncle sam", "dr jekyll", "mr hyde", "miss nelson is missing",
    "miss nelson is back", "miss nelson has a field day", "mrs wishy washy", "mr tiger goes wild",
    "mr poppers penguins", "mrs dalloway", "mr lincoln way", "miss rumphius", "ms rumphius", "luna lovegood", "draco malfoy", "neville longbottom",
    "severus snape", "rubeus hagrid", "judy moody", "esperanza ortega", "brian robeson",
    # scientists, mathematicians, inventors, explorers of science
    "isaac newton", "albert einstein", "galileo galilei", "nicolaus copernicus", "johannes kepler",
    "charles darwin", "gregor mendel", "marie curie", "madame curie", "pierre curie", "louis pasteur", "niels bohr",
    "john dalton", "ernest rutherford", "dmitri mendeleev", "antoine lavoisier", "robert boyle", "jacques charles",
    "amedeo avogadro", "michael faraday", "james clerk maxwell", "alessandro volta", "georg ohm", "robert hooke",
    "blaise pascal", "rene descartes", "leonhard euler", "carl friedrich gauss", "ada lovelace", "alan turing",
    "grace hopper", "katherine johnson", "dorothy vaughan", "rosalind franklin", "francis crick",
    "barbara mcclintock", "jane goodall", "rachel carson", "stephen hawking", "carl sagan", "neil degrasse tyson",
    "neil armstrong", "buzz aldrin", "john glenn", "gus grissom", "ellison onizuka", "ronald mcnair",
    "judith resnik", "kalpana chawla", "christa mcauliffe", "sally ride", "mae jemison", "ellen ochoa",
    "nikola tesla", "thomas edison", "alexander graham bell", "orville wright", "wilbur wright", "jonas salk",
    "edwin hubble", "max planck", "werner heisenberg", "erwin schrodinger", "enrico fermi", "robert oppenheimer",
    "j robert oppenheimer", "richard feynman", "linus pauling", "dian fossey", "edward jenner", "alexander fleming",
    "elizabeth blackwell", "leonardo da vinci", "henry ford", "steve jobs", "bill gates", "mark zuckerberg",
    "elon musk", "tim berners lee", "hedy lamarr", "srinivasa ramanujan", "emmy noether", "alfred wegener",
    "charles richter", "christian doppler", "robert bunsen", "reginald punnett", "john venn", "hans krebs",
    "gilbert n lewis",
    # explorers and world history
    "christopher columbus", "amerigo vespucci", "ferdinand magellan", "hernando de soto", "juan ponce de leon",
    "hernan cortes", "hernando cortes", "francisco pizarro", "francisco coronado", "francisco vasquez de coronado",
    "juan rodriguez cabrillo", "vasco nunez de balboa", "vasco da gama", "marco polo", "henry hudson",
    "jacques cartier", "samuel de champlain", "john cabot", "francis drake", "walter raleigh", "william bradford",
    "john winthrop", "anne hutchinson", "andrew carnegie", "john d rockefeller", "cornelius vanderbilt",
    "upton sinclair", "charles lindbergh", "amelia earhart", "winston churchill", "adolf hitler", "joseph stalin",
    "benito mussolini", "mikhail gorbachev", "napoleon bonaparte", "marie antoinette", "joan of arc",
    "genghis khan", "kublai khan", "alexander the great", "peter the great", "catherine the great",
    "johannes gutenberg", "john calvin", "queen elizabeth", "queen victoria", "king tut", "christopher wren",
    "sun yat sen", "ho chi minh", "chiang kai shek", "kim il sung", "kim jong il", "kim jong un", "deng xiaoping",
    "zhou enlai", "mao zedong", "mao tse tung", "sun tzu", "lao tzu", "pol pot", "xi jinping", "shinzo abe",
    "henry iv", "henry vi", "henry viii", "richard ii", "richard iii", "king john", "king henry", "elizabeth ii",
    "louis xiv", "louis xvi", "george iii",
    # artists, musicians, athletes
    "vincent van gogh", "pablo picasso", "georgia okeeffe", "andy warhol", "claude monet", "jacob lawrence",
    "romare bearden", "keith haring", "henri matisse", "salvador dali", "johannes vermeer",
    "wolfgang amadeus mozart", "ludwig van beethoven", "johann sebastian bach", "duke ellington", "louis armstrong",
    "ella fitzgerald", "scott joplin", "george gershwin", "aaron copland", "john philip sousa", "billie holiday",
    "nina simone", "elvis presley", "bob marley", "frederic chopin", "antonio vivaldi", "pyotr tchaikovsky",
    "leonard bernstein", "lin manuel miranda", "walt disney", "will rogers", "babe ruth", "lou gehrig",
    "roberto clemente", "jim thorpe", "wilma rudolph", "arthur ashe", "althea gibson", "muhammad ali",
    "billie jean king", "serena williams", "venus williams", "simone biles", "ronald mcdonald", "kobe bryant",
    "jim henson",
    # Native American namesakes
    "sitting bull", "crazy horse", "chief joseph", "black hawk", "red cloud", "quanah parker", "cynthia ann parker",
    "maria tallchief", "wilma mankiller",
    # Texas
    "sam houston", "stephen f austin", "stephen austin", "moses austin", "william b travis", "jim bowie",
    "james bowie", "lorenzo de zavala", "juan seguin", "jose antonio navarro", "mirabeau lamar", "mirabeau b lamar",
    "anson jones", "sul ross", "lawrence sullivan ross", "susanna dickinson", "james bonham", "ben milam",
    "deaf smith", "erastus deaf smith", "ann richards", "hector p garcia", "raul yzaguirre", "emma tenayuca",
    "jovita idar", "henry b gonzalez", "selena quintanilla", "audie murphy", "chester nimitz", "ima hogg",
    "j frank dobie", "katherine anne porter",
    # campuses of large Texas districts named after local people (an initial the entry holds must be there)
    "charles h milby", "jesse h jones", "michael e debakey", "lily b clayton", "justin f kimball",
    "h grady spruce", "irma lerma rangel", "ross s sterling", "w t white", "david w carter", "wilmer hutchins",
    "amon carter", "o d wyatt", "r l paschal", "i m terrell", "wm howard taft",
    # California and the West
    "dalip singh saund", "junipero serra", "father junipero serra", "john sutter", "john c fremont",
    "luther burbank", "earl warren", "leland stanford", "phoebe hearst", "julia morgan", "maya lin",
    "rafer johnson", "ruben salazar", "edward roybal", "george moscone", "hiram johnson", "theodore judah",
    # Latin America and the Caribbean
    "jose marti", "benito juarez", "miguel hidalgo", "miguel hidalgo y costilla", "jose maria morelos",
    "emiliano zapata", "pancho villa", "simon bolivar", "jose de san martin", "bernardo ohiggins",
    "gabriela mistral", "pablo neruda", "octavio paz", "sor juana", "sor juana ines de la cruz", "frida kahlo",
    "diego rivera", "jose vasconcelos", "ignacio zaragoza", "vicente guerrero", "lazaro cardenas",
    "eugenio maria de hostos", "luis munoz marin", "luis munoz rivera", "ramon emeterio betances",
    "julia de burgos", "antonia pantoja", "toussaint louverture", "juan pablo duarte", "celia cruz", "tito puente",
    "oscar romero", "rigoberta menchu",
    # Catholic and other school namesakes
    "bishop lynch", "bishop gorman", "bishop mccort", "bishop kenny", "bishop ireton", "bishop amat",
    "bishop montgomery", "cardinal newman", "cardinal gibbons", "cardinal ritter", "cardinal spellman",
    "cardinal mooney", "cardinal hayes", "father ryan", "father judge", "father flanagan", "elizabeth ann seton",
    "elizabeth seton", "mother seton", "john neumann", "katharine drexel", "frances cabrini", "mother cabrini",
    "thomas aquinas", "thomas more", "ignatius loyola", "francis xavier", "john bosco", "don bosco",
    "francis de sales", "vincent de paul", "john paul ii", "pope john paul", "pope francis", "fulton sheen",
    "thomas merton", "sister clara muhammad", "monsignor pace",
    # education authors and frameworks named after people
    "ruby payne", "jan richardson", "lucy calkins", "irene fountas", "gay su pinnell", "charlotte danielson",
    "robert marzano", "doug lemov", "harry wong", "ron clark", "jo boaler", "carol dweck", "maria montessori",
    "rudolf steiner", "spencer kagan", "carol ann tomlinson", "grant wiggins", "jay mctighe", "benjamin bloom",
    "howard gardner", "lev vygotsky", "jean piaget", "abraham maslow", "john hattie", "dylan wiliam", "sal khan",
    # federal programs, scholarships, awards and grants named after people
    "horatio alger", "jack kent cooke", "robert c byrd", "claiborne pell", "carl d perkins", "carl perkins",
    "richard b russell", "jacob k javits", "jacob javits", "stewart b mckinney", "stewart mckinney",
    "ronald e mcnair", "augustus f hawkins", "augustus hawkins", "robert t stafford", "william d ford",
    "john h chafee", "morris k udall", "barry goldwater", "barry m goldwater", "benjamin a gilman",
    "benjamin gilman", "charles b rangel", "patsy t mink", "patsy mink", "edward m kennedy", "ted kennedy",
    # D34 freeze (2026-10-06): authors, characters and figures of K-12 gradebooks, also with ordinary names
    "sandra cisneros", "gary soto", "jason reynolds", "katherine paterson", "margaret peterson haddix",
    "suzanne collins", "harper lee", "rudolfo anaya", "tomas rivera", "pat mora", "francisco jimenez",
    "richard connell", "arthur conan doyle", "jimmy santiago baca", "angie thomas", "meg medina", "erika l sanchez",
    "elizabeth acevedo", "kelly yang", "david shannon", "don freeman", "gail gibbons", "seymour simon",
    "naomi shihab nye", "sherman alexie", "nikki giovanni", "toni cade bambara", "piri thomas", "lensey namioka",
    "yoshiko uchida", "laurence yep", "sharon draper", "nic stone", "jesmyn ward", "ta nehisi coates",
    "ibram x kendi", "ruta sepetys", "alan gratz", "lisa yee", "christina soontornvat", "kekla magoon",
    "kate messner", "carl hiaasen", "sharon creech", "patricia maclachlan", "barbara park", "megan mcdonald",
    "james marshall", "judith viorst", "steven kellogg", "ishmael beah", "esmeralda santiago", "amy tan",
    "w w jacobs", "shirley jackson", "harrison bergeron", "maniac magee", "charlie gordon", "mr falker",
    "mr terupt", "mr lemoncello", "ms bixby", "edward tulane", "henry huggins", "nancy drew", "artemis fowl",
    "tom robinson", "george milton", "daisy buchanan", "willy loman", "john proctor", "abigail williams",
    "walter lee younger", "winston smith", "johnny cade", "della young", "general zaroff", "louise mallard",
    "salva dut", "josh bell", "jordan banks", "merci suarez", "mia tang", "melody brooks", "starr carter",
    "castle cranshaw", "gilly hopkins", "opal buloni", "despereaux tilling", "holling hoodhood", "jess aarons",
    "leslie burke", "lyddie worthen", "elizabeth bennet", "miss lottie", "mrs luella bates washington jones",
    "annabeth chase", "amber brown", "hank zipzer", "stink moody", "christopher robin", "peter rabbit",
    "arthur read", "mike mulligan", "strega nona", "john henry", "paul bunyan", "pecos bill", "casey jones",
    "sally ann thunder", "slue foot sue", "annie oakley", "calamity jane", "buffalo bill", "wild bill hickok",
    "febold feboldson", "mr peanut", "ms pac man", "mr holland", "mrs doubtfire", "mister rogers", "mr rogers",
    "william barret travis", "james fannin", "juan cortina", "lucy parsons", "hendrick arnold",
    "emily west morgan", "gail borden", "richard coke", "james hogg", "jim hogg", "miriam ma ferguson",
    "ma ferguson", "pa ferguson", "oveta culp hobby", "bessie coleman", "howard hughes", "babe didrikson zaharias",
    "lulu belle madison white", "walter cronkite", "andrew johnson", "john tyler", "chester a arthur",
    "warren g harding", "bill clinton", "donald trump", "joe biden", "marquis de lafayette", "haym salomon",
    "james armistead", "mercy otis warren", "abigail adams", "deborah sampson", "mary ludwig hays",
    "molly pitcher", "benedict arnold", "bernardo de galvez", "stephen douglas", "j p morgan", "cabeza de vaca",
    "king george iii", "o henry", "gus garcia", "bertha sadler means", "kay bailey hutchison",
    "ross shaw sterling", "rosie m collins sorrells", "norman sims", "guerrero thompson", "bryan adams",
    "lebron james",
})

#: Single words that name programs, networks, assessments and products used in K-12 data and are rarely the name
#: of a person (``KIPP_Austin``, ``IDEA_Rundberg``, ``AVID_Roster``, ``NWEA_Jordan``): never a name word in a name.
#: Words that are also common given names or surnames (``Noble``, ``Khan``, ``Amira``) are left out.
PROGRAM_WORDS: frozenset[str] = frozenset({
    "kipp", "idea", "uncommon", "aspire", "alliance", "summit", "rocketship", "uplift", "basis", "mastery",
    "achievement", "democracy", "success", "responsive", "upward", "headstart", "avid", "pell", "jrotc", "rotc",
    "ffa", "fbla", "deca", "hosa", "skillsusa", "fccla", "nhs", "mathcounts", "lexia", "iready", "dibels", "nwea",
    "istation", "dreambox", "zearn", "eureka", "amplify", "wonders", "envision", "edgenuity", "naviance",
    "schoology", "powerschool", "skyward", "aeries", "synergy", "clever", "ixl", "brainpop", "newsela",
    "achieve", "imagine", "pbis", "mtss", "fafsa", "tsia", "staar", "parcc", "sbac", "cogat", "nnat",
    "terranova", "wida", "elpac", "telpas", "lexile", "quantile", "renaissance", "freckle", "seesaw", "nearpod",
    "kahoot", "flocabulary", "readworks", "commonlit", "desmos", "geogebra", "montessori", "waldorf",
    "reggio", "gearup", "trio", "edmentum", "illuminate", "infinite", "aimsweb", "fastbridge", "mclass",
    "acadience", "heggerty", "fundations", "eduphoria", "frontline", "focus", "ascender", "chromebook",
    "chromebooks", "ipad", "ipads",
})

#: Multi-word program, network and product names, eponyms of science and history units and vendor names whose
#: words are on the name lists (``Green Dot``, ``Cal Grant``, ``Lewis Dot``, ``Lincoln Douglas``, ``John Deere``).
PROGRAM_PHRASES: frozenset[str] = frozenset({
    "green dot", "cal grant", "pell grant", "achievement first", "yes prep", "success academy",
    "democracy prep", "great hearts", "big picture", "high tech high", "teach for america", "city year",
    "boys and girls club", "boys girls club", "big brothers big sisters", "junior achievement",
    "national honor society", "beta club", "odyssey of the mind", "destination imagination",
    "first robotics", "girls who code", "khan academy", "accelerated reader", "star reading", "star math",
    "map growth", "read 180", "system 44", "st math", "imagine learning", "imagine math", "go math",
    "carnegie learning", "into reading", "into math", "study island", "reading plus", "lexia core5",
    "sanford harmony", "second step", "wilson reading", "orton gillingham", "lindamood bell",
    "fountas pinnell", "fountas and pinnell", "units of study", "guided reading", "head start",
    "early head start", "upward bound", "gear up", "title i", "title iii", "title ix", "title one",
    "gifted talented", "advanced placement", "dual credit", "early college", "cristo rey", "responsive ed",
    "idea public schools", "harmony public schools", "harmony science academy", "uncommon schools",
    "aspire public schools", "summit public schools", "noble network", "mastery charter", "uplift education",
    "basis charter", "great oaks", "mater academy", "imagine schools", "national heritage academies",
    "charter schools usa", "jack and jill", "alpha kappa alpha", "delta sigma theta", "phi theta kappa",
    "mu alpha theta", "kappa delta pi", "quill and scroll", "key club", "dell scholars", "gates scholarship",
    "gates millennium", "coca cola scholars", "burger king scholars", "national merit", "presidential scholars",
    "rhodes scholarship", "fulbright", "goldwater scholarship", "truman scholarship", "lewis and clark",
    "wright brothers", "reggio emilia", "jose marti mast", "dr seuss", "lamar consolidated",
    "davidson young scholars", "mckinney vento", "woodcock johnson", "woodcock munoz", "stanford binet",
    "piers harris", "kaufman brief", "beery vmi", "gray oral reading",
    # eponyms of science, math and history units
    "lewis clark", "lincoln douglas", "lewis dot", "lewis structure", "lewis structures", "charles law",
    "boyle law", "gay lussac", "hardy weinberg", "punnett square", "punnett squares", "venn diagram",
    "bunsen burner", "darwin finches", "jim crow", "magic tree house", "ivy and bean", "frog and toad",
    "pete the cat", "henry and mudge", "mercy watson", "dog man", "captain underpants", "nate the great",
    "horrible harry", "fancy nancy", "skippyjon jones", "bad kitty", "esperanza rising", "bud not buddy",
    "sarah plain and tall", "number the stars", "bridge to terabithia", "where the red fern grows",
    "the giver", "hatchet",
})

#: Vendor and brand names whose words are on the name lists (``Sherwin Williams``, ``Ben E. Keith``, ``Dell
#: Latitude``): ``brand`` names are never name words; ``person`` brands are also a very common person's name and
#: count only beside a business word or a number (:data:`BUSINESS_WORDS`: ``Sara Lee Bread``, ``Edward Jones
#: 403b``), so a student named ``Sara Lee`` still FAILs. One dict of more than 50 entries (data).
BRANDS: dict[str, str] = {
    "sherwin williams": "brand", "john deere": "brand", "lamb weston": "brand", "general mills": "brand",
    "little debbie": "brand", "mrs baird": "brand", "mrs fields": "brand", "mr clean": "brand",
    "dr pepper": "brand", "ben e keith": "brand", "tyler traversa": "brand", "tyler technologies": "brand",
    "tyler sis": "brand", "tyler munis": "brand", "tony pizza": "brand", "dell chromebook": "brand",
    "dell chromebooks": "brand", "dell latitude": "brand", "dell laptop": "brand", "dell laptops": "brand",
    "dell optiplex": "brand", "dell inspiron": "brand", "dell precision": "brand", "dell monitor": "brand",
    "dell monitors": "brand", "dell desktop": "brand", "dell desktops": "brand", "dell computers": "brand",
    "herff jones": "brand", "aunt annie": "brand", "uncle ben": "brand",
    "ben jerry": "brand", "ben and jerry": "brand", "dave buster": "brand", "dave and buster": "brand",
    "jimmy john": "brand", "papa john": "brand", "marco pizza": "brand", "sally beauty": "brand",
    "trader joe": "brand", "sam club": "brand", "winn dixie": "brand",
    "wells fargo": "brand", "johns hopkins": "brand", "johnson controls": "brand", "johnson johnson": "brand",
    "paul davis restoration": "brand", "smucker uncrustables": "brand", "gordon food service": "brand",
    "ruiz foods": "brand", "shamrock foods": "brand", "land olakes": "brand", "jr simplot": "brand",
    "j r simplot": "brand", "marie callender": "person", "chef boyardee": "brand", "jimmy dean": "person",
    "oscar mayer": "person", "robert wood johnson": "brand", "sara lee": "person", "edward jones": "person",
    "raymond james": "person", "benjamin moore": "person", "morgan stanley": "person", "martha stewart": "person",
    "paul davis": "person", "charles schwab": "person", "betty crocker": "person", "merrill lynch": "person",
}

#: Words after a person-like brand that make it a business: ``Sara Lee Bread``, ``Edward Jones Investments``.
BUSINESS_WORDS: frozenset[str] = frozenset({
    "bread", "bakery", "foods", "food", "products", "desserts", "frozen", "investments", "investment",
    "financial", "retirement", "annuity", "brokerage", "paint", "paints", "coatings", "inc", "llc", "ltd",
    "co", "corp", "company", "restoration", "vendor", "vendors", "invoice", "invoices", "deduction",
    "deductions", "account", "accounts", "order", "orders", "purchase", "purchases", "contract", "sausage",
    "sausages", "bacon", "ham", "meat", "meats", "breakfast", "bowls", "biscuits", "pies", "pie", "cakes", "cake",
    "mix", "mixes", "bank", "banking", "securities", "wealth", "advisors", "advisor", "pot", "dinners",
})

#: Words right after a name that make it an eponym of a law, model, structure, event or era: ``Charles's Law``,
#: ``Hardy-Weinberg Principle``, ``Lewis Dot Structures``, ``John Brown's Raid``, ``Ming Dynasty``.
EPONYM_WORDS: frozenset[str] = frozenset({
    "law", "laws", "theorem", "theorems", "principle", "principles", "effect", "equation", "equations",
    "constant", "constants", "structure", "structures", "diagram", "diagrams", "paradox", "conjecture",
    "lemma", "formula", "formulas", "sequence", "triangle", "burner", "finches", "rebellion", "revolt",
    "raid", "expedition", "expeditions", "debate", "debates", "trial", "doctrine", "compromise",
    "proclamation", "treaty", "administration", "presidency", "dynasty", "empire", "era", "war", "wars",
    "revolution", "voyage", "voyages", "assassination", "conjectures", "hypothesis", "postulate",
    "postulates", "inequality", "identity", "distribution", "series", "cycle",
})

#: Suffixes, particles and honorifics left out when a namesake is compared (``Martin Luther King Jr``).
_DROP = frozenset({"jr", "sr", "ii", "iii", "iv", "de", "del", "la", "y", "and", "dr"})
#: Text tokens that may sit inside a namesake run without being part of the entry (``Martin Luther King Jr``,
#: ``Booker T. Washington`` for ``booker washington``, ``Dr. Martin Luther King``).
_SKIPPABLE = frozenset({"jr", "sr", "ii", "iii", "iv", "dr", "st"})
#: Name particles glued to the next word in a namesake entry (``Mary McLeod Bethune``: ``mc`` + ``leod``).
_GLUE = frozenset({"mc", "mac", "o", "de", "di", "da", "van", "von", "le", "la", "du", "st"})


@lru_cache(maxsize=1)
def namesake_words() -> frozenset[tuple[str, ...]]:
    """Sorted name-word tuples of :data:`NAMESAKES` (any order matches; suffixes, particles and honorifics
    dropped; an initial the entry holds is kept, so ``Robert Lee`` is not ``Robert E Lee``)."""
    out: set[tuple[str, ...]] = set()
    for phrase in NAMESAKES:
        words = tuple(sorted(w for w in phrase.split() if w not in _DROP))
        if len(words) >= 2:
            out.add(words)
    return frozenset(out)


@lru_cache(maxsize=1)
def program_phrases() -> dict[str, tuple[tuple[str, ...], ...]]:
    """:data:`PROGRAM_PHRASES` and the ``brand`` entries of :data:`BRANDS` (2 words or more) as token tuples
    keyed by their first token, longest first."""
    by_first: dict[str, list[tuple[str, ...]]] = {}
    for phrase in list(PROGRAM_PHRASES) + [b for b, kind in BRANDS.items() if kind == "brand"]:
        toks = tuple(phrase.split())
        if len(toks) >= 2:
            by_first.setdefault(toks[0], []).append(toks)
    return {k: tuple(sorted(set(v), key=len, reverse=True)) for k, v in by_first.items()}


@lru_cache(maxsize=1)
def phrase_index() -> dict[str, tuple[tuple[tuple[str, ...], str], ...]]:
    """:data:`NAMESAKES` (``namesake``; a two-word entry also in reverse order: ``Parks, Rosa``) and the
    ``person`` entries of :data:`BRANDS` (``person``) as (token tuple, kind) keyed by their first token,
    longest first."""
    by_first: dict[str, set[tuple[tuple[str, ...], str]]] = {}
    for phrase in NAMESAKES:
        toks = tuple(phrase.split())
        if len(toks) < 2:
            continue
        by_first.setdefault(toks[0], set()).add((toks, "namesake"))
        if len(toks) == 2:
            by_first.setdefault(toks[1], set()).add(((toks[1], toks[0]), "namesake"))
    for phrase, kind in BRANDS.items():
        if kind == "person":
            toks = tuple(phrase.split())
            by_first.setdefault(toks[0], set()).add((toks, "person"))
    return {k: tuple(sorted(v, key=lambda e: (-len(e[0]), e))) for k, v in by_first.items()}


def find_phrases(keys: list[str], skippable: list[bool]) -> list[tuple[int, int, str]]:
    """Runs of ``keys`` (folded word tokens) that spell an entry of :func:`phrase_index`: (first, last, kind).
    The entry's words must come in order; a token flagged ``skippable`` (an initial, particle or suffix of the
    text) may sit between them; a particle glued to the next word matches one entry word (``mc`` + ``leod``)."""
    index = phrase_index()
    out: list[tuple[int, int, str]] = []
    n = len(keys)
    i = 0
    while i < n:
        best: tuple[int, str] | None = None
        for seq, kind in index.get(keys[i], ()):
            j, m, end = i, 0, -1
            while j < n and m < len(seq):
                if keys[j] == seq[m]:
                    m, end, j = m + 1, j, j + 1
                elif keys[j] in _GLUE and j + 1 < n and keys[j] + keys[j + 1] == seq[m]:
                    m, end, j = m + 1, j + 1, j + 2
                elif m > 0 and (skippable[j] or keys[j] in _SKIPPABLE):
                    j += 1
                else:
                    break
            if m == len(seq) and (best is None or end > best[0]):
                best = (end, kind)
        if best is not None:
            out.append((i, best[0], best[1]))
            i = best[0] + 1
        else:
            i += 1
    return out


def is_namesake(words: tuple[str, ...] | list[str]) -> bool:
    """The folded name words of a person-name run are exactly one namesake of two words or more (any order;
    suffixes and particles aside)."""
    key = tuple(sorted(w for w in words if w and w not in _DROP))
    return len(key) >= 2 and key in namesake_words()


__all__ = ["NAMESAKES", "PROGRAM_WORDS", "PROGRAM_PHRASES", "BRANDS", "BUSINESS_WORDS", "EPONYM_WORDS",
           "namesake_words", "program_phrases", "phrase_index", "find_phrases", "is_namesake"]

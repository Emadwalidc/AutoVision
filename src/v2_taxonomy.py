"""Complete AutoVision v2 taxonomy (49 A-Z makes, 555+ models, 1950-2026 years, 8 body types).

This module unifies label naming between `timdwms/us-car-models-1948-2026` and `VMMRdb`,
provides curated Body Type mappings + per-photo Wikimedia text refinement, parses
generation/facelift year intervals, and computes YOLO-consistent padded crop boxes.
"""

from __future__ import annotations

import re

YEAR_MIN_V2: int = 1950
YEAR_MAX_V2: int = 2026
NUM_YEAR_BINS_V2: int = YEAR_MAX_V2 - YEAR_MIN_V2 + 1

BODY_TYPE_CLASSES: tuple[str, ...] = (
    "Sedan",
    "SUV",
    "Pickup",
    "Minivan",
    "Coupe",
    "Hatchback",
    "Wagon",
    "Convertible",
)

# Canonical alias remaps unifying VMMRdb & timdwms spelling differences.
_MODEL_ALIASES: dict[tuple[str, str], tuple[str, str]] = {
    ("dodge", "ram_1500"): ("ram", "1500"),
    ("dodge", "ram_hd"): ("ram", "ram_hd"),
    ("mercedes-benz", "sl"): ("mercedes-benz", "sl-class"),
    ("mini", "cooper_countryman"): ("mini", "countryman"),
    ("saab", "9-7"): ("saab", "9-7x"),
    ("toyota", "landcruiser"): ("toyota", "land_cruiser"),
    ("toyota", "fjcruiser"): ("toyota", "fj_cruiser"),
    ("mazda", "mx5_miata"): ("mazda", "mx-5_miata"),
    ("mazda", "rx-7"): ("mazda", "rx7"),
    ("mazda", "rx-8"): ("mazda", "rx8"),
}

_MAKE_DISPLAY_OVERRIDES: dict[str, str] = {
    "alfa-romeo": "Alfa Romeo",
    "bmw": "BMW",
    "gmc": "GMC",
    "land-rover": "Land Rover",
    "mercedes-benz": "Mercedes-Benz",
    "mg": "MG",
    "mini": "MINI",
    "ram": "RAM",
    "rolls-royce": "Rolls-Royce",
}

_TOKEN_DISPLAY_OVERRIDES: dict[str, str] = {
    "amg": "AMG",
    "ats": "ATS",
    "brz": "BRZ",
    "bz4x": "bZ4X",
    "c-hr": "C-HR",
    "c-max": "C-Max",
    "cc": "CC",
    "cl": "CL",
    "cl-class": "CL-Class",
    "cla-class": "CLA-Class",
    "clk-class": "CLK-Class",
    "cls-class": "CLS-Class",
    "cr-v": "CR-V",
    "cr-z": "CR-Z",
    "ct": "CT",
    "ct4": "CT4",
    "ct5": "CT5",
    "ct6": "CT6",
    "cts": "CTS",
    "cx3": "CX-3",
    "cx30": "CX-30",
    "cx5": "CX-5",
    "cx50": "CX-50",
    "cx7": "CX-7",
    "cx9": "CX-9",
    "cx90": "CX-90",
    "delsol": "del Sol",
    "deville": "DeVille",
    "e-pace": "E-Pace",
    "e-tron": "e-tron",
    "e350": "E-350",
    "elr": "ELR",
    "eqe": "EQE",
    "eqs": "EQS",
    "es": "ES",
    "ev": "EV",
    "ev6": "EV6",
    "ev9": "EV9",
    "ex30": "EX30",
    "ex90": "EX90",
    "f-pace": "F-Pace",
    "f-type": "F-Type",
    "f100": "F-100",
    "f150": "F-150",
    "f250-f350": "F-250/F-350",
    "fj": "FJ",
    "frs": "FR-S",
    "fx35": "FX35",
    "g70": "G70",
    "g80": "G80",
    "g90": "G90",
    "gl-class": "GL-Class",
    "gla-class": "GLA-Class",
    "glb-class": "GLB-Class",
    "glc-class": "GLC-Class",
    "gle-class": "GLE-Class",
    "glk-class": "GLK-Class",
    "gls-class": "GLS-Class",
    "grandam": "Grand Am",
    "grandmarquis": "Grand Marquis",
    "grandprix": "Grand Prix",
    "granturismo": "GranTurismo",
    "gs": "GS",
    "gt": "GT",
    "gti": "GTI",
    "gto": "GTO",
    "gtr": "GT-R",
    "gv60": "GV60",
    "gv70": "GV70",
    "gv80": "GV80",
    "gx": "GX",
    "hd": "HD",
    "hhr": "HHR",
    "hr-v": "HR-V",
    "hs": "HS",
    "i-miev": "i-MiEV",
    "i-pace": "I-Pace",
    "i3": "i3",
    "i30": "I30",
    "i8": "i8",
    "id.4": "ID.4",
    "ilx": "ILX",
    "iq": "iQ",
    "is": "IS",
    "ix": "iX",
    "k900": "K900",
    "lacrosse": "LaCrosse",
    "lc": "LC",
    "lesabre": "LeSabre",
    "ls": "LS",
    "lx": "LX",
    "m2": "M2",
    "m3": "M3",
    "m35": "M35",
    "m4": "M4",
    "m5": "M5",
    "mach-e": "Mach-E",
    "mdx": "MDX",
    "mkc": "MKC",
    "mks": "MKS",
    "mkt": "MKT",
    "mkx": "MKX",
    "mkz": "MKZ",
    "montecarlo": "Monte Carlo",
    "mpv": "MPV",
    "mr2": "MR2",
    "mx-5": "MX-5",
    "nsx": "NSX",
    "nv200": "NV200",
    "nx": "NX",
    "parkavenue": "Park Avenue",
    "promaster": "ProMaster",
    "pt": "PT",
    "q3": "Q3",
    "q4": "Q4",
    "q5": "Q5",
    "q50": "Q50",
    "q60": "Q60",
    "q7": "Q7",
    "q70": "Q70",
    "q8": "Q8",
    "qx30": "QX30",
    "qx4": "QX4",
    "qx50": "QX50",
    "qx55": "QX55",
    "qx56": "QX56",
    "qx60": "QX60",
    "qx80": "QX80",
    "r1s": "R1S",
    "r1t": "R1T",
    "r8": "R8",
    "ram": "Ram",
    "rangerover": "Range Rover",
    "rav4": "RAV4",
    "rc": "RC",
    "rdx": "RDX",
    "rl": "RL",
    "rlx": "RLX",
    "rsx": "RSX",
    "rx": "RX",
    "rx7": "RX-7",
    "rx8": "RX-8",
    "rz": "RZ",
    "s10": "S-10",
    "s2000": "S2000",
    "s4": "S4",
    "s40": "S40",
    "s60": "S60",
    "s70": "S70",
    "s80": "S80",
    "s90": "S90",
    "santafe": "Santa Fe",
    "sl-class": "SL-Class",
    "slk-class": "SLK-Class",
    "srx": "SRX",
    "sts": "STS",
    "suv": "SUV",
    "sx4": "SX4",
    "tc": "tC",
    "tl": "TL",
    "tlx": "TLX",
    "town&country": "Town & Country",
    "towncar": "Town Car",
    "transam": "Trans Am",
    "tsx": "TSX",
    "tt": "TT",
    "tx": "TX",
    "ux": "UX",
    "v50": "V50",
    "v60": "V60",
    "v70": "V70",
    "v90": "V90",
    "wrx": "WRX",
    "x1": "X1",
    "x2": "X2",
    "x3": "X3",
    "x4": "X4",
    "x5": "X5",
    "x6": "X6",
    "x7": "X7",
    "xb": "xB",
    "xc40": "XC40",
    "xc60": "XC60",
    "xc70": "XC70",
    "xc90": "XC90",
    "xd": "xD",
    "xe": "XE",
    "xf": "XF",
    "xj": "XJ",
    "xk": "XK",
    "xlr": "XLR",
    "xt4": "XT4",
    "xt5": "XT5",
    "xt6": "XT6",
    "xts": "XTS",
    "yukon": "Yukon",
    "zdx": "ZDX",
}

# Explicit 8-class body mapping across timdwms 555 models + VMMRdb models.
# Remaining sedans resolve cleanly to the default "Sedan".
_V2_BODY_GROUPS: dict[str, tuple[str, ...]] = {
    "SUV": (
        "acura__mdx", "acura__rdx", "acura__zdx",
        "alfa-romeo__stelvio", "alfa-romeo__tonale",
        "audi__e-tron", "audi__q3", "audi__q4_e-tron", "audi__q5", "audi__q7", "audi__q8",
        "bmw__ix", "bmw__x1", "bmw__x2", "bmw__x3", "bmw__x4", "bmw__x5", "bmw__x6", "bmw__x7",
        "buick__enclave", "buick__encore", "buick__encore_gx", "buick__envision",
        "buick__envista", "buick__rendezvous",
        "cadillac__escalade", "cadillac__lyriq", "cadillac__srx",
        "cadillac__xt4", "cadillac__xt5", "cadillac__xt6",
        "chevrolet__blazer", "chevrolet__blazer_ev", "chevrolet__equinox",
        "chevrolet__equinox_ev", "chevrolet__suburban", "chevrolet__tahoe",
        "chevrolet__tracker", "chevrolet__trailblazer", "chevrolet__traverse",
        "chevrolet__trax",
        "chrysler__aspen",
        "dodge__durango", "dodge__hornet", "dodge__journey", "dodge__nitro",
        "fiat__500x",
        "ford__bronco", "ford__bronco_sport", "ford__ecosport", "ford__edge",
        "ford__escape", "ford__excursion", "ford__expedition", "ford__explorer",
        "ford__flex", "ford__mustang_mach-e",
        "genesis__gv60", "genesis__gv70", "genesis__gv80",
        "gmc__acadia", "gmc__envoy", "gmc__hummer_ev", "gmc__jimmy",
        "gmc__suburban_1500", "gmc__terrain", "gmc__yukon_1500",
        "honda__cr-v", "honda__element", "honda__hr-v", "honda__passport",
        "honda__pilot", "honda__prologue",
        "hummer__h2", "hummer__h3",
        "hyundai__kona", "hyundai__palisade", "hyundai__santafe",
        "hyundai__tucson", "hyundai__venue", "hyundai__veracruz",
        "infiniti__fx35", "infiniti__qx30", "infiniti__qx4", "infiniti__qx50",
        "infiniti__qx55", "infiniti__qx56", "infiniti__qx60", "infiniti__qx80",
        "isuzu__rodeo", "isuzu__trooper",
        "jaguar__e-pace", "jaguar__f-pace", "jaguar__i-pace",
        "jeep__cherokee", "jeep__commander", "jeep__compass", "jeep__grand_cherokee",
        "jeep__liberty", "jeep__patriot", "jeep__renegade", "jeep__wagoneer",
        "jeep__wrangler",
        "kia__borrego", "kia__ev6", "kia__ev9", "kia__niro", "kia__seltos",
        "kia__sorento", "kia__sportage", "kia__telluride",
        "land-rover__defender", "land-rover__discovery", "land-rover__discovery_sport",
        "land-rover__freelander", "land-rover__rangerover",
        "land-rover__rangerover_evoque", "land-rover__rangerover_sport",
        "land-rover__rangerover_velar",
        "lexus__gx", "lexus__lx", "lexus__nx", "lexus__rx", "lexus__rz",
        "lexus__tx", "lexus__ux",
        "lincoln__aviator", "lincoln__corsair", "lincoln__mkc", "lincoln__mkt",
        "lincoln__mkx", "lincoln__nautilus", "lincoln__navigator",
        "maserati__grecale", "maserati__levante",
        "mazda__cx3", "mazda__cx30", "mazda__cx5", "mazda__cx50", "mazda__cx7",
        "mazda__cx9", "mazda__cx90", "mazda__tribute",
        "mercedes-benz__eqe_suv", "mercedes-benz__eqs_suv", "mercedes-benz__g-class",
        "mercedes-benz__gl-class", "mercedes-benz__gla-class", "mercedes-benz__glb-class",
        "mercedes-benz__glc-class", "mercedes-benz__gle-class", "mercedes-benz__glk-class",
        "mercedes-benz__gls-class", "mercedes-benz__m-class",
        "mercury__mariner", "mercury__mountaineer",
        "mini__countryman", "mini__paceman",
        "mitsubishi__endeavor", "mitsubishi__montero", "mitsubishi__outlander",
        "nissan__ariya", "nissan__armada", "nissan__juke", "nissan__kicks",
        "nissan__murano", "nissan__pathfinder", "nissan__rogue", "nissan__x-trail",
        "nissan__xterra",
        "oldsmobile__bravada",
        "peugeot__3008", "plymouth__trailduster",
        "polestar__polestar_3",
        "pontiac__aztek", "pontiac__torrent",
        "porsche__cayenne", "porsche__macan",
        "renault__captur",
        "rivian__r1s",
        "saab__9-7x",
        "saturn__outlook", "saturn__vue",
        "subaru__ascent", "subaru__crosstrek", "subaru__forester",
        "subaru__solterra", "subaru__tribeca",
        "suzuki__grand_vitara", "suzuki__samurai", "suzuki__sidekick",
        "suzuki__vitara", "suzuki__x90", "suzuki__xl7",
        "tesla__model_x", "tesla__model_y",
        "toyota__4runner", "toyota__bz4x", "toyota__c-hr", "toyota__corolla_cross",
        "toyota__fj40", "toyota__fj_cruiser", "toyota__fortuner",
        "toyota__grand_highlander", "toyota__highlander", "toyota__land_cruiser",
        "toyota__rav4", "toyota__sequoia", "toyota__venza",
        "volkswagen__atlas", "volkswagen__id.4", "volkswagen__taos",
        "volkswagen__tiguan", "volkswagen__touareg",
        "volvo__c40", "volvo__ex30", "volvo__ex90", "volvo__xc40",
        "volvo__xc60", "volvo__xc90",
        "willys__cj2a", "willys__cj3b",
    ),
    "Pickup": (
        "chevrolet__avalanche", "chevrolet__c10", "chevrolet__colorado",
        "chevrolet__el_camino", "chevrolet__s10", "chevrolet__silverado",
        "chevrolet__silverado_ev",
        "dodge__dakota",
        "ford__f100", "ford__f150", "ford__f250-f350", "ford__maverick", "ford__ranger",
        "gmc__canyon", "gmc__sierra_1500", "gmc__sierra_hd", "gmc__sonoma",
        "honda__ridgeline",
        "hyundai__santa_cruz",
        "jeep__gladiator",
        "mitsubishi__mighty_max", "mitsubishi__raider",
        "nissan__720", "nissan__d21_hardbody", "nissan__frontier", "nissan__titan",
        "ram__1500", "ram__2500", "ram__3500", "ram__ram_hd",
        "rivian__r1t",
        "subaru__baja", "subaru__brat",
        "suzuki__equator",
        "tesla__cybertruck",
        "toyota__1-ton_truck", "toyota__hilux", "toyota__pickup",
        "toyota__t100", "toyota__tacoma", "toyota__tundra",
    ),
    "Minivan": (
        "chevrolet__astro", "chevrolet__express", "chevrolet__uplander", "chevrolet__venture",
        "chrysler__pacifica", "chrysler__town&country",
        "dodge__grand_caravan",
        "fiat__500l",
        "ford__c-max", "ford__e350", "ford__freestar", "ford__transit",
        "ford__transit_connect", "ford__windstar",
        "gmc__savana",
        "honda__odyssey",
        "kia__rondo", "kia__sedona",
        "mazda__5", "mazda__mpv",
        "mercedes-benz__metris", "mercedes-benz__r-class", "mercedes-benz__sprinter",
        "mercury__villager",
        "mitsubishi__chariot", "mitsubishi__delica",
        "nissan__nv200", "nissan__presage", "nissan__quest",
        "oldsmobile__silhouette", "opel__vivaro",
        "plymouth__voyager",
        "pontiac__montana",
        "ram__promaster", "ram__promaster_city",
        "saturn__relay",
        "toyota__estima", "toyota__hiace", "toyota__previa", "toyota__sienna",
        "volkswagen__bus", "volkswagen__camper", "volkswagen__eurovan",
        "volkswagen__routan", "volkswagen__vanagon",
    ),
    "Wagon": (
        "dodge__magnum",
        "mini__clubman",
        "peugeot__407_wagon",
        "saab__9-2",
        "subaru__outback",
        "volkswagen__squareback",
        "volvo__145_wagon", "volvo__245", "volvo__v40", "volvo__v50",
        "volvo__v60", "volvo__v70", "volvo__v90", "volvo__xc70",
    ),
    "Convertible": (
        "buick__cascada",
        "cadillac__xlr",
        "fiat__124_spider",
        "honda__delsol", "honda__s2000",
        "mazda__mx-5_miata",
        "mercedes-benz__sl-class", "mercedes-benz__slk-class",
        "mg__mgb", "mg__midget",
        "plymouth__prowler",
        "pontiac__solstice",
        "porsche__914", "porsche__boxster",
        "rolls-royce__corniche",
        "saturn__sky",
        "tesla__roadster",
        "triumph__spitfire", "triumph__tr",
        "volkswagen__cabrio", "volkswagen__dunebuggy", "volkswagen__eos",
    ),
    "Coupe": (
        "acura__cl", "acura__nsx", "acura__rsx",
        "alfa-romeo__4c",
        "audi__a5", "audi__r8", "audi__tt",
        "bmw__2_series", "bmw__4_series", "bmw__6_series", "bmw__8_series",
        "bmw__i8", "bmw__m2", "bmw__m4", "bmw__z4",
        "cadillac__eldorado", "cadillac__elr",
        "chevrolet__camaro", "chevrolet__corvette", "chevrolet__montecarlo",
        "chrysler__crossfire",
        "dodge__challenger", "dodge__viper",
        "ford__gt", "ford__mustang", "ford__thunderbird",
        "honda__cr-z", "honda__prelude",
        "hyundai__tiburon",
        "infiniti__q60",
        "jaguar__f-type", "jaguar__xk",
        "lexus__lc", "lexus__rc",
        "maserati__granturismo",
        "mazda__rx7", "mazda__rx8",
        "mercedes-benz__amg_gt", "mercedes-benz__cl-class",
        "mercedes-benz__clk-class", "mercedes-benz__slr",
        "mercury__capri", "mercury__cougar", "mercury__cyclone",
        "mitsubishi__3000gt", "mitsubishi__eclipse", "mitsubishi__starion",
        "nash__metropolitan",
        "nissan__200sx", "nissan__240sx", "nissan__300zx", "nissan__350z",
        "nissan__370z", "nissan__gtr", "nissan__skyline", "nissan__z",
        "oldsmobile__442", "oldsmobile__starfire", "oldsmobile__toronado",
        "opel__gt",
        "plymouth__barracuda", "plymouth__duster", "plymouth__gtx",
        "plymouth__laser", "plymouth__roadrunner", "plymouth__scamp",
        "pontiac__fiero", "pontiac__firebird", "pontiac__gto", "pontiac__transam",
        "porsche__911", "porsche__912", "porsche__917", "porsche__924",
        "porsche__928", "porsche__944", "porsche__cayman",
        "saturn__sc",
        "scion__frs", "scion__tc",
        "subaru__brz", "subaru__svx",
        "sunbeam__rapier",
        "toyota__86", "toyota__celica", "toyota__mr2", "toyota__paseo",
        "toyota__solara", "toyota__supra",
        "triumph__gt6",
        "volkswagen__corrado", "volkswagen__karmannghia", "volkswagen__scirocco",
        "volvo__142", "volvo__242", "volvo__780", "volvo__c70",
    ),
    "Hatchback": (
        "acura__integra",
        "audi__a3",
        "bmw__1_series", "bmw__i3",
        "chevrolet__aveo", "chevrolet__bolt", "chevrolet__hhr",
        "chevrolet__sonic", "chevrolet__spark", "chevrolet__volt",
        "chrysler__pt_cruiser",
        "dodge__caliber",
        "ford__fiesta", "ford__focus",
        "honda__fit", "honda__insight",
        "hyundai__ioniq", "hyundai__ioniq_5", "hyundai__veloster",
        "kia__rio", "kia__soul",
        "lexus__ct",
        "mazda__2", "mazda__3", "mazda__speed3",
        "mercedes-benz__a-class", "mercedes-benz__b-class",
        "mini__austin", "mini__cooper",
        "mitsubishi__colt", "mitsubishi__i-miev", "mitsubishi__mirage",
        "nissan__cube", "nissan__leaf", "nissan__pulsar",
        "opel__astra",
        "plymouth__colt", "plymouth__horizon", "plymouth__sundance",
        "polestar__polestar_2",
        "pontiac__g39", "pontiac__vibe", "pontiac__wave",
        "saab__9-6",
        "saturn__astra",
        "scion__iq", "scion__xa", "scion__xb", "scion__xd",
        "smart__fortwo",
        "subaru__360",
        "suzuki__reno", "suzuki__swift", "suzuki__sx4",
        "toyota__matrix", "toyota__prius", "toyota__starlet", "toyota__yaris",
        "volkswagen__beetle", "volkswagen__golf", "volkswagen__gti", "volkswagen__r32",
        "volvo__c30",
    ),
}

_CANONICAL_BODY_LOOKUP: dict[str, str] = {
    cls_name: body_type
    for body_type, items in _V2_BODY_GROUPS.items()
    for cls_name in items
}


def _slugify_part(raw: str) -> str:
    return re.sub(r"\s+", "_", raw.strip().lower())


def _format_part_display(slug: str) -> str:
    tokens = slug.replace("_", " ").split()
    out: list[str] = []
    for tok in tokens:
        if tok in _TOKEN_DISPLAY_OVERRIDES:
            out.append(_TOKEN_DISPLAY_OVERRIDES[tok])
        elif "-" in tok:
            sub = tok.split("-")
            if all(len(s) <= 3 and s.isalnum() for s in sub) and not any(s.isalpha() and len(s) > 2 for s in sub):
                out.append(tok.upper())
            else:
                out.append("-".join(p[:1].upper() + p[1:] if p else "" for p in sub))
        elif tok.isdigit():
            out.append(tok)
        elif tok[:1].isdigit() and any(ch.isalpha() for ch in tok):
            out.append(tok.upper())
        else:
            out.append(tok[:1].upper() + tok[1:])
    return " ".join(out)


def canonicalize_make_model(raw_make: str, raw_model: str) -> tuple[str, str, str, str]:
    """Returns `(canonical_make, canonical_class, make_display, model_display)`."""
    mk = raw_make.strip().lower().replace(" ", "-")
    md = _slugify_part(raw_model)

    if (mk, md) in _MODEL_ALIASES:
        mk, md = _MODEL_ALIASES[(mk, md)]

    canonical_class = f"{mk}__{md}"
    make_display = _MAKE_DISPLAY_OVERRIDES.get(
        mk, "-".join(p[:1].upper() + p[1:] for p in mk.split("-") if p)
    )
    if canonical_class == "polestar__polestar_2":
        model_display = "Polestar 2"
    elif canonical_class == "polestar__polestar_3":
        model_display = "Polestar 3"
    else:
        model_display = _format_part_display(md)
    return mk, canonical_class, make_display, model_display


def resolve_default_body_type(canonical_class: str) -> str:
    """Resolves the primary Body Type for any canonical class (defaults cleanly to Sedan)."""
    return _CANONICAL_BODY_LOOKUP.get(canonical_class, "Sedan")


def refine_body_type_from_text(
    default_body: str,
    commons_category: str = "",
    commons_title: str = "",
) -> str:
    """Refines per-image Body Type from Wikimedia category/title cues on multi-body passenger cars."""
    if default_body in ("SUV", "Pickup", "Minivan"):
        return default_body

    text = f"{commons_category} {commons_title}".lower()
    if not text.strip():
        return default_body

    # Ignore 4-door coupe marketing terms that remain Sedans unless explicitly Cabrio/Wagon.
    text_clean = (
        text.replace("gran coupe", " ")
        .replace("4-door coupe", " ")
        .replace("four-door coupe", " ")
        .replace("suv coupe", " ")
    )

    if any(k in text_clean for k in ("convertible", "cabrio", "cabriolet", "roadster", "spyder", "spider", "drophead")):
        return "Convertible"
    if any(k in text_clean for k in ("wagon", "touring", "avant", "estate", "sportwagen", "variant", "shooting brake", "alltrack")):
        return "Wagon"
    if any(k in text_clean for k in ("hatchback", "5-door", "3-door", "sportback")):
        return "Hatchback"
    if "coupe" in text_clean or "coupé" in text_clean:
        return "Coupe"
    return default_body


def parse_timdwms_year_interval(
    years_str: str,
    phase_str: str = "",
    date_taken_str: str = "",
) -> tuple[float, float, bool]:
    """Parses `timdwms` generation spans and narrows wide intervals using phase and photo date."""
    matches = [int(y) for y in re.findall(r"(?:19|20)\d{2}", years_str or "")]
    if not matches:
        return 0.0, 0.0, False

    lo = float(max(YEAR_MIN_V2, min(YEAR_MAX_V2, min(matches))))
    hi = float(max(YEAR_MIN_V2, min(YEAR_MAX_V2, max(matches))))
    if hi < lo:
        hi = lo

    span = hi - lo
    phase = (phase_str or "").strip().lower()
    if span >= 4.0:
        if phase == "pre":
            hi = round(lo + 0.55 * span, 1)
        elif phase == "facelift":
            lo = round(lo + 0.45 * span, 1)
        elif phase in ("facelift2", "facelift3"):
            lo = round(lo + 0.65 * span, 1)

    if date_taken_str:
        dt_matches = [int(y) for y in re.findall(r"(?:19|20)\d{2}", date_taken_str[:10])]
        if dt_matches:
            dt_year = float(dt_matches[0])
            if lo <= dt_year < hi - 1.0:
                hi = min(hi, dt_year + 1.0)

    if hi < lo:
        hi = lo
    return lo, hi, True


def compute_padded_crop_box(
    car_box_str: str,
    img_w: int,
    img_h: int,
    pad: float = 0.05,
) -> tuple[int, int, int, int] | None:
    """Parses space-separated `x1 y1 x2 y2` crop boxes with 5% padding matching live YOLOv8n."""
    if img_w <= 16 or img_h <= 16 or not (car_box_str or "").strip():
        return None
    parts = car_box_str.strip().split()
    if len(parts) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(p) for p in parts)
    except ValueError:
        return None

    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    if bw < 24.0 or bh < 24.0:
        return None

    px = bw * float(pad)
    py = bh * float(pad)
    cx1 = max(0, int(round(x1 - px)))
    cy1 = max(0, int(round(y1 - py)))
    cx2 = min(int(img_w), int(round(x2 + px)))
    cy2 = min(int(img_h), int(round(y2 + py)))
    if (cx2 - cx1) < 24 or (cy2 - cy1) < 24:
        return None
    return cx1, cy1, cx2, cy2

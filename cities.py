# Vietnamese cities and provinces for location matching
VIETNAMESE_CITIES = {
    # Major cities
    "hà nội": ["ha noi", "hn", "hanoi"],
    "hồ chí minh": ["ho chi minh", "hcm", "saigon", "sài gòn", "sai gon", "sg"],
    "đà nẵng": ["da nang", "danang", "dn"],
    "cần thơ": ["can tho", "canho", "ct"],
    "hải phòng": ["hai phong", "haiphong", "hp"],
    
    # Provinces
    "bình dương": ["binh duong", "bd"],
    "đồng nai": ["dong nai"],
    "bà rịa - vũng tàu": ["ba ria vung tau", "vung tau", "vt"],
    "long an": [],
    "tiền giang": ["tien giang"],
    "bến tre": ["ben tre"],
    "trà vinh": ["tra vinh"],
    "vĩnh long": ["vinh long"],
    "an giang": [],
    "kiên giang": ["kien giang"],
    "cà mau": ["ca mau"],
    "sóc trăng": ["soc trang"],
    
    "quảng ninh": ["quang ninh"],
    "hải dương": ["hai duong"],
    "hưng yên": ["hung yen"],
    "hà nam": ["ha nam"],
    "nam định": ["nam dinh"],
    "ninh bình": ["ninh binh"],
    "thanh hoá": ["thanh hoa"],
    "nghệ an": ["nghe an"],
    "hà tĩnh": ["ha tinh"],
    "quảng bình": ["quang binh"],
    "quảng trị": ["quang tri"],
    "thừa thiên huế": ["thua thien hue", "hue"],
    
    "quảng nam": ["quang nam"],
    "quảng ngãi": ["quang ngai"],
    "bình định": ["binh dinh"],
    "phú yên": ["phu yen"],
    "khánh hoà": ["khanh hoa", "nha trang"],
    "ninh thuận": ["ninh thuan"],
    "bình thuận": ["binh thuan"],
    
    "đắk lắk": ["dak lak"],
    "đắk nông": ["dak nong"],
    "gia lai": ["gia lai"],
    "kon tum": ["kon tum"],
    "lâm đồng": ["lam dong", "da lat"],
    
    "bắc kạn": ["bac kan"],
    "cao bằng": ["cao bang"],
    "lạng sơn": ["lang son"],
    "bắc giang": ["bac giang"],
    "phú thọ": ["phu tho"],
    "vĩnh phúc": ["vinh phuc"],
    "tuyên quang": ["tuyen quang"],
    "yên bái": ["yen bai"],
    "sơn la": ["son la"],
    "hòa bình": ["hoa binh"],
    "điện biên": ["dien bien"],
    "lai châu": ["lai chau"],
    
    "thái nguyên": ["thai nguyen"],
    "bắc thái": ["bac thai"],
}

def get_canonical_city(text):
    """Extract canonical city name from text"""
    if not text:
        return ""
    
    text_normalized = text.lower().strip()
    text_normalized = text_normalized.replace("ă", "a").replace("â", "a")
    text_normalized = text_normalized.replace("ê", "e").replace("ế", "e").replace("è", "e")
    text_normalized = text_normalized.replace("ô", "o").replace("ơ", "o")
    text_normalized = text_normalized.replace("ư", "u").replace("ũ", "u")
    text_normalized = text_normalized.replace("đ", "d")
    text_normalized = text_normalized.replace("ị", "i").replace("ị", "i")
    
    for canonical, aliases in VIETNAMESE_CITIES.items():
        canonical_norm = canonical.lower()
        if canonical_norm in text_normalized:
            return canonical_norm
        for alias in aliases:
            if alias.lower() in text_normalized:
                return canonical_norm
    
    return ""


def is_city_match(cv_city, job_city):
    """Check if two city references match"""
    if not cv_city or not job_city:
        return False
    
    cv_canonical = get_canonical_city(cv_city)
    job_canonical = get_canonical_city(job_city)
    
    if cv_canonical and job_canonical:
        return cv_canonical == job_canonical
    
    return False

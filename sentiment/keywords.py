# region imports
from AlgorithmImports import *
# endregion
keywords = {
    "PFE.TiingoNews": ["pfizer", "pfe", "pharmaceutical", "healthcare", "drug", "vaccine", "biotech"],
    "KO.TiingoNews": ["coca-cola", "ko", "beverage", "soda", "consumer goods", "brand", "market share"],
    "DGX.TiingoNews": ["quest", "dgx", "diagnostics", "laboratory", "health services", "medical testing"],
    "PKG.TiingoNews": ["packaging corp", "pkg", "packaging", "manufacturing", "supply chain"],
    "NMM.TiingoNews": ["navios", "nmm", "shipping", "logistics", "maritime", "transport"],
    "WLL.TiingoNews": ["oasis", "wll", "oil", "gas", "energy", "exploration", "production"],
    "MSFT.TiingoNews": ["microsoft", "msft", "technology", "cloud", "windows", "software", "enterprise"],
    "HBC.TiingoNews": ["hsbc", "bank", "finance", "international", "investment banking", "financial services"],
    "UNH.TiingoNews": ["unitedhealth", "unh", "healthcare", "insurance", "medical", "services", "health"],
    "ASMLF.TiingoNews": ["asml", "semiconductor", "equipment", "manufacturing", "europe", "chipmaker"],
    "TSM.TiingoNews": ["tsm", "taiwan semiconductor", "chips", "hardware", "asia", "foundry"],
    "FPL.TiingoNews": ["nextera energy", "nee", "utilities", "renewables", "solar", "wind", "clean energy"],
    "BHP.TiingoNews": ["bhp", "mining", "resources", "commodities", "metals", "materials"],
    "DIS.TiingoNews": ["disney", "dis", "entertainment", "media", "streaming", "parks", "movies"],
    "WMT.TiingoNews": ["walmart", "wmt", "retail", "consumer", "grocery", "e-commerce", "supply chain"]
}


tiingo_to_ticker = {
    "PFE.TiingoNews": "PFE",
    "KO.TiingoNews": "KO",
    "DGX.TiingoNews": "DGX",
    "PKG.TiingoNews": "PKG",
    "NMM.TiingoNews": "NMM",
    "WLL.TiingoNews": "WLL",
    "MSFT.TiingoNews": "MSFT",
    "HBC.TiingoNews": "HSBC", 
    "UNH.TiingoNews": "UNH",
    "ASMLF.TiingoNews": "ASML",  
    "TSM.TiingoNews": "TSM",
    "FPL.TiingoNews": "NEE",  
    "BHP.TiingoNews": "BHP",
    "DIS.TiingoNews": "DIS",
    "WMT.TiingoNews": "WMT"
}    
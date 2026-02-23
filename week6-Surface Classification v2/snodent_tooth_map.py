"""
SNODENT-to-FDI Tooth Mapping
=============================

Comprehensive mapping between SNODENT dental codes used in AIM XML annotations
and FDI (Fédération Dentaire Internationale) two-digit tooth numbering system
(ISO 3950). Also includes surface code mappings.

Reference:
- FDI Notation: https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation
- SNODENT: Systematized Nomenclature of Dentistry

Author: Senior Research Engineer - Dental AI / CAD
Date: 2026
"""

# =============================================================================
# SNODENT Code → FDI Tooth ID Mapping
# =============================================================================
# Key = SNODENT code from XML <ImagingPhysicalEntityCharacteristic>
# Value = FDI two-digit tooth number
#
# FDI Quadrants:
#   Q1 (11–18): Upper Right    Q2 (21–28): Upper Left
#   Q3 (31–38): Lower Left     Q4 (41–48): Lower Right

SNODENT_TO_FDI = {
    # ===================== UPPER RIGHT (Quadrant 1) =====================
    "161006D": "11",  # Permanent upper right central incisor tooth
    "160842D": "12",  # Permanent upper right lateral incisor tooth
    "160288D": "13",  # Permanent upper right canine tooth
    "161286D": "14",  # Permanent upper right first premolar tooth
    "160450D": "15",  # Permanent upper right second premolar tooth
    "160770D": "16",  # Permanent upper right first molar tooth       # NOTE: also used for lower in some datasets
    "161204D": "17",  # Permanent upper right second molar tooth
    "160618D": "18",  # Permanent upper right third molar tooth

    # ===================== UPPER LEFT (Quadrant 2) =====================
    "160194D": "21",  # Permanent upper left central incisor tooth
    "160132D": "22",  # Permanent upper left lateral incisor tooth
    "160506D": "23",  # Permanent upper left canine tooth
    "161340D": "24",  # Permanent upper left first premolar tooth
    "160682D": "25",  # Permanent upper left second premolar tooth
    "161074D": "26",  # Permanent upper left first molar tooth
    "160386D": "27",  # Permanent upper left second molar tooth
    "160922D": "28",  # Permanent upper left third molar tooth

    # ===================== LOWER LEFT (Quadrant 3) =====================
    "161136D": "31",  # Permanent lower left central incisor tooth
    "160556D": "32",  # Permanent lower left lateral incisor tooth
    "160068D": "33",  # Permanent lower left canine tooth
    "160326D": "34",  # Permanent lower left first premolar tooth
    "161248D": "35",  # Permanent lower left second premolar tooth
    "160730D": "36",  # Permanent lower left first molar tooth
    "161166D": "37",  # Permanent lower left second molar tooth
    "160580D": "38",  # Permanent lower left third molar tooth

    # ===================== LOWER RIGHT (Quadrant 4) =====================
    "160964D": "41",  # Permanent lower right central incisor tooth
    "160350D": "42",  # Permanent lower right lateral incisor tooth
    "160894D": "43",  # Permanent lower right canine tooth
    "160230D": "44",  # Permanent lower right first premolar tooth
    "161412D": "45",  # Permanent lower right second premolar tooth
    "160770D": "46",  # Permanent lower right first molar tooth       # CAUTION: shared code
    "161102D": "47",  # Permanent lower right second molar tooth
    "160488D": "48",  # Permanent lower right third molar tooth
}

# Reverse mapping: FDI → SNODENT (may have multiple SNODENT codes per FDI)
FDI_TO_SNODENT = {}
for code, fdi in SNODENT_TO_FDI.items():
    if fdi not in FDI_TO_SNODENT:
        FDI_TO_SNODENT[fdi] = []
    FDI_TO_SNODENT[fdi].append(code)


# =============================================================================
# SNODENT Surface Code → Surface Name Mapping
# =============================================================================
# Key = SNODENT code from XML (questionIndex="1", "What part of the region...")
# Value = Standardized clinical surface name

SNODENT_SURFACE_MAP = {
    "144414D": "Occlusal",          # Occlusal surface
    "146014D": "Distal",            # Distal Surface
    "145374D": "Mesial",            # Mesial Surface
    "144854D": "Buccal",            # Buccal surface (cheek-side)
    "145134D": "Lingual",           # Lingual surface (tongue-side)
    "145714D": "Palatal",           # Palatal surface (upper tongue-side)
    "144474D": "Occlusal",          # Occlusal surface (alternate code)
    "146074D": "Distal",            # Distal surface (alternate code)
    "145434D": "Mesial",            # Mesial surface (alternate code)
    "144914D": "Buccal",            # Buccal surface (alternate code)
    "145194D": "Lingual",           # Lingual surface (alternate code)
    "145774D": "Cervical",          # Cervical (neck of tooth)
}

# Display-name-based fallback mapping (from iso:displayName in XML)
DISPLAY_NAME_TO_SURFACE = {
    "Occlusal surface": "Occlusal",
    "Occlusal Surface": "Occlusal",
    "Distal Surface": "Distal",
    "Distal surface": "Distal",
    "Mesial Surface": "Mesial",
    "Mesial surface": "Mesial",
    "Buccal surface": "Buccal",
    "Buccal Surface": "Buccal",
    "Lingual surface": "Lingual",
    "Lingual Surface": "Lingual",
    "Palatal surface": "Palatal",
    "Palatal Surface": "Palatal",
    "Cervical": "Cervical",
}


# =============================================================================
# FDI Tooth Name Display Lookup (for reporting)
# =============================================================================

FDI_TOOTH_NAMES = {
    "11": "UR Central Incisor",    "12": "UR Lateral Incisor",
    "13": "UR Canine",             "14": "UR First Premolar",
    "15": "UR Second Premolar",    "16": "UR First Molar",
    "17": "UR Second Molar",       "18": "UR Third Molar",
    "21": "UL Central Incisor",    "22": "UL Lateral Incisor",
    "23": "UL Canine",             "24": "UL First Premolar",
    "25": "UL Second Premolar",    "26": "UL First Molar",
    "27": "UL Second Molar",       "28": "UL Third Molar",
    "31": "LL Central Incisor",    "32": "LL Lateral Incisor",
    "33": "LL Canine",             "34": "LL First Premolar",
    "35": "LL Second Premolar",    "36": "LL First Molar",
    "37": "LL Second Molar",       "38": "LL Third Molar",
    "41": "LR Central Incisor",    "42": "LR Lateral Incisor",
    "43": "LR Canine",             "44": "LR First Premolar",
    "45": "LR Second Premolar",    "46": "LR First Molar",
    "47": "LR Second Molar",       "48": "LR Third Molar",
}


def snodent_display_to_fdi(display_name: str) -> str:
    """
    Parse an FDI tooth ID from the SNODENT display name string.
    
    E.g.: "Permanent lower right second premolar tooth" → "45"
    
    Uses keyword matching against quadrant/position patterns.
    
    Args:
        display_name: The iso:displayName value from XML
        
    Returns:
        FDI two-digit string, or "" if not parseable
    """
    dn = display_name.lower()
    
    # Determine quadrant
    if "upper" in dn and "right" in dn:
        quadrant = 1
    elif "upper" in dn and "left" in dn:
        quadrant = 2
    elif "lower" in dn and "left" in dn:
        quadrant = 3
    elif "lower" in dn and "right" in dn:
        quadrant = 4
    else:
        return ""
    
    # Determine tooth position within quadrant
    if "central incisor" in dn:
        pos = 1
    elif "lateral incisor" in dn:
        pos = 2
    elif "canine" in dn:
        pos = 3
    elif "first premolar" in dn:
        pos = 4
    elif "second premolar" in dn:
        pos = 5
    elif "third molar" in dn:
        pos = 8
    elif "second molar" in dn:
        pos = 7
    elif "first molar" in dn:
        pos = 6
    else:
        return ""
    
    return f"{quadrant}{pos}"

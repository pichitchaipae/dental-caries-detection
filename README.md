# 🦷 Dental Caries & Surface Detection System

A computer vision project designed to detect dental caries and classify the affected tooth surface using 3D point cloud analysis and image processing techniques. This project aims to assist dental professionals by automating the identification of tooth IDs and specific decay surfaces.

---

## 👥 Team Members

| Name | Student ID | GitHub | Email |
| :--- | :--- | :--- | :--- |
| **Sukollapat Pisuchpen** (Pond) | 6687052 | [@SukollapatPis](https://github.com/SukollapatPis) | sukollapat.pis@gmail.com |
| **Pichitchai Paecharoenchai** (Jao) | 6687033 | [@pichitchaipae](https://github.com/pichitchaipae) | jao.pichitchai@gmail.com |
| **Naris Pholpak** (Phai) | 6687025 | [@1tshadowz](https://github.com/1tshadowz) | phainaris@gmail.com |

**Faculty:** Faculty of Information and Communication Technology (ICT), Mahidol University  
**Advisor:** Dr. Sirawich Vachmanus (sirawich.vac@mahidol.ac.th)

---

## 🎯 Project Objectives

The main goal is to analyze dental data to identify caries with high precision.

### 1. Tooth Identification (FDI Notation) ✅ **Completed**
- **Goal:** Identify the specific tooth using the FDI World Dental Federation notation (Two-digit number).
- **Status:** Successfully implemented (Week 2-3).

### 2. Caries Surface Classification 🚧 **In Progress**
- **Goal:** Determine the specific surface of the tooth affected by caries.
- **Scope:** Focusing on 3 out of 5 main surfaces:
  - Buccal / Labial (Outer)
  - Lingual / Palatal (Inner)
  - Occlusal (Biting surface)
- **Current Status:** Algorithm development and refinement.

---

## 🛠️ Technology Stack

- **Language:** Python 3.x
- **Core Libraries:** `NumPy`, `Pandas`, `Open3D` (for point cloud processing), `Scikit-learn` (PCA/Classification)
- **Tools:** Git, GitHub

## 📂 Project Structure

```bash
dental-caries-detection/
├── data/                   # Raw and processed dental data (excluded from git)
├── docs/                   # Documentation and references
├── src/                    # Source code
│   ├── preprocessing/      # Data cleaning and point cloud preparation
│   ├── models/             # Classification models (FDI ID, Surface detection)
│   └── utils/              # Helper functions (visualization, file I/O)
├── notebooks/              # Jupyter notebooks for experiments
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

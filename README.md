# Phishing Detection — IEEE Research Implementation

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![IEEE](https://img.shields.io/badge/IEEE-Paper%20Implementation-00629B?style=for-the-badge&logo=ieee&logoColor=white)
![Status](https://img.shields.io/badge/Status-Phase%201%20%7C%20Structure-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade implementation of the IEEE phishing-URL detection research,  
featuring multi-layer feature engineering, ensemble ML classifiers, and a real-time Flask API.**

</div>

---

## Table of Contents

1. [Overview](#overview)
2. [Paper Reference](#paper-reference)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Feature Engineering Roadmap](#feature-engineering-roadmap)
6. [Technology Stack](#technology-stack)
7. [Getting Started](#getting-started)
8. [Development Phases](#development-phases)
9. [API Reference](#api-reference)
10. [Contributing](#contributing)
11. [License](#license)

---

## Overview

This project implements the phishing-website detection methodology described in the IEEE research paper.  
The system analyses URLs and web-page content across **three feature domains** — URL-level, domain-level,  
and HTML/JS-level — and feeds them into an ensemble classifier to distinguish legitimate sites from  
phishing attempts with high precision and recall.

The implementation is structured for **production readiness**:

- Clean separation of feature extraction, model training, and inference.
- A Flask REST API that serves real-time predictions.
- Notebook-driven experimentation with reproducible pipelines.
- Comprehensive logging, testing, and code-quality tooling.

---

## Paper Reference

> *"Detection of Phishing Websites Using Machine Learning"*  
> Published in: **IEEE Transactions / IEEE Conference Proceedings**  
> DOI: *(to be added upon final paper confirmation)*

This implementation faithfully reproduces the paper's feature set and extends it with  
additional engineering choices suited for deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client (Browser / API Consumer)          │
└───────────────────────────────┬─────────────────────────────────┘
                                │  HTTP/REST
┌───────────────────────────────▼─────────────────────────────────┐
│                       Flask REST API  (backend/app.py)          │
│                                                                 │
│   /api/v1/predict   ──►  Feature Extraction  ──►  ML Model     │
│   /api/v1/explain   ──►  SHAP / LIME Explainer                 │
│   /api/v1/report    ──►  Report Generator                      │
└──────────────┬────────────────────────────────────┬────────────┘
               │                                    │
   ┌───────────▼──────────┐            ┌────────────▼────────────┐
   │   utils/             │            │   models/               │
   │  ├── url_features.py │            │  ├── classifier.py      │
   │  ├── html_features.py│            │  ├── feature_selector.py│
   │  ├── dns_features.py │            │  └── saved/  (.joblib)  │
   │  └── preprocessor.py │            └─────────────────────────┘
   └──────────────────────┘
```

---

## Project Structure

```
phishing_detection_ieee/
│
├── backend/                    # Flask web application
│   ├── app.py                  # Application factory & entry point
│   ├── templates/              # Jinja2 HTML templates (UI)
│   └── static/
│       ├── css/                # Stylesheets
│       ├── js/                 # Frontend scripts
│       └── images/             # Static assets
│
├── data/                       # Dataset storage (git-ignored)
│   ├── raw/                    # Original, unmodified datasets
│   ├── processed/              # Cleaned & feature-engineered CSVs
│   └── external/               # Third-party reference lists (Alexa, PhishTank)
│
├── models/                     # ML model artefacts
│   ├── saved/                  # Serialised trained models (.joblib)
│   └── checkpoints/            # Intermediate training checkpoints
│
├── utils/                      # Feature extraction & preprocessing
│   ├── url_features.py         # URL-lexical feature extractor
│   ├── html_features.py        # HTML/JS content feature extractor
│   ├── dns_features.py         # DNS / WHOIS feature extractor
│   └── preprocessor.py         # Data cleaning & transformation pipeline
│
├── notebooks/                  # Jupyter experimentation notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation_and_explainability.ipynb
│
├── reports/                    # Generated analysis reports
│   ├── figures/                # Saved plots & charts
│   └── outputs/                # Classification reports, metrics
│
├── requirements.txt            # Python dependencies (pinned versions)
├── README.md                   # This document
└── .gitignore                  # Git exclusion rules
```

---

## Feature Engineering Roadmap

The IEEE paper defines features across three layers.  All features listed below  
will be implemented in the `utils/` package during **Phase 2**.

### Layer 1 — URL-Lexical Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | `url_length` | Total character count of the raw URL |
| 2 | `has_ip_address` | IP address used instead of domain name |
| 3 | `at_symbol` | Presence of `@` in the URL |
| 4 | `double_slash_redirect` | `//` appearing beyond position 7 |
| 5 | `prefix_suffix` | Hyphen `-` in domain name |
| 6 | `subdomain_count` | Number of subdomains |
| 7 | `https` | HTTPS protocol indicator |
| 8 | `domain_registration_length` | Registration period (WHOIS) |
| 9 | `favicon` | Favicon loaded from external domain |
| 10 | `port` | Non-standard port in URL |
| 11 | `https_token` | `https` token appearing in domain part |

### Layer 2 — HTML / JavaScript Features

| # | Feature | Description |
|---|---------|-------------|
| 12 | `request_url` | Percentage of objects loaded from external domains |
| 13 | `url_of_anchor` | Percentage of `<a>` tags pointing externally |
| 14 | `links_in_meta_script_link` | Links in `<meta>`, `<script>`, `<link>` tags |
| 15 | `sfh` | Server Form Handler destination |
| 16 | `submitting_to_email` | Form submits to email address |
| 17 | `abnormal_url` | Host name not present in raw URL |
| 18 | `redirect_count` | Number of HTTP redirects |
| 19 | `on_mouseover` | `onmouseover` event changes status bar |
| 20 | `right_click_disabled` | Right-click disabled via JS |
| 21 | `popup_window` | Pop-up window with text field |
| 22 | `iframe` | Use of invisible `<iframe>` |

### Layer 3 — Domain / External-Service Features

| # | Feature | Description |
|---|---------|-------------|
| 23 | `age_of_domain` | Domain age in months (WHOIS) |
| 24 | `dns_record` | DNS record present |
| 25 | `web_traffic` | Alexa rank (log-normalised) |
| 26 | `page_rank` | Google PageRank indicator |
| 27 | `google_index` | URL indexed by Google |
| 28 | `links_pointing_to_page` | Number of external back-links |
| 29 | `statistical_report` | URL / host appears in PhishTank / StopBadware |

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.10+ | Core implementation |
| Web API | Flask 3 + Flask-CORS | REST endpoint serving |
| Data | Pandas, NumPy | Tabular data manipulation |
| Feature Engineering | tldextract, BeautifulSoup4, dnspython, python-whois | URL / HTML / DNS parsing |
| ML (Phase 3) | scikit-learn, XGBoost, LightGBM | Classification models |
| Deep Learning (Phase 4) | PyTorch, Transformers | BERT-based URL encoding |
| Explainability | SHAP, LIME | Model interpretability |
| Visualisation | Matplotlib, Seaborn, Plotly | EDA & evaluation charts |
| Notebooks | Jupyter | Experimentation |
| Testing | pytest, pytest-cov | Unit & integration tests |
| Code Quality | Black, Flake8, isort, mypy | Style & type checking |
| Logging | Loguru | Structured application logs |

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- (Optional) CUDA-enabled GPU for deep-learning phases

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/phishing_detection_ieee.git
cd phishing_detection_ieee

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the Flask development server
python backend/app.py
```

Open [http://localhost:5000/health](http://localhost:5000/health) to verify the service is running.

### Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```env
FLASK_ENV=development
PORT=5000
```

---

## Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | Project structure, dependencies, documentation |
| **Phase 2** | 🔲 Planned | Feature extraction (`utils/`) — all 29 IEEE features |
| **Phase 3** | 🔲 Planned | Model training, evaluation, serialisation (`models/`) |
| **Phase 4** | 🔲 Planned | Flask API endpoints for real-time prediction |
| **Phase 5** | 🔲 Planned | Frontend dashboard (HTML/CSS/JS) |
| **Phase 6** | 🔲 Planned | SHAP / LIME explainability layer |
| **Phase 7** | 🔲 Planned | Report generation & CI/CD pipeline |

---

## API Reference

> API endpoints will be documented here as each phase is completed.

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "phishing-detection-ieee"
}
```

### Predict *(Phase 4)*

```
POST /api/v1/predict
Content-Type: application/json

{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "url": "https://example.com",
  "prediction": "legitimate",
  "confidence": 0.97,
  "features": { ... }
}
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/url-feature-extractor`
3. Commit changes: `git commit -m "feat: add URL lexical feature extractor"`
4. Push to branch: `git push origin feature/url-feature-extractor`
5. Open a Pull Request

Please follow [Conventional Commits](https://www.conventionalcommits.org/) for commit messages  
and ensure all tests pass (`pytest --cov`) before submitting.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ as an academic research implementation of IEEE phishing detection.

</div>

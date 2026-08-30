# Error Analysis Report

**Total Test Samples:** 35185
**Total Errors:** 1
**False Positives (Phishing predicted as Legitimate):** 0
**False Negatives (Legitimate predicted as Phishing):** 1

---

## Error 1: False Negative

| Property | Value |
|----------|-------|
| Sample Index | 10405 |
| Actual Class | Legitimate |
| Predicted Class | Phishing |
| P(Legitimate) | 0.226126 |
| Confidence | 77.38999938964844% |

### Top 5 Contributing Features

| Feature | Value | Contribution |
|---------|-------|--------------|
| NoOfSelfRef | 7.0000 | 0.772960 |
| LineOfCode | 222.0000 | 0.771552 |
| NoOfExternalRef | 3.0000 | 0.771115 |
| NoOfJS | 1.0000 | 0.769414 |
| NoOfImage | 7.0000 | 0.768284 |

### Full Top20 Feature Vector

| Feature | Value |
|---------|-------|
| URLSimilarityIndex | 100.0000 |
| NoOfExternalRef | 3.0000 |
| LineOfCode | 222.0000 |
| NoOfSelfRef | 7.0000 |
| IsHTTPS | 1.0000 |
| NoOfImage | 7.0000 |
| NoOfJS | 1.0000 |
| HasSocialNet | 0.0000 |
| NoOfCSS | 1.0000 |
| HasCopyrightInfo | 0.0000 |
| NoOfOtherSpecialCharsInURL | 2.0000 |
| LargestLineLength | 417.0000 |
| HasDescription | 0.0000 |
| NoOfDegitsInURL | 0.0000 |
| URLLength | 22.0000 |
| IsResponsive | 1.0000 |
| DegitRatioInURL | 0.0000 |
| DomainTitleMatchScore | 100.0000 |
| SpacialCharRatioInURL | 0.0910 |
| HasSubmitButton | 0.0000 |

**Possible Explanation:** This legitimate sample has unusual feature values that resemble phishing patterns (e.g., low URLSimilarityIndex, high digit ratio, or few self-references), causing the model to misclassify it as phishing.

---


# 🔒 Data Governance & Security

The **Home Credit Scoring Pipeline** is designed with a "Privacy-First" architecture, ensuring that sensitive applicant data is protected throughout the modeling lifecycle.

## PII Hashing (SHA-256)

To meet **SOC2** and **GDPR** "Privacy by Design" standards, all Primary Identifiers (`sk_id_curr`) are anonymized immediately upon ingestion.

### Implementation

- **Algorithm**: SHA-256 (Secure Hash Algorithm 2)
- **Format**: Hex-encoded uppercase string
- **Location**: `src/ingest_credit.py`

```sql
-- Example SQL logic used during ingestion
SELECT 
    upper(hex(sha256(sk_id_curr::VARCHAR))) AS sk_id_curr,
    ...
FROM raw_application_train
```

### Why SHAs?

By hashing identifiers at the source, we ensure that:

1. **The Modelling Layer is Anonymized**: Data scientists and ML models never see the raw internal IDs, preventing accidental joins with external PII databases.
2. **Consitency**: The same `sk_id_curr` always produces the same hash, allowing for multi-table joins across the pipeline without ever exposing the raw ID.

> [!IMPORTANT]
> **Production Note on Salting**: This repository demonstrates a standard hashing implementation. For a client's live production environment, we recommend adding a **Cryptographic Salt** before hashing (e.g., `sha256(sk_id_curr || 'secret_salt')`). This prevents "Rainbow Table" attacks where an adversary could pre-calculate hashes for known IDs.

## Access Control

This pipeline is designed to be deployed within a secured, VPC-isolated environment (AWS, GCP, or Azure).

- **Storage**: DuckDB files are local-first, meaning they stay within your compute instance and are not exposed to the public internet.
- **Serving**: The Streamlit dashboard includes a "Demo Mode" to prevent raw data exposure during stakeholder previews.

---
**Data Security Lead**: Kshitij Bhatt
**Project Repo**: [Home-Credit-Scoring-Pipeline](https://github.com/Kshitijbhatt1998/Home-Credit-Scoring-Pipeline)

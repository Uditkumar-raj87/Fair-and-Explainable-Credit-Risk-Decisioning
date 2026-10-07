# Feature Policy

## Educational scope

This project is a non-production demonstration. It must not be used to approve,
deny, price, or otherwise make lending decisions. Use synthetic or public data only;
do not add private customer data, protected-class proxies, or credentials.

## Feature rules

- Document source, collection date, owner, missingness, and intended use for every feature.
- Exclude direct identifiers and post-outcome or leakage variables.
- Review age, geography, and other sensitive attributes with legal and fairness experts.
- Keep protected attributes only in a controlled evaluation slice when needed for auditing.
- Reproduce preprocessing from versioned code and record the exact data snapshot.
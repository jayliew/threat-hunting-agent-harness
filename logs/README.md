# ECS lookalike evidence packages

These three synthetic, selected evidence excerpts are intended to test false positives and evidence-based uncertainty. They are not full captures, authentic vendor exports, or proof that the surrounding environment is safe. All public IPs are documentation addresses; example.net names and organization-specific products are fictional. Filenames and these instructor notes are not sent to the model by the harness.

## Representation and collection semantics

Each `.ecs.jsonl` line is one JSON event document using nested [Elastic Common Schema 8.17](https://www.elastic.co/guide/en/ecs/8.17/index.html) fields. These are document bodies, not Elasticsearch Bulk API action/document pairs; use a JSON/NDJSON ingestion pipeline and ECS-compatible mappings for Elasticsearch. No cluster or index setup is included.

- `@timestamp`: UTC ISO 8601 occurrence time with milliseconds. For firewall flow summaries it is the flow end time; `event.start`, `event.end`, and `event.duration` supply the interval. Durations are integer **nanoseconds**.
- `event.id`: short string in the form `c01-e001`, with a neutral case prefix and an event number. IDs are unique across the six files; prefixes do not encode the expected verdict. Keep existing IDs stable when reordering or extending a case. `event.created`: later collector read time. `event.category` and `event.type` are arrays. `event.outcome` describes the operation, not whether the activity is malicious.
- Scan connection outcomes (`internal-network-scan.jsonl` and `scheduled-discovery.ecs.jsonl`) describe transport connection establishment or a UDP reply: `established`/`replied` → `success`; initial-SYN rejection (`reset`), `timeout`, or `incomplete` handshake → `failure`. Firewall permission is recorded separately by `event.action: allow` or `event.type: allowed`. A SYN/ACK without completion is not an established connection; a reset after a completed handshake does not undo successful establishment. A failed connection does not imply a failed scan job or malicious activity.
- `event.dataset` identifies the simulated source; `observer` identifies the observing appliance or management system. `host` identifies the endpoint or server on which the recorded event occurred.
- `http.request.bytes` and `http.response.bytes` count application headers plus body, excluding TCP/TLS overhead. A `204` has zero response-body bytes. The proxy records decrypted HTTPS transactions and explicitly records inspection under `corp.proxy.tls_inspected`.
- Firewall `source.bytes` and `destination.bytes` count IP packet bytes, including IP/TCP headers, from initiator to responder and back. `network.bytes`/`network.packets` are directional totals. Endpoint connection-start records do not claim complete byte counts.
- Custom, non-ECS fields live exclusively under `corp`. The `corp.vpn` and `corp.auth` objects contain identity/session/device-binding observations; `corp.deployment` contains applied package configuration; `corp.endpoint` contains a deployment reference; `corp.proxy` contains TLS inspection status; `corp.change`, `corp.asset`, and `corp.scan` contain approval, asset, and job records; `corp.flow` contains normalized state and observed TCP flag unions. These require custom mappings if indexed. The executable SHA-256 is a synthetic fixture value, not a real software reputation indicator.

## Evidence ID prefixes

| Case | Prefix |
| --- | --- |
| `http-beaconing.jsonl` | `c01` |
| `internal-network-scan.jsonl` | `c02` |
| `managed-telemetry.ecs.jsonl` | `c03` |
| `password-spray.jsonl` | `c04` |
| `scheduled-discovery.ecs.jsonl` | `c05` |
| `shared-vpn-logins.ecs.jsonl` | `c06` |

Cite IDs verbatim, for example `c01-e006`. Historical reports retain the IDs in their saved input; rerun the updated fixtures when comparing current evidence citations.

## Shared VPN logins: `shared-vpn-logins.ecs.jsonl`

30 events: six device-bound VPN tunnel establishments, six password failures, six successful retries, six FIDO2 validations, and six application-session issuances. Six users appear behind `198.51.100.7`; first failures span 151.4 seconds. Each retries from the same device/session after 8.9–26.1 seconds.

Expected assessment: **benign**, threat type **none**, for the supplied excerpt. Many accounts sharing a source address resembles spraying. The explanation requires VPN egress mappings, distinct verified devices and tunnel sessions, per-user retry sequences, and registered FIDO2 verification before application sessions are issued. A source address alone must not be treated as a single actor. Successful password validation alone is not a fully authenticated session.

Useful evidence includes VPN records plus representative password/MFA/session sequences. A model should acknowledge that shared egress and MFA do not categorically exclude abuse. Removing the corroborating identity records should reduce confidence; it does not necessarily force a suspicious verdict.

## Managed telemetry: `managed-telemetry.ecs.jsonl`

28 events: an applied deployment policy, service process start, twelve endpoint connections, twelve proxy transactions, and two unrelated browsing transactions. The check-ins go to the literal IP `203.0.113.77:443` using `/v1/status`, about every five minutes, with small requests and `204` responses.

Expected assessment: **benign**, threat type **none**, for the supplied excerpt. The model must connect the configuration system's executable path/hash, endpoint URL and interval to the service process, then join endpoint connections to proxy transactions using source/destination IPs, ports and timing. All twelve endpoint connections have the same process entity ID as the recorded service start. The user-agent or regular timing alone is insufficient evidence of legitimacy or C2.

Useful evidence includes the deployment, service start, and multiple endpoint/proxy pairs. The records support the configured telemetry explanation; they do not prove a trusted process could never be compromised. Removing deployment/process evidence leaves potentially suspicious or inconclusive periodic HTTPS activity.

## Scheduled discovery: `scheduled-discovery.ecs.jsonl`

39 events: approved change, asset inventory snapshot, scan-job start, 32 scan flows, three unrelated flows, and scan-job completion. `10.47.12.66` contacts eight destinations (`10.47.20.20`–`10.47.20.27`) on ports 22, 445, 3389 and 5985. Four connects succeed, 20 receive resets, and eight time out. Successful probes model a TCP connect followed by reset; closed-port and timeout packet counts are separate.

Expected assessment: **benign**, threat type **none**, with the summary explicitly recognizing **authorized scanning**. The approval's source, exact target list, ports, execution account and time window must match the job and observed traffic. Approval status or a scanner-like hostname alone is not sufficient. No network flow is pre-labeled with a change ID: correlation is required. Successful TCP connects do not establish successful authentication, exploitation, or lateral movement.

Useful evidence includes approval, job start, representative flows across destinations/ports and job completion. Removing approval/execution context should leave scanning visible with authorization unknown. Changing the source, scope, account or time outside the approved window should change the assessment.

## Evaluation cautions

See the [private answer keys for all six cases](../evals/answer-keys.md) for the three-check manual grading rubric and current evidence IDs.

Keep instructor expectations out of the model prompt. Score supporting and conflicting claims, evidence relevance, and uncertainty separately from output-format validity; do not require every event to be cited.

ECS is the target representation for the entire suite. These additions are contextual lookalikes, not strictly matched counterfactual pairs. Normalized copies with context removed can test confidence changes, but label those variants according to the evidence actually retained. Event IDs and usernames can be varied without changing the intended interpretation.

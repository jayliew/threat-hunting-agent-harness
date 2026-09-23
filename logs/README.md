# ECS evidence packages

These six synthetic, selected evidence excerpts test suspicious-pattern recognition, false positives and evidence-based uncertainty. Both classes include collection metadata and contextual evidence; context must be correlated with the activity rather than treated as a benign label. They are not full captures, authentic vendor exports, or proof that the surrounding environment is safe. All public IPs are documentation addresses; example.net names and organization-specific products are fictional. Filenames and these instructor notes are not sent to the model by the harness.

## Representation and collection semantics

Each `.jsonl` line is one JSON event document using nested [Elastic Common Schema 8.17](https://www.elastic.co/guide/en/ecs/8.17/index.html) fields. These are document bodies, not Elasticsearch Bulk API action/document pairs; use a JSON/NDJSON ingestion pipeline and ECS-compatible mappings for Elasticsearch. No cluster or index setup is included.

- `@timestamp`: UTC ISO 8601 occurrence time with milliseconds. For firewall flow summaries it is the flow end time; `event.start`, `event.end`, and `event.duration` supply the interval. Durations are integer **nanoseconds**.
- `event.id`: short string in the form `c01-e001`, with a neutral case prefix and an event number. IDs are unique across the six files; prefixes do not encode the expected verdict. Keep existing IDs stable when reordering or extending a case. `event.created`: later collector read time. `event.category` and `event.type` are arrays. `event.outcome` describes the operation, not whether the activity is malicious.
- Scan connection outcomes (`internal-network-scan.jsonl` and `scheduled-discovery.jsonl`) describe transport connection establishment or a UDP reply: `established`/`replied` → `success`; initial-SYN rejection (`reset`), `timeout`, or `incomplete` handshake → `failure`. Firewall permission is recorded separately by `event.action: allow` or `event.type: allowed`. A SYN/ACK without completion is not an established connection; a reset after a completed handshake does not undo successful establishment. A failed connection does not imply a failed scan job or malicious activity.
- `event.dataset` identifies the simulated source; `observer` identifies the observing appliance or management system. `host` identifies the endpoint or server on which the recorded event occurred.
- `http.request.bytes` and `http.response.bytes` count application headers plus body, excluding TCP/TLS overhead. A `204` has zero response-body bytes. The proxy records decrypted HTTPS transactions and explicitly records inspection under `corp.proxy.tls_inspected`.
- Firewall `source.bytes` and `destination.bytes` count IP packet bytes, including IP/TCP headers, from initiator to responder and back. `network.bytes`/`network.packets` are directional totals. Endpoint connection-start records do not claim complete byte counts.
- Custom, non-ECS fields live exclusively under `corp`. The `corp.vpn` and `corp.auth` objects contain identity/session/device-binding observations; `corp.deployment` contains applied package configuration; `corp.endpoint` contains a deployment reference; `corp.proxy` contains TLS inspection status; `corp.change`, `corp.asset`, and `corp.scan` contain approval, asset, and job records; `corp.flow` contains normalized state and observed TCP flag unions. These require custom mappings if indexed. Executable SHA-256 values are synthetic fixture values, not real software reputation indicators.

## Evidence ID prefixes

| Case | Prefix |
| --- | --- |
| `http-beaconing.jsonl` | `c01` |
| `internal-network-scan.jsonl` | `c02` |
| `managed-telemetry.jsonl` | `c03` |
| `password-spray.jsonl` | `c04` |
| `scheduled-discovery.jsonl` | `c05` |
| `shared-vpn-logins.jsonl` | `c06` |

Cite IDs verbatim, for example `c01-e006`. Historical reports retain the IDs in their saved input; rerun the updated fixtures when comparing current evidence citations.

## HTTP beaconing: `http-beaconing.jsonl`

60 events: the original 43 traffic observations plus an applied deployment, two process starts, twelve endpoint connections for the recurring requests, and one endpoint/proxy pair for the deployed service. Existing event IDs, traffic timestamps and recurring-request pattern are preserved. HTTPS observations now use the same proxy schema as managed telemetry; former directional byte counts are explicitly HTTP application bytes.

Expected assessment: **suspicious**, suspected beaconing. Deployment `c01-e044` and service `c01-e045` explain `/v1/status`, including the observed pair `c01-e059`/`c01-e060`. The twelve `/api/heartbeat` requests instead correlate to process `c01-e046` through connections `c01-e047`–`c01-e058`. It has the same basename but a different executable path/hash, user and parent. Neither deployment presence nor a familiar process name establishes legitimacy; these differences also do not prove malware.

## Internal network scan: `internal-network-scan.jsonl`

53 events: the original 49 flows plus approval, inventory, job-start and job-completion records. Original probe IDs, timestamps, scope and connection outcomes are retained. TCP observations are normalized under `corp.flow`; flow intervals use `event.start`/`event.end`/`event.duration`.

Expected assessment: **suspicious**, scanning outside the supplied approval window. Approval `c02-e050` covers the same source, account, targets and ports at 15:30–15:40 UTC. Job `c02-e052` and the 32 probes occur around 14:30 UTC. Inventory and a job's change reference do not override the actual approval window. A scheduling error or another approval is possible; neither compromise nor malicious intent is proven.

## Password spray: `password-spray.jsonl`

26 events: fourteen original password checks, one authenticated VPN tunnel, five background FIDO2/session pairs, and a FIDO2 challenge after Bob's password success. Original password-check IDs, times, users, source addresses and outcomes are preserved. Authentication fields use the same dataset and semantics as shared VPN logins.

Expected assessment: **suspicious**, a possible multi-account credential attack. The six target-account sequences reference one device/tunnel authenticated as Morgan (`c04-e015`), unlike the six independently authenticated devices in the benign case. VPN presence alone is insufficient to dismiss the pattern. Background MFA/session pairs are `c04-e016`–`c04-e025`; Bob's challenge is `c04-e026`, with no completion supplied. In authentication events `user` is the account being checked; in a VPN event it is the tunnel's authenticated identity. Device binding is independently observed and does not assert that the target user owns that device.

## Shared VPN logins: `shared-vpn-logins.jsonl`

30 events: six device-bound VPN tunnel establishments, six password failures, six successful retries, six FIDO2 validations, and six application-session issuances. Six users appear behind `198.51.100.7`; first failures span 151.4 seconds. Each retries from the same device/session after 8.9–26.1 seconds.

Expected assessment: **benign**, threat type **none**, for the supplied excerpt. Many accounts sharing a source address resembles spraying. The explanation requires VPN egress mappings, distinct verified devices and tunnel sessions, per-user retry sequences, and registered FIDO2 verification before application sessions are issued. A source address alone must not be treated as a single actor. Successful password validation alone is not a fully authenticated session.

Useful evidence includes VPN records plus representative password/MFA/session sequences. A model should acknowledge that shared egress and MFA do not categorically exclude abuse. Removing the corroborating identity records should reduce confidence; it does not necessarily force a suspicious verdict.

## Managed telemetry: `managed-telemetry.jsonl`

28 events: an applied deployment policy, service process start, twelve endpoint connections, twelve proxy transactions, and two unrelated browsing transactions. The check-ins go to the literal IP `203.0.113.77:443` using `/v1/status`, about every five minutes, with small requests and `204` responses.

Expected assessment: **benign**, threat type **none**, for the supplied excerpt. The model must connect the configuration system's executable path/hash, endpoint URL and interval to the service process, then join endpoint connections to proxy transactions using source/destination IPs, ports and timing. All twelve endpoint connections have the same process entity ID as the recorded service start. The user-agent or regular timing alone is insufficient evidence of legitimacy or C2.

Useful evidence includes the deployment, service start, and multiple endpoint/proxy pairs. The records support the configured telemetry explanation; they do not prove a trusted process could never be compromised. Removing deployment/process evidence leaves potentially suspicious or inconclusive periodic HTTPS activity.

## Scheduled discovery: `scheduled-discovery.jsonl`

39 events: approved change, asset inventory snapshot, scan-job start, 32 scan flows, three unrelated flows, and scan-job completion. `10.47.12.66` contacts eight destinations (`10.47.20.20`–`10.47.20.27`) on ports 22, 445, 3389 and 5985. Four connects succeed, 20 receive resets, and eight time out. Successful probes model a TCP connect followed by reset; closed-port and timeout packet counts are separate.

Expected assessment: **benign**, threat type **none**, with the summary explicitly recognizing **authorized scanning**. The approval's source, exact target list, ports, execution account and time window must match the job and observed traffic. Approval status or a scanner-like hostname alone is not sufficient. No network flow is pre-labeled with a change ID: correlation is required. Successful TCP connects do not establish successful authentication, exploitation, or lateral movement.

Useful evidence includes approval, job start, representative flows across destinations/ports and job completion. Removing approval/execution context should leave scanning visible with authorization unknown. Changing the source, scope, account or time outside the approved window should change the assessment.

## Evaluation cautions

See the [private answer keys for all six cases](../evals/answer-keys.md) for the three-check manual grading rubric and current evidence IDs.

Keep instructor expectations out of the model prompt. Score supporting and conflicting claims, evidence relevance, and uncertainty separately from output-format validity; do not require every event to be cited.

All six files now contain `event.created`, `event.dataset` and `corp` context. Both suspicious and benign cases contain operational context; the distinction depends on its relationship to the observed activity. This removes those field-presence shortcuts but does not make a six-case synthetic suite free of all presentation artifacts. ECS is the target representation for the entire suite. These additions are contextual lookalikes, not strictly matched counterfactual pairs. Normalized copies with context removed can test confidence changes, but label those variants according to the evidence actually retained. Event IDs and usernames can be varied without changing the intended interpretation.

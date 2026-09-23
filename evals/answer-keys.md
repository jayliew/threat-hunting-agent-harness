# MVP answer keys — human grading only

These keys cover the five current ECS evidence packages. Keep this file out of model prompts. The harness sends the selected JSONL events; it does not load these keys. Grade only what the supplied events justify, not the author's hidden scenario intent.

## How to grade

For each response, mark three checks **pass** or **fail**, with a short reason:

| Check | Pass requirement |
| --- | --- |
| Interpretation | A verdict consistent with the supplied evidence, and the central pattern or benign explanation identified. |
| Evidence | The explanation connects the decisive observations and cites events that support them. A valid ID alone is insufficient. |
| Restraint | No invented facts or material claims stronger than the evidence, such as proven compromise from a login success. |

Record output-format validity separately using the harness. An analytical case passes when all three checks pass. If the answer is invalid or truncated, record that execution/format failure; do not silently omit it from results. Model names can be hidden during grading to reduce bias.

Expected verdicts below are the preferred assessments under the harness's definitions: `suspicious` means potentially malicious, not proven malicious; `benign` applies to the observed activity, not the entire environment. None of these five is designed as an inconclusive case. Review a well-supported alternative manually and record your reason; do not accept generic uncertainty as a substitute for analyzing the available context. Apply the same judgment to every model.

Exact wording is not required. Threat-type synonyms are acceptable. Counts and precise timing below are reference facts, not requirements to recite every number. Approximate timing is fine. A short paragraph can pass. Representative citations are sufficient when they support the stated relationships; models need not cite every listed event or reproduce an exact evidence set. Distinguish a benign alternative offered as a hypothesis from a benign explanation actually supported by records.

Suggested worksheet:

| Model/configuration | Case | Format valid | Interpretation | Evidence | Restraint | Failure/review note |
| --- | --- | --- | --- | --- | --- | --- |

Run each case as a fresh conversation. To select all five from the repository directory, use `uv run python compare_models.py --logs logs/*.jsonl`. The default list may cover fewer cases. All records in a file belong to that case's independent scenario; do not join reused addresses across files.

## c01 — HTTP beaconing

**Input:** [http-beaconing.jsonl](../logs/http-beaconing.jsonl) — 60 events.

**Expected verdict:** `suspicious`. **Threat type:** suspected HTTP/HTTPS beaconing or possible C2 beaconing.

**Must notice:** `10.47.12.88` repeatedly contacts `203.0.113.77:443` with near-five-minute timing and small, stable byte counts. The combination supports automated beacon-like behavior distinct from the surrounding traffic. Calling it merely "HTTPS traffic" misses the pattern. The applied deployment does not explain these requests: its service executable/hash and `/v1/status` URL differ from the observed interactive process and `/api/heartbeat` requests.

**Reference facts:** Twelve `GET /api/heartbeat` requests, all HTTP 204, with the same user-agent. Intervals average about 300.4 seconds and range from 287.3 to 314.3 seconds. Application request counts are 187–218 bytes and response counts 65–92 bytes (headers plus body, excluding transport/TLS overhead).

**Supporting evidence:** `c01-e006`, `c01-e011`, `c01-e016`, `c01-e020`, `c01-e023`, `c01-e024`, `c01-e027`, `c01-e031`, `c01-e035`, `c01-e038`, `c01-e041`, `c01-e043`. For example, several early and later requests support recurrence; a single request does not establish periodicity. Deployment `c01-e044` and service start `c01-e045` identify the approved service. Process start `c01-e046` records a different path/hash, user and parent despite the same executable basename. Endpoint connections `c01-e047` through `c01-e058` link this process to the twelve requests by source/destination tuples and timing; cite representative pairs, not an ID range in model output. `c01-e059` and `c01-e060` corroborate one actual check-in by the approved service. A passing explanation must distinguish that service from the process responsible for the recurring requests. Slack traffic is additional background.

**Appropriate uncertainty:** Regular requests can also come from legitimate software. The responsible process is recorded, but its differing path/hash and interactive launch do not prove malware. The supplied deployment does not cover its behavior; another legitimate explanation remains possible. Destination reputation is not established.

**Must not claim:** Confirmed malware/C2, a named threat actor or malware family, proven exfiltration, or that the destination is known malicious. Documentation IP space is not malicious reputation. Do not interpret HTTP application bytes as complete TLS wire traffic or treat the approved service's destination as blanket authorization for other processes.

## c02 — Internal network scanning

**Input:** [internal-network-scan.jsonl](../logs/internal-network-scan.jsonl) — 53 events.

**Expected verdict:** `suspicious`. **Threat type:** internal network/port scanning or network-service discovery.

**Must notice:** `10.47.12.66` systematically probes several ports across multiple internal hosts in a short burst. This is scanning behavior. The supplied approval covers the same source, targets and ports at 15:30–15:40 UTC, but the job and probes occur around 14:30 UTC. Isolated failed connections from other hosts are not enough to establish another scan.

**Reference facts:** 32 probes to eight hosts (`10.47.20.20`–`10.47.20.27`) on 22, 445, 3389 and 5985, spanning about 22 seconds. Twenty are initially rejected, eight time out, and four receive SYN/ACK without completing the handshake. These probes all have connection outcome `failure`. Approval `c02-e050`, asset `c02-e051`, job start `c02-e052` and completion `c02-e053` provide context comparable to the benign discovery case. The recorded `discovery-agent` job runs under the approved account and references the change, but starts outside its window. A probe can discover a listening port even though a connection was not established.

**Representative evidence:** `c02-e006`, `c02-e007`, `c02-e009`, `c02-e010` show four ports on one host; `c02-e011`, `c02-e019`, `c02-e021`, `c02-e027`, `c02-e040`, `c02-e044` show expansion across hosts and outcomes. The isolated reset/retry at `c02-e045`, `c02-e046` is background activity. Other representative scanning events are acceptable. Cite the approval and job/flow timing to establish the window mismatch; merely observing that an approval exists is insufficient.

**Appropriate uncertainty:** An administrative scheduling error or another authorization is possible, but this supplied approval does not cover the observed execution time. A good answer can be confident about scanning while uncertain about intent.

**Must not claim:** Successful lateral movement, exploitation, authentication, all ports closed, proven workstation compromise, or a tool absent from the records. The job explicitly records `discovery-agent`; do not infer nmap or a malware family. An `allow` firewall action is not a successful TCP connection. SYN/ACK alone does not establish a completed handshake.

## c03 — Managed telemetry

**Input:** [managed-telemetry.jsonl](../logs/managed-telemetry.jsonl) — 28 events.

**Expected verdict:** `benign`. **Threat type:** `none`; describe managed telemetry in the summary.

**Must notice:** The periodic HTTPS traffic matches an applied deployment configuration and the observed service process. Explain the link between the configured executable/destination/interval, the endpoint connections and the proxy requests. A familiar-looking user-agent or process name alone is insufficient.

**Reference facts:** Policy specifies `https://203.0.113.77/v1/status`, a 300-second period and 15-second jitter. Twelve endpoint connections belong to the same process entity, and twelve proxy transactions use the matching source/destination tuples with near-adjacent times. All twelve return HTTP 204. Two unrelated browsing transactions are also present.

**Supporting evidence:** `c03-e001` supplies the deployment configuration; `c03-e002` supplies the matching executable/hash and process identity. `c03-e003`, `c03-e004` and `c03-e005`, `c03-e006` are endpoint/proxy pairs. `c03-e027`, `c03-e028` show the last pair. Cite deployment/process evidence and representative traffic; citing the deployment alone does not verify that observed traffic matches it.

**Appropriate uncertainty:** The observed activity is consistent with the managed service. Configuration and process evidence do not prove the endpoint can never be compromised.

**Must not claim:** C2 solely because traffic is regular or uses a literal IP; a real-world reputation lookup for the synthetic hash; verified code signing (not recorded); or that all traffic from SYSTEM is safe.

## c04 — Password spray pattern

**Input:** [password-spray.jsonl](../logs/password-spray.jsonl) — 26 events.

**Expected verdict:** `suspicious`. **Threat type:** suspected password spraying or an automated multi-account credential attack. A cautious broader label is acceptable if the many-account pattern is recognized.

**Must notice:** One source (`198.51.100.7`) fails against six different users, then authenticates successfully as Bob. Distinguish this from isolated same-user failure/retry pairs. VPN context is present, but all six target-account sequences bind to one device and tunnel authenticated as Morgan.

**Reference facts:** One failure each for Alice, Bob, Carol, Dave, Erin and Frank, 30 seconds apart over 150 seconds. Bob's subsequent success occurs 90 seconds after the last failure, from the same source. This pattern differs from high-volume guessing against a single account. `c04-e015` records the Morgan tunnel (`vpn-session-810`, `endpoint-130`) and shared egress. Five background successes have corresponding FIDO2/session records (`c04-e016` through `c04-e025`). Bob receives a FIDO2 challenge (`c04-e026`); no completed MFA or application-session issuance for that sequence is supplied.

**Supporting evidence:** `c04-e005`, `c04-e006`, `c04-e007`, `c04-e008`, `c04-e009`, `c04-e010` are the failures; `c04-e011` is Bob's success. Representative failures across multiple users plus the success establish the credential pattern. Use `c04-e015` with representative password events to show that their device/tunnel is shared across target accounts. `c04-e003`, `c04-e004` and `c04-e013`, `c04-e014` are isolated internal failure/retry pairs. A valid tunnel and unrelated MFA successes do not explain away the multi-account pattern.

**Appropriate uncertainty:** Attempted passwords and credential provenance are absent. The events cannot prove password reuse, conclusively distinguish spraying from credential stuffing, or establish who performed the successful password check. A tunnel authenticated as Morgan does not prove Morgan personally initiated the attempts. Absence of a recorded MFA completion does not prove it failed or never occurred.

**Must not claim:** The same password was tried against every account, confirmed account takeover, MFA bypass, completed Bob application session, known malicious source reputation, or that any isolated failed login is an attack.

## c05 — Scheduled discovery

**Input:** [scheduled-discovery.jsonl](../logs/scheduled-discovery.jsonl) — 39 events.

**Expected verdict:** `benign`. **Threat type:** `none`; explicitly recognize authorized scanning in the summary.

**Must notice:** The approval's source, target hosts, ports and time window match the observed scan; execution records link the approved account and change to the job. Recognizing the approval word alone is insufficient.

**Reference facts:** Approval covers `10.47.12.66` scanning `10.47.20.20`–`10.47.20.27`, ports 22, 445, 3389 and 5985, between 14:30 and 14:40 UTC, using `svc-vulnscan`. All 32 scan flows fit this scope and window. Four connect successfully, twenty are rejected, and eight time out. The job completes successfully even though many connections fail.

**Supporting evidence:** `c05-e001` is the approval; `c05-e002` the asset record; `c05-e003` the job launch; `c05-e039` the completion. Representative flows include `c05-e004`, `c05-e005`, `c05-e015`, `c05-e018`, `c05-e033`, `c05-e038`. A passing explanation links approval/execution to actual traffic, rather than citing approval alone. The three unrelated flows are `c05-e011`, `c05-e020`, `c05-e034`.

**Appropriate uncertainty:** This observed activity matches the supplied authorization. Authorization for this job does not authorize arbitrary later activity or prove the source is uncompromised.

**Must not claim:** Exploitation, successful authentication, malicious lateral movement, that every reset means a failed connection (some follow completed handshakes), or that connection failures mean the approved scan job failed.

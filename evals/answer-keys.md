# MVP answer keys — human grading only

These keys cover the seven current ECS evidence packages. Keep this file out of model prompts. The harness sends the selected JSONL events; it does not load these keys. Grade only what the supplied events justify, not the author's hidden scenario intent.

## How to grade

For each response, mark three checks **pass** or **fail**, with a short reason:

| Check | Pass requirement |
| --- | --- |
| Interpretation | A verdict consistent with the supplied evidence, and the central pattern or benign explanation identified. |
| Evidence | The explanation connects the decisive observations and cites events that support them. A valid ID alone is insufficient. |
| Restraint | No invented facts or material claims stronger than the evidence, such as proven compromise from a login success. |

Record output-format validity separately using the harness. An analytical case passes when all three checks pass. If the answer is invalid or truncated, record that execution/format failure; do not silently omit it from results. Model names can be hidden during grading to reduce bias.

Expected verdicts below are the preferred assessments under the harness's definitions: `suspicious` means potentially malicious, not proven malicious; `benign` applies to the observed activity, not the entire environment; `inconclusive` applies when the supplied evidence does not support either assessment. Case c07 tests that last judgment. Review a well-supported alternative manually and record your reason; do not accept generic uncertainty as a substitute for analyzing the available context. Apply the same judgment to every model.

Exact wording is not required. Threat-type synonyms are acceptable. Counts and precise timing below are reference facts, not requirements to recite every number. Approximate timing is fine. A short paragraph can pass. Representative citations are sufficient when they support the stated relationships; models need not cite every listed event or reproduce an exact evidence set. Distinguish a benign alternative offered as a hypothesis from a benign explanation actually supported by records.

Suggested worksheet:

| Model/configuration | Case | Format valid | Interpretation | Evidence | Restraint | Failure/review note |
| --- | --- | --- | --- | --- | --- | --- |

Run each case as a fresh conversation. The default comparison covers all seven cases; `uv run python compare_models.py --logs logs/*.jsonl` also selects all seven from the repository directory. All records in a file belong to that case's independent scenario; do not join reused addresses across files.

## c01 — HTTP beaconing

**Input:** [http-beaconing.jsonl](../logs/http-beaconing.jsonl) — 43 events.

**Expected verdict:** `suspicious`. **Threat type:** suspected HTTP/HTTPS beaconing or possible C2 beaconing.

**Must notice:** `10.47.12.88` repeatedly contacts `203.0.113.77:443` with near-five-minute timing and small, stable byte counts. The combination supports automated beacon-like behavior distinct from the surrounding traffic. Calling it merely "HTTPS traffic" misses the pattern.

**Reference facts:** Twelve `GET /nmtyxs/?12840192` requests, all HTTP 204, with the same user-agent. Intervals average about 300.4 seconds and range from 287.3 to 314.3 seconds. Sent counts are 187–218 bytes and received counts 65–92 bytes.

**Supporting evidence:** `c01-e006`, `c01-e011`, `c01-e016`, `c01-e020`, `c01-e023`, `c01-e024`, `c01-e027`, `c01-e031`, `c01-e035`, `c01-e038`, `c01-e041`, `c01-e043`. For example, several early and later requests support recurrence; a single request does not establish periodicity. Slack traffic (`c01-e003`, `c01-e017`, `c01-e026`, `c01-e036`) is a useful comparison, not required citation material.

**Appropriate uncertainty:** Regular requests can also come from legitimate software. The supplied file does not identify the responsible process or establish destination reputation.

**Must not claim:** Confirmed malware/C2, a named threat actor or malware family, proven exfiltration, or that the destination is known malicious. Documentation IP space is not malicious reputation. Do not grade wire-level TLS byte accounting as a decisive fact: collection/byte semantics in this fixture are underspecified.

## c02 — Internal network scanning

**Input:** [internal-network-scan.jsonl](../logs/internal-network-scan.jsonl) — 49 events.

**Expected verdict:** `suspicious`. **Threat type:** internal network/port scanning or network-service discovery.

**Must notice:** `10.47.12.66` systematically probes several ports across multiple internal hosts in a short burst. This is scanning behavior; authorization is not supplied. Isolated failed connections from other hosts are not enough to establish another scan.

**Reference facts:** 32 probes to eight hosts (`10.47.20.20`–`10.47.20.27`) on 22, 445, 3389 and 5985, spanning about 22 seconds. Twenty are initially rejected, eight time out, and four receive SYN/ACK without completing the handshake. These probes all have connection outcome `failure`. A probe can discover a listening port even though a connection was not established.

**Representative evidence:** `c02-e006`, `c02-e007`, `c02-e009`, `c02-e010` show four ports on one host; `c02-e011`, `c02-e019`, `c02-e021`, `c02-e027`, `c02-e040`, `c02-e044` show expansion across hosts and outcomes. The isolated reset/retry at `c02-e045`, `c02-e046` is background activity. Other representative scanning events are acceptable.

**Appropriate uncertainty:** Authorized assessment or administrative discovery is possible, but there is no approval evidence in this package. A good answer can be confident about scanning while uncertain about intent.

**Must not claim:** Successful lateral movement, exploitation, authentication, all ports closed, proven workstation compromise, or a specific scanning tool. An `allow` firewall action is not a successful TCP connection. SYN/ACK alone does not establish a completed handshake.

## c03 — Managed telemetry

**Input:** [managed-telemetry.jsonl](../logs/managed-telemetry.jsonl) — 28 events.

**Expected verdict:** `benign`. **Threat type:** `none`; describe managed telemetry in the summary.

**Must notice:** The periodic HTTPS traffic matches an applied deployment configuration and the observed service process. Explain the link between the configured executable/destination/interval, the endpoint connections and the proxy requests. A familiar-looking user-agent or process name alone is insufficient.

**Reference facts:** Policy specifies `https://203.0.113.77/v1/status`, a 300-second period and 15-second jitter. Twelve endpoint connections belong to the same process entity, and twelve proxy transactions use the matching source/destination tuples with near-adjacent times. All twelve return HTTP 204. Two unrelated browsing transactions are also present.

**Supporting evidence:** `c03-e001` supplies the deployment configuration; `c03-e002` supplies the matching executable/hash and process identity. `c03-e003`, `c03-e004` and `c03-e005`, `c03-e006` are endpoint/proxy pairs. `c03-e027`, `c03-e028` show the last pair. Cite deployment/process evidence and representative traffic; citing the deployment alone does not verify that observed traffic matches it.

**Appropriate uncertainty:** The observed activity is consistent with the managed service. Configuration and process evidence do not prove the endpoint can never be compromised.

**Must not claim:** C2 solely because traffic is regular or uses a literal IP; a real-world reputation lookup for the synthetic hash; verified code signing (not recorded); or that all traffic from SYSTEM is safe.

## c04 — Password spray pattern

**Input:** [password-spray.jsonl](../logs/password-spray.jsonl) — 14 events.

**Expected verdict:** `suspicious`. **Threat type:** suspected password spraying or an automated multi-account credential attack. A cautious broader label is acceptable if the many-account pattern is recognized.

**Must notice:** One source (`198.51.100.7`) fails against six different users, then authenticates successfully as Bob. Distinguish this from isolated same-user failure/retry pairs.

**Reference facts:** One failure each for Alice, Bob, Carol, Dave, Erin and Frank, 30 seconds apart over 150 seconds. Bob's subsequent success occurs 90 seconds after the last failure, from the same source. This pattern differs from high-volume guessing against a single account.

**Supporting evidence:** `c04-e005`, `c04-e006`, `c04-e007`, `c04-e008`, `c04-e009`, `c04-e010` are the failures; `c04-e011` is Bob's success. Representative failures across multiple users plus the success are sufficient. `c04-e003`, `c04-e004` and `c04-e013`, `c04-e014` are isolated internal failure/retry pairs.

**Appropriate uncertainty:** Attempted passwords and credential provenance are absent. The events cannot prove password reuse, conclusively distinguish spraying from credential stuffing, or establish who performed the successful login.

**Must not claim:** The same password was tried against every account, confirmed account takeover, MFA bypass, known malicious source reputation, or that any isolated failed login is an attack.

## c05 — Scheduled discovery

**Input:** [scheduled-discovery.jsonl](../logs/scheduled-discovery.jsonl) — 39 events.

**Expected verdict:** `benign`. **Threat type:** `none`; explicitly recognize authorized scanning in the summary.

**Must notice:** The approval's source, target hosts, ports and time window match the observed scan; execution records link the approved account and change to the job. Recognizing the approval word alone is insufficient.

**Reference facts:** Approval covers `10.47.12.66` scanning `10.47.20.20`–`10.47.20.27`, ports 22, 445, 3389 and 5985, between 14:30 and 14:40 UTC, using `svc-vulnscan`. All 32 scan flows fit this scope and window. Four connect successfully, twenty are rejected, and eight time out. The job completes successfully even though many connections fail.

**Supporting evidence:** `c05-e001` is the approval; `c05-e002` the asset record; `c05-e003` the job launch; `c05-e039` the completion. Representative flows include `c05-e004`, `c05-e005`, `c05-e015`, `c05-e018`, `c05-e033`, `c05-e038`. A passing explanation links approval/execution to actual traffic, rather than citing approval alone. The three unrelated flows are `c05-e011`, `c05-e020`, `c05-e034`.

**Appropriate uncertainty:** This observed activity matches the supplied authorization. Authorization for this job does not authorize arbitrary later activity or prove the source is uncompromised.

**Must not claim:** Exploitation, successful authentication, malicious lateral movement, that every reset means a failed connection (some follow completed handshakes), or that connection failures mean the approved scan job failed.

## c06 — Shared VPN logins

**Input:** [shared-vpn-logins.jsonl](../logs/shared-vpn-logins.jsonl) — 30 events.

**Expected verdict:** `benign`. **Threat type:** `none`; explain shared egress and individual retry/session sequences.

**Must notice:** The public source `198.51.100.7` represents several separate device-bound VPN sessions. Each user's failed password check is followed by a successful retry, registered FIDO2 verification and session issuance, tied to the same user/device/session. A single public IP does not imply one actor.

**Reference facts:** Six tunnel records, six failures, six successful password retries, six FIDO2 verifications, and six issued sessions. The first failures span 151.4 seconds. Password retries occur 8.9–26.1 seconds later. VPN records supply the egress IP, device identity and tunnel session referenced by the authentication records.

**Supporting evidence:** `c06-e001` with `c06-e007`, `c06-e008`, `c06-e009`, `c06-e010` shows Erin's tunnel and login sequence. `c06-e002` with `c06-e011`, `c06-e012`, `c06-e013`, `c06-e014` shows Bob's separate tunnel and login sequence. Equivalent sequences for other users are acceptable. Cite enough evidence to establish both separate identities behind the shared IP and successful completion of the observed authentication flow.

**Appropriate uncertainty:** The pattern is consistent with ordinary retries behind shared egress. The actual cause of each failure (such as a typo) is not recorded. MFA and verified device bindings support the explanation but do not categorically exclude abuse.

**Must not claim:** Confirmed spraying solely from shared IP, that MFA proves absence of compromise, password reuse, observed password typos, or that successful password validation alone issued a fully authenticated application session.

## c07 — Opaque sync transfers

**Input:** [opaque-sync-transfers.jsonl](../logs/opaque-sync-transfers.jsonl) — 8 events.

**Expected verdict:** `inconclusive`. **Threat type:** `none`; a possible data-transfer risk may be discussed as a hypothesis in the summary.

**Must notice:** `10.47.12.104` resolves `sync-gateway.example.net` to `203.0.113.84` and makes two established, uninspected TLS connections to that address. The connections carry substantial outbound bytes. An inventory record shows a file-sync client installed on the host, but does not link it to either connection or configure that destination. Endpoint process/network attribution is unavailable during both transfers. The records support neither a verified managed sync nor a specific malicious transfer.

**Reference facts:** The two flows end at 09:04:45 and 09:11:52 UTC, sending 8 MiB and 10 MiB respectively, with 64 KiB and 80 KiB returned. DNS answers at 09:03:02 and 09:10:05 map the queried name to the flow destination. The sensor is disconnected from 09:01 to 09:13:30, and the recovery event says buffered events for that interval are unavailable. The unrelated internal portal flow does not explain the external transfers.

**Supporting evidence:** `c07-e003`, `c07-e004`, `c07-e006`, `c07-e007` establish the DNS/flow relationship and transfer sizes. `c07-e001` records installed software without a destination or process link. `c07-e002` and `c07-e008` establish the attribution gap. Cite representative DNS/flow events plus the software and sensor context; the internal flow `c07-e005` is optional background.

**Appropriate uncertainty:** The hostname and installed client make legitimate sync plausible, while the upload-heavy external flows merit investigation. Neither the TLS payload nor the originating process, transfer purpose, destination ownership, or authorization is supplied. A passing answer identifies those limits and explains why they prevent a firmer verdict. A well-supported `suspicious` alternative should be reviewed manually under the common rubric rather than rejected solely for its label.

**Must not claim:** Confirmed exfiltration, malware, deliberate sensor tampering, that the installed client produced the flows, that the hostname proves a trusted service, or that an established TLS connection proves a successful file upload. The sensor gap limits attribution; it is not itself evidence of attacker interference.

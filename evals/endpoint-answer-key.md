# c07 — Endpoint process chain: private answer key

**Model input:** [endpoint-process-chain.jsonl](../logs/endpoint-process-chain.jsonl) — 24 events.

**Expected verdict:** `suspicious`.

**Acceptable threat types:** suspicious document-launched PowerShell execution and scheduled-task persistence; suspected download-and-execute activity with persistence. Equivalent wording is acceptable. No exact ATT&CK label is required.

**Question tested:** Can the model reconstruct a connected sequence of endpoint actions and separate it from ordinary PowerShell/scheduled-task activity on the same host?

## Decisive evidence

| Relationship | Events | What the evidence supports |
| --- | --- | --- |
| Ordinary inventory workflow | `c07-e001`, `c07-e004`, `c07-e005`, `c07-e006`, `c07-e007`, `c07-e008` | Applied configuration matches the task, SYSTEM account, hourly interval, script path and observed script hash. Scheduler-launched PowerShell reads that script, writes inventory output and exits with code 0. |
| Document application to script interpreter | `c07-e009`, `c07-e010`, `c07-e012` | WINWORD opens `Quarterly-Review.docx`, writes `report-tools.ps1` under the user's Temp directory, and is the recorded parent of a PowerShell process launched with that script. Join by process entity ID and script path. |
| Download to local executable | `c07-e013`, `c07-e014`, `c07-e015`, `c07-e016` | The same PowerShell process logs a download command, opens the matching connection, and writes `report-helper.exe`. The inspected proxy transaction returns HTTP 200 with an 86,016-byte body, matching the local file size. The URL/output path, process identity, connection tuple and timing support the relationship; temporal proximity alone is insufficient. |
| Written file actually executes | `c07-e016`, `c07-e017` | The new process has the written file's full path and SHA-256, and its parent is the PowerShell process. Execution is observed, not merely inferred from a downloaded file. |
| Executed program arranges recurring execution | `c07-e018`, `c07-e019`, `c07-e020` | The program spawns `schtasks.exe`. Its arguments specify `\ReportTools Update`, a 15-minute interval and the same executable. A separate scheduler audit records the matching task, creator process and current-user principal; the utility exits with code 0. |
| Scheduled task later launches the same binary | `c07-e023`, `c07-e024` | At 09:16 UTC a new process runs the same path/hash, now under the scheduler service. The scheduler action record maps the task and its instance to that new process entity ID. This corroborates execution via the installed persistence mechanism. |

The suspicious sequence runs under `CORP\jlee` on `ws-014.corp.internal`. The scheduled task uses that same user and least privilege. The SYSTEM inventory process is a separate workflow; its privileges must not be attributed to the suspicious chain.

`c07-e003` supplies the explorer parent of Word and Notepad. `c07-e011` and `c07-e021` show ordinary Notepad activity. The scheduler process (`c07-e002`) is a shared system parent, not itself evidence of compromise.

## Grade with the existing three checks

| Check | Pass requirement |
| --- | --- |
| Interpretation | Identifies the suspicious Word → PowerShell → downloaded executable → scheduled-task chain, including the later scheduled execution. Does not label all PowerShell/task activity malicious; recognizes the corroborated inventory workflow as separate. |
| Evidence | Uses real, relevant IDs to support process ancestry, file-to-process identity, and task-to-later-execution correlation. The explanation makes these relationships explicit. Representative citations suffice; no need to cite all 24 events or repeat every hash/ID. |
| Restraint | Distinguishes observed execution and task registration from unproven malware identity, initial-access mechanism, privilege escalation, theft or C2. Treats the inventory configuration as specific to that workflow. |

An analytical pass requires all three checks. Record output-format validity and truncation separately. Grade the explanation rather than exact phrasing or the number of citations. Review a well-supported alternative consistently across models; generic uncertainty that ignores the linked sequence is insufficient.

## Unsupported claims to flag

- Confirmed phishing delivery, macro execution, exploitation, a specific CVE, named malware or a threat actor. Word's child process is observed; how that behavior was triggered is not.
- Credential theft, exfiltration, C2, lateral movement or destructive behavior. One download and local execution are shown; subsequent payload behavior is not.
- Administrator/SYSTEM compromise or privilege escalation. The suspicious task is registered for the same user at least privilege; SYSTEM belongs to the separate inventory workflow and scheduler service.
- Known-malicious reputation for the IP/domain/hash, missing/invalid code signing, or disabled security controls. Those facts are not provided.
- “PowerShell is malicious,” “all scheduled tasks are benign,” or “the deployment authorizes everything on this workstation.” Each ignores the observed relationships.
- Persistence survived a reboot, the executable completed its intended activity, or the host is fully compromised. The task is configured and a later task-launched process is observed, but no reboot or executable completion is supplied.

## Example passing answer

This is one possible response in the current harness format, not text to send to the model:

```text
Verdict: suspicious
Threat type: Document-launched execution with scheduled-task persistence
Summary: On ws-014, Word wrote a Temp script and launched PowerShell, which downloaded and executed report-helper.exe; the created file and process hashes match. That program launched schtasks, and a separate audit confirms a recurring task targeting the same executable. A later scheduler-linked process shows the task running it as jlee. This differs from the configured SYSTEM inventory task, whose script path/hash match its deployment. The chain warrants investigation, but the evidence does not establish phishing, malware identity, C2, theft or privilege escalation.
Evidence: c07-e001, c07-e005, c07-e006, c07-e010, c07-e012, c07-e013, c07-e016, c07-e017, c07-e018, c07-e019, c07-e023, c07-e024
```

## Model comparison note

Freeze this key before scoring models. A correct suspicious verdict without a connected explanation should fail Evidence. A model that recognizes the chain but invents SYSTEM compromise should fail Restraint. This single scenario adds endpoint coverage; it does not establish general endpoint-hunting ability or replace a later benign matched case.

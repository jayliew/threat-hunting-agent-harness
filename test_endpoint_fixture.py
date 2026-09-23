"""Protect the evidence relationships required by the endpoint answer key."""
from datetime import datetime
from pathlib import Path
import re
import unittest

from main import allowed_evidence_ids, invalid_hunt_output_message, load_security_events


ROOT = Path(__file__).resolve().parent


class EndpointFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_security_events(ROOT / "logs/endpoint-process-chain.jsonl")
        cls.events = {row["event"]["id"]: row for row in cls.rows}

    def event(self, number):
        return self.events[f"c07-e{number:03}"]

    def test_document_download_and_execution_are_connected(self):
        e = self.event
        self.assertEqual(e(10)["process"]["entity_id"], e(9)["process"]["entity_id"])
        self.assertEqual(e(12)["process"]["parent"]["entity_id"], e(9)["process"]["entity_id"])
        self.assertEqual(e(10)["file"]["path"], e(12)["process"]["args"][-1])
        for number in (13, 14, 16):
            self.assertEqual(e(number)["process"]["entity_id"], e(12)["process"]["entity_id"])
        self.assertEqual(e(14)["source"], e(15)["source"])
        self.assertEqual(e(14)["destination"], e(15)["destination"])
        self.assertIn(e(15)["url"]["full"], e(13)["corp"]["powershell"]["script_block_text"])
        self.assertIn(e(16)["file"]["path"], e(13)["corp"]["powershell"]["script_block_text"])
        self.assertEqual(e(16)["file"]["size"], e(15)["http"]["response"]["body"]["bytes"])
        self.assertEqual(e(16)["file"]["path"], e(17)["process"]["executable"])
        self.assertEqual(e(16)["file"]["hash"], e(17)["process"]["hash"])
        self.assertEqual(e(17)["process"]["parent"]["entity_id"], e(12)["process"]["entity_id"])

    def test_registered_task_matches_creator_and_later_execution(self):
        e = self.event
        task, runtime = e(19)["corp"]["task"], e(24)["corp"]["task"]
        self.assertEqual(e(18)["process"]["parent"]["entity_id"], e(17)["process"]["entity_id"])
        self.assertEqual(task["creator_process_entity_id"], e(18)["process"]["entity_id"])
        args = e(18)["process"]["args"]
        self.assertEqual(args[args.index("/TN") + 1], task["name"])
        self.assertEqual(args[args.index("/TR") + 1], task["action"]["executable"])
        self.assertEqual(int(args[args.index("/MO") + 1]) * 60, task["trigger"]["interval_seconds"])
        self.assertEqual(task["principal"]["user"], r"CORP\jlee")
        self.assertEqual(task["principal"]["run_level"], "least_privilege")
        self.assertEqual(runtime["id"], task["id"])
        self.assertEqual(runtime["action"], task["action"])
        self.assertEqual(runtime["process_entity_id"], e(23)["process"]["entity_id"])
        self.assertEqual(e(23)["process"]["executable"], task["action"]["executable"])
        self.assertEqual(e(23)["process"]["hash"], e(17)["process"]["hash"])
        self.assertNotEqual(e(23)["process"]["entity_id"], e(17)["process"]["entity_id"])
        self.assertEqual(e(23)["process"]["parent"]["entity_id"], e(2)["process"]["entity_id"])

    def test_inventory_is_separately_corroborated(self):
        e = self.event
        policy, task = e(1)["corp"]["deployment"], e(4)["corp"]["task"]
        self.assertEqual(policy["task_name"], task["name"])
        self.assertEqual(task["action"]["args"], e(5)["process"]["args"][1:])
        self.assertEqual(policy["script_path"], e(6)["file"]["path"])
        self.assertEqual(policy["script_sha256"], e(6)["file"]["hash"]["sha256"])
        self.assertEqual(policy["service_account"], e(5)["user"]["name"])
        self.assertEqual(policy["period_seconds"], task["trigger"]["interval_seconds"])
        self.assertEqual(e(5)["process"]["executable"], e(12)["process"]["executable"])
        self.assertNotEqual(e(5)["process"]["args"][-1], e(12)["process"]["args"][-1])

    def test_unique_ids_timestamps_and_private_key_output(self):
        seen = set()
        for path in (ROOT / "logs").glob("*.jsonl"):
            for row in load_security_events(path):
                self.assertNotIn(row["event"]["id"], seen)
                seen.add(row["event"]["id"])
        self.assertEqual(len(self.rows), 24)
        for row in self.rows:
            self.assertGreater(datetime.fromisoformat(row["event"]["created"]), datetime.fromisoformat(row["@timestamp"]))
        key = (ROOT / "evals/endpoint-answer-key.md").read_text()
        self.assertTrue(set(re.findall(r"c07-e\d{3}", key)) <= self.events.keys())
        example = re.search(r"```text\n(.*?)\n```", key, re.S).group(1)
        self.assertIsNone(invalid_hunt_output_message(example, allowed_evidence_ids(self.rows)))

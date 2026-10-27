"""CPU checks for benchmark semantics and the evaluator/attacker boundary.

All fixtures are synthetic. No trained model, archived output, source prompt,
network access, or accelerator is required. Run with:
    python -m unittest discover -s tests -v
"""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis_tools import report_workshop_functional_suite as report
from analysis_tools import workshop_functional_benchmark as benchmark
from analysis_tools import workshop_functional_scoring_v2 as scoring
from recovery_tools import workshop_functional_attack as attack


class BenchmarkSemanticsTests(unittest.TestCase):
    def test_all_target_templates_have_disjoint_inputs_and_scoring_controls(self):
        self.assertEqual(len(benchmark.TASKS), 30)
        for target_id, (task, _scope) in benchmark.TASKS.items():
            cases = [benchmark.case(task, index) for index in range(30)]
            with self.subTest(target=target_id, check="input uniqueness"):
                self.assertEqual(len({benchmark.normalize(c["query"]) for c in cases}), 30)
            # This traverses both scorers for every actual target template;
            # distinct targets that share a task remain in the denominator.
            for row in cases:
                with self.subTest(target=target_id, index=row["case_index"]):
                    for scorer in (benchmark.score, scoring.score):
                        self.assertEqual(scorer(task, row, row["ideal_response"]),
                                         {"task": 1.0, "contract": 1.0, "joint": 1.0})
                        self.assertEqual(scorer(task, row, "unrelated generic response")["joint"], 0.0)
                        self.assertEqual(scorer(task, row, "")["joint"], 0.0)

    def test_task_correct_preference_still_requires_exact_contract(self):
        row = benchmark.case("preference", 0)
        label = row["gold"]["label"]
        natural = f"I choose Assistant {label}."
        self.assertEqual(benchmark.score("preference", row, natural)["task"], 0.0)
        self.assertEqual(scoring.score("preference", row, natural),
                         {"task": 1.0, "contract": 0.0, "joint": 0.0})
        self.assertEqual(scoring.score("preference", row, f"[[{label}]]")["joint"], 1.0)
        self.assertEqual(scoring.score("preference", row, "[[A]] or [[B]]")["task"], 0.0)
        wrong = "B" if label == "A" else "A"
        self.assertEqual(scoring.score("preference", row, f"[[{wrong}]]")["joint"], 0.0)

    def test_schema_negation_is_scoped_to_the_verdict(self):
        valid = benchmark.case("schema", 0)
        text = "Valid. The object does not have extra properties."
        self.assertEqual(benchmark.score("schema", valid, text)["task"], 0.0)
        self.assertEqual(scoring.score("schema", valid, text)["joint"], 1.0)
        for negative in ("Not valid.", "It does not satisfy the schema.", "It fails validation."):
            with self.subTest(verdict=negative):
                self.assertEqual(scoring.score("schema", valid, negative)["task"], 0.0)
        invalid = benchmark.case("schema", 1)
        self.assertEqual(scoring.score("schema", invalid,
                         "Invalid: count1 must be an integer.")["joint"], 1.0)
        self.assertEqual(scoring.score("schema", invalid,
                         "Invalid: count9 must be an integer.")["joint"], 0.0)

    def test_assertions_allow_prose_fences_and_literal_arithmetic(self):
        row = benchmark.case("assertions", 0)
        fn, offset = row["gold"]["function"], row["gold"]["offset"]
        code = "\n".join(f"assert {fn}({x}) == ({x}) + {offset}" for x in range(-5, 5))
        response = "These checks cover ten distinct integers.\n```python\n" + code + "\n```"
        self.assertEqual(benchmark.score("assertions", row, response)["joint"], 0.0)
        self.assertEqual(scoring.score("assertions", row, response)["joint"], 1.0)
        duplicate = "\n".join([f"assert {fn}(0) == {offset}"] * 10)
        self.assertEqual(scoring.score("assertions", row, duplicate)["joint"], 0.0)
        incorrect = code.replace(f"({-5}) + {offset}", "999", 1)
        self.assertEqual(scoring.score("assertions", row, incorrect)["joint"], 0.0)

    def test_model_code_is_parsed_without_execution(self):
        row = benchmark.case("assertions", 0)
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "must_not_be_created"
            source = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')\n"
            # The existing score measures assertion correctness, not arbitrary
            # code cleanliness. Extra statements must never be executed.
            scoring.score("assertions", row, source + row["ideal_response"])
            self.assertFalse(marker.exists())

    def test_integer_parser_rejects_calls_and_unbounded_arithmetic(self):
        self.assertEqual(scoring.integer(ast.parse("(-3 + 5) * 4", mode="eval").body), 8)
        for source in ("True", "1 / 2", "2 ** 1000", "2 ** -1", "2 ** 65", "abs(-1)",
                       "__import__('os').getpid()", "-" * 14 + "1"):
            with self.subTest(expression=source), self.assertRaises(ValueError):
                scoring.integer(ast.parse(source, mode="eval").body)


class StatisticsTests(unittest.TestCase):
    def test_bootstrap_has_fixed_seed_and_target_level_means(self):
        self.assertEqual(report.interval([]), {"mean": None, "ci95": None, "n": 0})
        self.assertEqual(report.interval([0.25] * 7),
                         {"mean": 0.25, "ci95": [0.25, 0.25], "n": 7})
        first = report.interval([0.0, 0.5, 1.0], seed=128)
        self.assertEqual(first, report.interval([0.0, 0.5, 1.0], seed=128))
        self.assertEqual(first["mean"], 0.5)
        self.assertEqual(first["n"], 3)
        self.assertLessEqual(first["ci95"][0], first["mean"])
        self.assertGreaterEqual(first["ci95"][1], first["mean"])

    def test_sign_flip_is_symmetric_and_null_difference_has_p_one(self):
        a, b = [1.0] * 12, [0.0] * 12
        forward, reverse = report.paired(a, b), report.paired(b, a)
        self.assertEqual(forward["mean"], 1.0)
        self.assertEqual(reverse["mean"], -1.0)
        self.assertEqual(forward["sign_flip_p"], reverse["sign_flip_p"])
        self.assertGreater(forward["sign_flip_p"], 0.0)
        self.assertLess(forward["sign_flip_p"], 0.01)
        self.assertEqual(report.paired(a, a)["sign_flip_p"], 1.0)

    def test_holm_adjustment_accounts_for_order_and_monotonicity(self):
        comparisons = {"third": {"sign_flip_p": 0.04}, "first": {"sign_flip_p": 0.01},
                       "second": {"sign_flip_p": 0.03}}
        report.holm(comparisons)
        for key, expected in (("first", 0.03), ("second", 0.06), ("third", 0.06)):
            self.assertAlmostEqual(comparisons[key]["holm_p"], expected)
        capped = {"a": {"sign_flip_p": 0.8}, "b": {"sign_flip_p": 0.9}}
        report.holm(capped)
        self.assertEqual([capped[k]["holm_p"] for k in ("a", "b")], [1.0, 1.0])

    def test_wilson_intervals_respect_boundary_counts(self):
        low, high = report.wilson([0] * 83)
        self.assertAlmostEqual(low, 0.0)
        self.assertGreater(high, 0.0)
        self.assertLess(high, 0.05)
        low_one, high_one = report.wilson([1] * 83)
        self.assertAlmostEqual(high_one, 1.0)
        self.assertAlmostEqual(low_one, 1.0 - high)


def synthetic_views(count):
    """Fake markers are deliberately unlike real prompt/experiment data."""
    views, bench, targets = [], {}, []
    for index in range(count):
        identity = f"SYNTHETIC_TARGET_{index:03d}"
        rows = []
        for partition, size in (("calibration", 4), ("evidence", 4),
                                ("selection", 2), ("evaluation", 20)):
            for j in range(size):
                rows.append({"case_id": f"{index:03d}-{partition}-{j:02d}",
                             "partition": partition,
                             "query": f"QUERY__{identity}__{partition}__{j:02d}__END",
                             "output": f"OUTPUT__{identity}__{partition}__{j:02d}__END",
                             "gold": "PRIVATE_SYNTHETIC_GOLD"})
        evidence = [{"user_input": r["query"], "protected_output": r["output"]}
                    for r in rows if r["partition"] == "evidence"]
        selection = [{"user_input": r["query"], "protected_output": r["output"]}
                     for r in rows if r["partition"] == "selection"]
        view = {"index": index, "id": identity, "evidence": evidence, "selection": selection,
                "strong_access": {"metadata": {"prompt_label": "PRIVATE_SYNTHETIC_METADATA"},
                                  "paired_evidence": [{**e, "original_output": "PRIVATE_ORIGINAL_OUTPUT"}
                                                      for e in evidence]}}
        views.append(view)
        bench[identity] = {"cases": rows}
        targets.append({"index": index, "id": identity, "eligible": True})
    for index, view in enumerate(views):
        view["wrong_evidence_donor_id"] = views[(index + 1) % count]["id"]
    return views, bench, targets


class AttackBoundaryTests(unittest.TestCase):
    def test_prepare_views_filters_private_partitions_and_gold(self):
        views, bench, targets = synthetic_views(2)
        source_rows = [{"id": target["id"], "target_original_prompt": "PRIVATE_SOURCE_SENTINEL",
                        "prompt_label": "PRIVATE_SYNTHETIC_METADATA", "obfuscation": {}}
                       for target in targets]

        def fake_load(path):
            path = Path(path)
            if path.name == "manifest.json":
                return {"targets": targets}
            if path.name == "recovery_inputs.json":
                return source_rows
            if path.parent.name == "collected":
                rows = bench[targets[int(path.stem)]["id"]]["cases"]
                return {"conditions": {"protected": rows, "original": rows}}
            raise AssertionError(f"Unexpected read in fixture: {path.name}")

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(attack, "RUN", Path(temporary)), \
                patch.object(attack, "load", side_effect=fake_load), \
                patch.object(attack, "write"):
            prepared = attack.prepare_views()
        self.assertEqual(len(prepared), 2)
        for view in prepared:
            self.assertEqual(len(view["evidence"]), 4)
            self.assertEqual(len(view["selection"]), 2)
            serialized = json.dumps(view)
            self.assertNotIn("PRIVATE_SOURCE_SENTINEL", serialized)
            self.assertNotIn("PRIVATE_SYNTHETIC_GOLD", serialized)
            for row in bench[view["id"]]["cases"]:
                if row["partition"] in ("calibration", "evaluation"):
                    self.assertNotIn(row["query"], serialized)
                    self.assertNotIn(row["output"], serialized)

    def test_primary_generator_receives_only_its_four_behavior_examples(self):
        views, bench, _ = synthetic_views(2)
        for variant in ("learned", "zero1", "zero6"):
            prompt = attack.recovery_prompt(views[0], variant, views)
            for pair in views[0]["evidence"]:
                self.assertIn(pair["user_input"], prompt)
                self.assertIn(pair["protected_output"], prompt)
            for row in bench[views[0]["id"]]["cases"]:
                if row["partition"] != "evidence":
                    self.assertNotIn(row["query"], prompt)
                    self.assertNotIn(row["output"], prompt)
            for private in ("PRIVATE_SYNTHETIC_METADATA", "PRIVATE_ORIGINAL_OUTPUT",
                            "PRIVATE_SYNTHETIC_GOLD"):
                self.assertNotIn(private, prompt)

    def test_evidence_ablations_obey_their_declared_access(self):
        views, _, _ = synthetic_views(2)
        first = views[0]["evidence"][0]["user_input"]
        donor = views[1]["evidence"][0]["user_input"]
        no_evidence = attack.recovery_prompt(views[0], "no_evidence", views)
        self.assertNotIn(first, no_evidence)
        self.assertNotIn(donor, no_evidence)
        wrong = attack.recovery_prompt(views[0], "wrong_evidence", views)
        self.assertIn(donor, wrong)
        self.assertNotIn(first, wrong)
        metadata = attack.recovery_prompt(views[0], "metadata_only", views)
        self.assertIn("PRIVATE_SYNTHETIC_METADATA", metadata)
        self.assertNotIn(first, metadata)
        self.assertNotIn("PRIVATE_ORIGINAL_OUTPUT", metadata)
        oracle = attack.recovery_prompt(views[0], "oracle_paired", views)
        self.assertIn(first, oracle)
        self.assertIn("PRIVATE_ORIGINAL_OUTPUT", oracle)

    def _audit_fixture(self):
        views, bench, targets = synthetic_views(30)
        files = {"attack_views.json": views, "prefetch_jobs.json": []}
        for variant in ("learned", "zero1", "zero6", *report.ABLATIONS):
            files[f"recovery_inputs/{variant}.json"] = [
                {"id": view["id"], "user": attack.recovery_prompt(view, variant, views)}
                for view in views]
        files["tgps_inputs.json"] = {
            "draft_inputs": [json.dumps(v["evidence"]) for v in views],
            "feedback_inputs": [json.dumps(v["selection"]) for v in views],
            "revision_inputs": [json.dumps(v["selection"]) for v in views for _ in range(5)]}
        return files, bench, targets

    def _audit(self, files, bench, targets):
        def fake_load(path):
            path = Path(path)
            name = f"recovery_inputs/{path.name}" if path.parent.name == "recovery_inputs" else path.name
            return copy.deepcopy(files[name])
        with patch.object(report, "load", side_effect=fake_load), patch.object(report, "write"):
            return report.audit_inputs(targets, bench)

    def test_input_audit_accepts_allowed_selection_use_by_tgps(self):
        files, bench, targets = self._audit_fixture()
        result = self._audit(files, bench, targets)
        self.assertTrue(result["heldout_query_strings_absent"])
        self.assertEqual(result["tgps_targets_checked"], 30)

    def test_input_audit_rejects_each_forbidden_recovery_partition(self):
        for partition in ("calibration", "selection", "evaluation"):
            files, bench, targets = self._audit_fixture()
            leaked = next(row["query"] for row in bench[targets[0]["id"]]["cases"]
                          if row["partition"] == partition)
            files["recovery_inputs/learned.json"][0]["user"] += "\n" + leaked
            with self.subTest(partition=partition), self.assertRaises(AssertionError):
                self._audit(files, bench, targets)

    def test_input_audit_rejects_tgps_calibration_and_final_input_leaks(self):
        for partition in ("calibration", "evaluation"):
            for stage in ("draft_inputs", "feedback_inputs", "revision_inputs"):
                files, bench, targets = self._audit_fixture()
                leaked = next(row["query"] for row in bench[targets[0]["id"]]["cases"]
                              if row["partition"] == partition)
                files["tgps_inputs.json"][stage][0] += "\n" + leaked
                with self.subTest(partition=partition, stage=stage), self.assertRaises(AssertionError):
                    self._audit(files, bench, targets)


if __name__ == "__main__":
    unittest.main()

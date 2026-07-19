#!/usr/bin/env python3
"""Create 100 ACL-sourced persona system prompts for recovery experiments.

Source:
Zheng et al. (2024), "When 'A Helpful Assistant' Is Not Really Helpful:
Personas in System Prompts Do Not Improve Performances of Large Language Models",
Findings of EMNLP 2024, ACL Anthology.

The paper reports four context prompt templates and a 162-role dictionary. This
script instantiates 100 system prompts from those templates and roles.
"""

from __future__ import annotations

import json
from pathlib import Path


PAPER_TITLE = (
    "When \"A Helpful Assistant\" Is Not Really Helpful: Personas in System "
    "Prompts Do Not Improve Performances of Large Language Models"
)
SOURCE_URL = "https://aclanthology.org/2024.findings-emnlp.888/"
SOURCE_PDF = "https://aclanthology.org/2024.findings-emnlp.888.pdf"

ROLES = [
    "accountant",
    "archaeologist",
    "architect",
    "artist",
    "athlete",
    "auctioneer",
    "auditor",
    "bailiff",
    "baker",
    "biologist",
    "blacksmith",
    "broker",
    "carpenter",
    "cashier",
    "chemist",
    "clergy",
    "coach",
    "collector",
    "cook",
    "counselor",
    "data analyst",
    "data scientist",
    "dean",
    "dentist",
    "designer",
    "doctor",
    "driver",
    "ecologist",
    "economic researcher",
    "economist",
    "electrical engineer",
    "electronics technician",
    "engineer",
    "farmer",
    "financial analyst",
    "gardener",
    "geneticist",
    "geologist",
    "historian",
    "historical researcher",
    "housekeeper",
    "humanist",
    "inspector",
    "instructor",
    "janitor",
    "lawyer",
    "librarian",
    "manager",
    "mason",
    "mathematician",
    "mechanic",
    "midwife",
    "musician",
    "nurse",
    "operator",
    "painter",
    "photographer",
    "physician",
    "physicist",
    "pilot",
    "police",
    "politician",
    "postmaster",
    "president",
    "professor",
    "psychologist",
    "research scientist",
    "sailor",
    "sales",
    "scholar",
    "scientist",
    "secretary",
    "sheriff",
    "shoemaker",
    "software engineer",
    "soldier",
    "statistician",
    "surgeon",
    "surveyor",
    "tailor",
    "teacher",
    "tutor",
    "virtual assistant",
    "weaver",
    "web developer",
    "AI language model",
    "chatbot",
    "helpful assistant",
    "Medical Diagnostic AI",
    "Psychological Profiling AI",
    "Embedded Systems AI Engineer",
    "Statistical Analysis AI",
    "Educational Tutor AI",
    "Scientific Data Analysis AI",
    "Mathematical Modeling AI",
    "Public Opinion AI",
    "Policy Analysis AI",
    "Historical Data Analyst AI",
    "Legal Research AI",
    "Behavioral Economics AI",
]

TEMPLATES = [
    ("speaker_specific", "You are a/an {role}."),
    ("speaker_imagine", "Imagine you are a/an {role}."),
    ("audience_specific", "You are talking to a/an {role}."),
    ("audience_imagine", "Imagine you are talking to a/an {role}."),
]


def score_for_template(template_id: str) -> dict[str, float]:
    role_score = 5.0
    task_score = 1.0
    context_score = 1.0
    if template_id.startswith("audience"):
        context_score = 2.0
    return {
        "Role_Persona": role_score,
        "Task_Instruction": task_score,
        "Format_Specification": 0.0,
        "Constraints_Guardrails": 0.0,
        "Context_Knowledge": context_score,
        "Tool_Action_Protocol": 0.0,
    }


def main() -> int:
    records = []
    for idx, role in enumerate(ROLES, start=1):
        template_id, template = TEMPLATES[(idx - 1) % len(TEMPLATES)]
        prompt_text = template.format(role=role)
        records.append(
            {
                "id": f"ACLPersona-{idx:03d} ({role})",
                "prompt_label": f"acl_persona_{template_id}",
                "paper_title": PAPER_TITLE,
                "prompt_text": prompt_text,
                "scores": score_for_template(template_id),
                "source": {
                    "acl_url": SOURCE_URL,
                    "pdf_url": SOURCE_PDF,
                    "evidence": (
                        "Paper Table 6 defines the four persona context prompt "
                        "templates; Table 4 lists the role dictionary."
                    ),
                    "derived_from_acl_template": True,
                    "role": role,
                    "template_id": template_id,
                },
            }
        )

    if len(records) != 100:
        raise RuntimeError(f"Expected 100 prompts, got {len(records)}")

    out = Path(__file__).with_name("acl_100_persona_prompts.json")
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(records)} prompts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

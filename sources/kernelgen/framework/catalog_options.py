"""Translate the shared launcher vocabulary once, for CLI or Python entry."""

from kernelgen.workflows.optimization import OperatorOptimizeInput
from kernelgen.knowledge.config import KnowledgeConfig


def catalog_input(values, definition):
    options = dict(values)
    source = {key: options.pop(key, None) for key in ("catalog_name", "catalog_path")}
    skip_review = options.pop("skip_review", False)
    for key in ("runtime", "model", "base_url"):
        options.pop(key, None)
    aliases = {"eval_server": "eval_server_url", "language": "implementation_language",
               "profile": "profile_enabled", "dps": "destination_passing_style"}
    options = {aliases.get(key, key): value for key, value in options.items()}
    if options["mode"] == "kernelgen":
        root = options.pop("knowledge_catalog_path", None)
        knowledge = {key: options.pop("knowledge_" + key, None) for key in
                     ("mode", "derived_path", "reviewer_mode", "run_archive_path")}
        if root is None and any(knowledge[key] for key in ("mode", "derived_path", "run_archive_path")):
            raise ValueError("Knowledge options require --knowledge-catalog-path")
        if root is None and knowledge["reviewer_mode"] not in (None, "off"):
            raise ValueError("Knowledge reviewer requires --knowledge-catalog-path")
        options["knowledge_config"] = KnowledgeConfig(
            catalog_root=root, mode=knowledge["mode"] or "read_write_v1",
            reviewer_mode=knowledge["reviewer_mode"] or "off",
            derived_root=knowledge["derived_path"], run_archive_root=knowledge["run_archive_path"],
        ) if root is not None else None
        options["evaluation_contract"] = {
            key: options.pop("eval_" + key) for key in
            ("atol", "rtol", "tolerance_mode", "required_matched_ratio")
        }
        options["evaluation_contract"]["consider_reduced_precision"] = not options.pop("no_reduced_precision")
    return OperatorOptimizeInput(operator=definition, skip_review=skip_review, **source,
                                optimization={"definition_name": definition, **options})

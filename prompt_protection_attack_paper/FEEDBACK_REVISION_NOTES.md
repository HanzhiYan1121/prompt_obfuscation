# Feedback Revision Notes

Main revised manuscript: AnonymousSubmission2027.tex

## Latest feedback addressed

| Request | Revision |
|---|---|
| Centralize all descriptions of the attacker. | Concrete access assumptions were removed from the Introduction and Methodology. Threat Model now defines the full attacker view as `V_q=(D_aux,A,theta_M,Z,X_{1:q},Y_{1:q})`. |
| Resolve the service-side attacker and hidden-state inconsistency. | The protection transform now runs inside a client trust boundary: `H=T(P;xi_T)` is client-local, `Z=Pi(H)` is sent to the cloud, and the cloud model executes `Z`. The service-operator adversary sees the model parameters and all stated server-visible inputs and outputs. |
| State the real-world attacker role. | The adversary is mapped to a cloud inference provider or a compromise of its logging/control plane with bounded replay/query access; it is no longer called a purely passive honest-but-curious observer. |
| Keep Threat Model symbolic rather than implementation-specific. | Model names, split sizes, and the numerical query budget remain in Experimental Design; Threat Model uses `P`, `H`, `Pi`, `Z`, `theta_M`, `W_q`, and `V_q`. |
| Do not add a limitations discussion. | The standalone Limitations section was deleted. Benchmark scope and reproduction requirements are recorded in Methodology. |
| Explain or remove the thresholded success metric. | Removed `Success@0.50`, its calibration, and every threshold-based claim. The paper now reports continuous Text F1, ROUGE-L F1, and Utility F1. |
| Add full prompt-experiment reproduction. | Added the Prompt Obfuscation foundation reproduction and its verified USENIX citation, plus prompt-selection counts, 522 unique IDs, exact transformation parameters, 80/20 splits, all QLoRA and decoding settings, aggregation steps, and a separate end-to-end reproduction protocol. |
| Correct the mutual-information statement. | Replaced `I(s(p);Y|X,Z)>0` with a view-based decomposition conditioned on the model. Under `S -> Z -> Y` given `(X,theta_M)`, responses add zero conditional Shannon information; the paper only claims that they can be computationally useful recovery features. |

## Changes made

| Feedback | Revision |
|---|---|
| The abstract has vague findings and no numerical evidence. | Rewrote the abstract with the exact benchmark scale, Text F1 range (0.380--0.679), ROUGE-L F1 range (0.312--0.606), Utility F1 range (0.361--0.433), and the strongest conditions. |
| The paper needs an Introduction before the literature review. | Added an Introduction that states the problem, evaluation gap, attack idea, threat setting, contributions, and headline findings. |
| The literature review is too list-like. | Shortened it, organized it by protection family, and added a feature table describing the protected surface and retained utility signal. |
| The problem formulation is too long and there is no clear method section. | Replaced it with System Model and Problem Definition, followed by a separate Methodology and Attack Design section. |
| The threat model is ambiguous about black-box access and projected representations. | Defined a bounded chosen-query service-operator adversary. The client-local transform emits the cloud-visible payload `Z`, and the adversary observes `Z`, the model, queries, and responses; this is not a strict query-only black-box attack. |
| “Depending on the defense” makes attacker capabilities inconsistent. | Fixed the common attacker view: server-visible protected artifact, user queries, outputs, and auxiliary training data. Defense-specific artifacts are now treated as values of the same observable channel. |
| “Matched probes” is undefined. | Defined a matched probe as an input drawn from the cached test-input set associated with the same prompt and following that prompt's task format. |
| The recovery model and training process are unclear. | Added Qwen3-32B, QLoRA, 4-bit NF4, rank, alpha, dropout, target modules, epochs, batch size, gradient accumulation, learning rate, sequence length, split, decoding, and adapter count. |
| Recovery data and executable prompt construction are unclear. | Listed every recovery input channel, explained the six-field JSON targets, and stated that nonempty recovered fields are concatenated into an executable system prompt. |
| “Enough recovery” is not measurable. | Removed the binary “enough recovery” claim; the paper reports continuous Utility F1 instead of imposing an arbitrary universal cutoff. |
| Benchmark approximations may be mistaken for complete attacks on published defenses. | Renamed them mechanism-aligned or “-style” transformations, added a scope table, and stated which components of each original system are not reproduced. |
| The attack pipeline is difficult to follow. | Added an algorithm describing evidence construction, adapter training, structured prediction, executable-prompt construction, and evaluation. |

## Experimental scope recorded in Methodology

1. Behavior traces use cached original-prompt outputs as a high-utility proxy rather than outputs from complete protected-system implementations.
2. Recovery inputs include metadata such as prompt labels, paper titles, and six-dimensional scores, so this is a multi-evidence attack rather than a behavior-only attack.
3. Prompt IDs are held out from QLoRA training, but the same six matched queries are used for recovery evidence and Utility F1 evaluation.
4. Structured targets are rule-generated and may contain duplicated or incomplete fields.
5. The current results therefore support a mechanism-level pilot claim, not an end-to-end break of the eight cited systems.

## End-to-end reproduction track

Run an end-to-end, query-disjoint, channel-ablated experiment:

- generate outputs using each actual protected implementation;
- split evidence queries from final evaluation queries;
- compare metadata-only, representation-only, behavior-only, and combined evidence;
- use human-audited structured targets;
- repeat over multiple split and generation seeds.

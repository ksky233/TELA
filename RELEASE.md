# Release Notes

Prepared on 2026-10-28 for the ICCBD+AI 2026 conference code release.

## Scientific Implementation

- `models/`, `layers/`, and `data_provider/` are byte-identical to the verified research implementation.
- The training objective, EMA updates, validation selection, metrics, and forecasting computations are preserved.
- Runner defaults now match the conference protocol; reproduction commands also specify the complete protocol explicitly.
- The single-target runner exposes `S` and `MS` input settings.

## Packaging Changes

- Portable repository-relative paths replace the machine-specific default data directory.
- A fixed conference configuration and multi-site/seed/ablation runner are provided.
- The reproduction wrapper protects existing outputs; direct-run overwrite behavior is documented.
- Checkpoint loading checks all parameter keys, allowing only the known inactive all-zero legacy `query_power_bias` entries.
- Result audits additionally record package versions, device information, and input-data fingerprints.
- Chronological prediction export and focused software tests are included.
- Repository documentation is in English, and source code is released under the MIT License.
- Unused research scripts, trained weights, datasets, caches, figures, private registration files, and baseline source trees are outside this standalone package.

## Verification

The verification procedure includes unit tests, CPU training/checkpoint-reload
agreement on synthetic inputs, reference input-schema/fingerprint checks for all
three sites, original-checkpoint forward parity, full Site1 test-set reevaluation,
chronological forecast export, and execution of the attention/bias plotting scripts.
Detailed check outcomes and software versions are recorded in `VERIFICATION.json`.
This is a packaging verification, not a new full training run of the research experiments.

The source code is distributed under the MIT License in `LICENSE`. The final
repository URL can be added to the paper/reviewer-response materials once the
repository is public.

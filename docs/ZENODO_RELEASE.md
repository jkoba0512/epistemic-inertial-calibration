# Zenodo release procedure

This repository uses Zenodo's GitHub integration to archive immutable GitHub
releases and assign a DOI. The tagged repository contains the software and the
final S5 run-level research data. Manuscript files are excluded.

## One-time setup

1. Sign in to Zenodo using the GitHub account that owns the repository.
2. Open the Zenodo GitHub settings page.
3. Enable `jkoba0512/epistemic-inertial-calibration`.
4. Confirm that Zenodo can see the repository.

## Release checklist

1. Confirm that `README.md`, `CITATION.cff`, and `.zenodo.json` describe the
   intended release and publication status accurately.
2. Run the full S5 experiments and confirm that all files listed in
   `artifacts/README.md` are present.
3. Run:

   ```bash
   uv sync --locked --extra iiwa
   uv run pytest -q
   uv run ruff check .
   git diff --check
   ```

4. Confirm that the source archive contains no manuscript files:

   ```bash
   git archive --format=tar HEAD | tar -tf - | grep '^paper/' && exit 1 || true
   ```

5. Confirm that the worktree is clean and the release commit is pushed.
6. Create a new annotated version tag. Never reuse or move a tag after Zenodo
   has archived it.
7. Push the tag and create a GitHub Release from that exact tag.
8. Wait for Zenodo to archive the release.
9. Verify the title, creator, ORCID, version, license, and archived files.
10. Add the version DOI to `CITATION.cff` and the associated article. Add the
    concept DOI to project documentation when a link to the latest release is
    desired.

## DOI policy

- The version DOI identifies the exact software and data snapshot used by the
  article.
- The concept DOI resolves to the latest archived release.
- The article DOI and software-release DOI identify different research outputs
  and should not be used interchangeably.

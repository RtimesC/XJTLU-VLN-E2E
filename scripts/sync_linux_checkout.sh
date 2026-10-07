#!/usr/bin/env bash
set -euo pipefail

repo="${HOME}/Desktop/XJTLU-VLN-E2E"
expected_branch="${VLN_SYNC_BRANCH:-main}"

if [[ ! -d "${repo}/.git" ]]; then
  printf 'sync refused: Git checkout missing: %s\n' "${repo}" >&2
  exit 2
fi

current_branch="$(git -C "${repo}" branch --show-current)"
if [[ "${current_branch}" != "${expected_branch}" ]]; then
  printf 'sync refused: branch is %s; expected %s\n' "${current_branch}" "${expected_branch}" >&2
  exit 2
fi

if [[ -n "$(git -C "${repo}" status --porcelain=v1 --untracked-files=normal)" ]]; then
  printf 'sync refused: checkout has tracked or untracked changes\n' >&2
  git -C "${repo}" status --short >&2
  exit 2
fi

before="$(git -C "${repo}" rev-parse HEAD)"
git -C "${repo}" pull --ff-only origin "${expected_branch}"
after="$(git -C "${repo}" rev-parse HEAD)"
fetched="$(git -C "${repo}" rev-parse FETCH_HEAD)"

if [[ "${after}" != "${fetched}" ]]; then
  printf 'sync refused: checkout HEAD differs from fetched branch (%s vs %s)\n' "${after}" "${fetched}" >&2
  exit 2
fi

printf 'sync ok: %s -> %s (%s)\n' "${before}" "${after}" "${expected_branch}"

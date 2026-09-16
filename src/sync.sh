#!/usr/bin/env bash
# Artifact sync between Mac and Slurm cluster, keyed by run_id.
# rsync of models/, outputs/, scores/ for a given run_id (or everything).
#
# IMPORTANT: cluster host/user/path must be filled in below AFTER the
# authorized-access decision by the user. No remote host is hardcoded yet.
#
# Usage:
#   src/sync.sh pull-run <run_id>     cluster -> local
#   src/sync.sh push-run <run_id>     local -> cluster
#   src/sync.sh pull-all              cluster -> local (models/, outputs/, scores/)
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- configure after access decision -------------------------------------------
CLUSTER_HOST="${CLUSTER_HOST:-}"   # e.g. "user@login.cluster.org"
CLUSTER_PATH="${CLUSTER_PATH:-}"   # e.g. "~/model-compression/"
# --------------------------------------------------------------------------------

[ -n "$CLUSTER_HOST" ] || { echo "ERROR: CLUSTER_HOST not configured in src/sync.sh (pending access decision)."; exit 2; }
[ -n "$CLUSTER_PATH" ] || { echo "ERROR: CLUSTER_PATH not configured in src/sync.sh."; exit 2; }

cmd="${1:-}"
run="${2:-}"

case "$cmd" in
  pull-run)
    [ -n "$run" ] || { echo "usage: sync.sh pull-run <run_id>"; exit 1; }
    for d in "models" "outputs/$run" "scores/$run"; do
      rsync -az --progress "$CLUSTER_HOST:$CLUSTER_PATH/$d/" "$REPO/$d/" 2>/dev/null || true
    done
    echo "pulled $run"
    ;;
  push-run)
    [ -n "$run" ] || { echo "usage: sync.sh push-run <run_id>"; exit 1; }
    for d in "models" "outputs/$run" "scores/$run"; do
      rsync -az --progress "$REPO/$d/" "$CLUSTER_HOST:$CLUSTER_PATH/$d/" 2>/dev/null || true
    done
    echo "pushed $run"
    ;;
  pull-all)
    for d in "models" "outputs" "scores"; do
      rsync -az --progress "$CLUSTER_HOST:$CLUSTER_PATH/$d/" "$REPO/$d/"
    done
    echo "pulled all"
    ;;
  *)
    echo "usage: sync.sh {pull-run|push-run|pull-all} [run_id]"
    exit 1
    ;;
esac
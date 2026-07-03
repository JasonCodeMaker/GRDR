#!/usr/bin/env bash
# Shared per-cell functions for the four pipeline stages, plus the loop runner.
# Sourced by recall-stage.sh / rerank-stage.sh / recall-latency.sh / rerank-latency.sh.
# Dispatch rule (volatility split): grdr_ref -> GRDR/* (volatile, edited often);
# every other method -> baselines/* (stable). Cell bodies are a faithful port of
# the prior sweep_pass_a / sweep_pass_b logic with native paths.

default_ops_for () {
    case "$1" in
        grdr_ref)                      echo "20 50 100 200 300" ;;
        tiger|avg|t2vindexer)          echo "20 50 100 200 300" ;;
        eercf)                         echo "1 10 25 50" ;;
        hnsw|ivf|ivfpq|opq)            echo "20 40 100 200" ;;
        *) echo "" ;;
    esac
}

# Look up the paired ANN search-breadth knob for a budget K from a "K:val ..." map.
# Falls back to the last (max) value when K is unmapped.
ann_knob_for () {
    local map="$1" key="$2" pair last=""
    for pair in ${map}; do
        last="${pair#*:}"
        [ "${pair%%:*}" = "${key}" ] && { echo "${pair#*:}"; return 0; }
    done
    echo "${last}"
}

use_dataset_xpool () {
    local ds_lower="${1,,}"
    if [ "${EVAL_MODE:-zeroshot}" = "indist" ]; then
        PANDA_CKPT="$(xpool_ckpt_for "${ds_lower}")"
        CACHE_PARENT="$(xpool_cache_for "${ds_lower}")"
    fi
}

write_placeholder_candidate_json () {
    local out_json=$1 method=$2 ds_lower=$3 setting=$4 op=$5 status=$6 reason=$7
    mkdir -p "$(dirname "${out_json}")"
    "${PYTHON}" - "$out_json" "$method" "$ds_lower" "$setting" "$op" "$status" "$reason" <<'PY'
import json
import sys
from pathlib import Path

path, method, dataset, setting, op, status, reason = sys.argv[1:]
payload = {
    "metadata": {
        "method": method,
        "dataset": dataset,
        "setting": int(setting),
        "operating_point": int(op),
        "status": status,
        "reason": reason,
    },
    "metrics": {},
    "results": [],
}
Path(path).write_text(json.dumps(payload, indent=2) + "\n")
PY
}

cap_panda_gr_baseline_candidates () {
    local out_json=$1 method=$2 ds_lower=$3 op=$4
    [ "${ds_lower}" = "panda" ] || return 0
    [ -s "${out_json}" ] || return 0
    "${PYTHON}" "${FUNC_DIR}/utils/cap_candidate_json.py" \
        --path "${out_json}" \
        --cap "$(( GR_BASELINE_HANDOFF_CAP_MULT * op ))" \
        --method "${method}" \
        --cap-policy "panda_gr_baseline_cap=${GR_BASELINE_HANDOFF_CAP_MULT}x_beam"
}

# ---- Recall-Stage: stage-1 candidate export (effectiveness; CanHit) ----
cell_recall_stage () {
    local baseline=$1 ds=$2 setting=$3 op=$4
    local ds_lower="${ds,,}"
    # indist: ANN (hnsw/ivf) stage-1 is built over per-dataset X-Pool features.
    use_dataset_xpool "${ds_lower}"
    case "${baseline}" in
        grdr_ref)
            local cand_out="${CAND_GRDR_ROOT}/${ds_lower}/${ds_lower}_t${setting}_${op}_candidates.json"
            mkdir -p "$(dirname "${cand_out}")"
            if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${cand_out}" ]; then echo "  skip-existing: ${cand_out}"; return 0; fi
            if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY: grdr_ref ${ds} t${setting} op=${op}"; return 0; fi
            DS_LOWER="${ds_lower}" SETTING="${setting}" OP="${op}" CAND_OUT="${cand_out}" \
                LOG_DIR="${RECALL_STAGE_ROOT}/_logs" \
                bash "${FUNC_DIR}/GRDR/grdr_stage1_export.sh"
            ;;
        tiger|avg)
            local cand_out="${CAND_BASE_ROOT}/${baseline}/${ds_lower}/${ds_lower}_${baseline}_${op}_candidates_t${setting}.json"
            mkdir -p "$(dirname "${cand_out}")"
            if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${cand_out}" ]; then
                cap_panda_gr_baseline_candidates "${cand_out}" "${baseline}" "${ds_lower}" "${op}"
                echo "  skip-existing: ${cand_out}"
                return 0
            fi
            if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY: ${baseline} ${ds} t${setting} op=${op}"; return 0; fi
            # indist: per-dataset C=128/L=3 ckpt (env so the expanded TIGER_CKPT/AVG_CKPT parse as assignments).
            local indist_env=()
            if [ "${EVAL_MODE:-zeroshot}" = "indist" ]; then
                local gr_ck; gr_ck="$(baseline_gr_ckpt_for "${baseline}" "${ds_lower}")"
                [ -z "${gr_ck}" ] && { echo "ERROR: no indist ${baseline} ckpt for ${ds_lower} (train it first)"; return 2; }
                local ck_var=TIGER_CKPT; [ "${baseline}" = "avg" ] && ck_var=AVG_CKPT
                if [ "${ds_lower}" = "panda" ]; then
                    indist_env=(env "CODE_NUM=${GRDR_CODE_NUM}" "LAYERS=${GRDR_MAX_LENGTH}" "${ck_var}=${gr_ck}")
                else
                    indist_env=(env "CODE_NUM=128" "LAYERS=3" "${ck_var}=${gr_ck}")
                fi
            fi
            DEVICE="${DEVICE}" SEED="${SEED}" TOPK="${op}" \
                BASELINES="${baseline}" DATASETS="${ds}" SETTINGS="${setting}" \
                CAND_ROOT="${CAND_BASE_ROOT}" LOG_ROOT="${RECALL_STAGE_ROOT}/_logs/avg_tiger_op${op}" \
                "${indist_env[@]}" bash "${FUNC_DIR}/baselines/avg_tiger_stage1_export.sh"
            cap_panda_gr_baseline_candidates "${cand_out}" "${baseline}" "${ds_lower}" "${op}"
            ;;
        t2vindexer)
            local cand_out="${CAND_BASE_ROOT}/t2vindexer/${ds_lower}/${ds_lower}_t${setting}_${op}_candidates.json"
            mkdir -p "$(dirname "${cand_out}")"
            if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${cand_out}" ]; then echo "  skip-existing: ${cand_out}"; return 0; fi
            if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY: t2vindexer ${ds} t${setting} op=${op}"; return 0; fi
            if [ "${ds_lower}" = "panda" ]; then
                write_placeholder_candidate_json "${cand_out}" "t2vindexer" "${ds_lower}" "${setting}" "${op}" "OOM" \
                    "Panda k=4096 T2VIndexer export exceeds the figure workstation memory budget"
                return 0
            fi
            if [ "${EVAL_MODE:-zeroshot}" = "indist" ]; then
                # Real c128l3 export: main.py --mode eval (test()) writes the figure candidate JSON directly.
                local t2v_ck; t2v_ck="$(t2v_ckpt_for "${ds_lower}")"
                [ -z "${t2v_ck}" ] && { echo "ERROR: no indist t2vindexer ckpt for ${ds_lower} (train it first)"; return 2; }
                DS_LOWER="${ds_lower}" SETTING="${setting}" OP="${op}" INFER_CKPT="${t2v_ck}" CAND_OUT="${cand_out}" \
                    DEVICE="${DEVICE}" SEED="${SEED}" \
                    bash "${FUNC_DIR}/baselines/t2v_stage1_export.sh"
            else
                # zeroshot/panda: T2VIndexer OOMs at kary=4096; emit placeholder (effectiveness_validity=OOM).
                echo "{\"metadata\": {\"method\": \"t2vindexer\", \"dataset\": \"${ds_lower}\", \"setting\": ${setting}, \"operating_point\": ${op}, \"status\": \"OOM\"}, \"metrics\": {}, \"results\": []}" > "${cand_out}"
            fi
            ;;
        eercf)
            local cand_out="${CAND_BASE_ROOT}/eercf/${ds_lower}/${ds_lower}_eercf_${op}_candidates_t${setting}.json"
            mkdir -p "$(dirname "${cand_out}")"
            if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${cand_out}" ]; then echo "  skip-existing: ${cand_out}"; return 0; fi
            if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY: eercf ${ds} t${setting} op=${op}"; return 0; fi
            local eercf_dt
            case "${ds}" in
                MSRVTT) eercf_dt=msrvtt ;; ACTNET) eercf_dt=activity ;;
                DIDEMO) eercf_dt=didemo ;; LSMDC) eercf_dt=lsmdc ;;
                PANDA)  write_placeholder_candidate_json "${cand_out}" "eercf" "${ds_lower}" "${setting}" "${op}" "unsupported" \
                            "No Panda EERCF feature cache/checkpoint is available in this figure pipeline"; return 0 ;;
                *) echo "Unknown EERCF ds ${ds}" >&2; return 2 ;;
            esac
            # indist: per-dataset EERCF model (P3D feature cache stays shared/dataset-keyed).
            local eercf_init="${EERCF_INIT_MODEL}"
            [ "${EVAL_MODE:-zeroshot}" = "indist" ] && eercf_init="$(eercf_ckpt_for "${ds_lower}")"
            # Per-DEVICE MASTER_PORT avoids EADDRINUSE when EERCF runs concurrently on GPU 0+1.
            DEVICE="${DEVICE}" DATASETS="${eercf_dt}" SETTINGS="${setting}" RERANTOPK="${op}" \
                INIT_MODEL="${eercf_init}" CACHE_ROOT="${EERCF_CACHE_ROOT}" \
                MATRIX_ROOT="${MATRIX_ROOT}" LOG_DIR="${RECALL_STAGE_ROOT}/_logs/eercf" \
                MANIFEST_DIR="${SENTINEL_DIR}" \
                MASTER_PORT="$((29501 + DEVICE * 10))" \
                bash "${FUNC_DIR}/baselines/eercf/stage1_eercf.sh"
            local rc=$?; [ "${rc}" -ne 0 ] && return "${rc}"
            "${PYTHON}" "${FUNC_DIR}/baselines/eercf/import_eercf_matrix.py" \
                --pkg-root "${RUNTIME_ROOT}" --candidate-root "${CAND_BASE_ROOT}" \
                --query-set-root "${EERCF_QUERY_SET_ROOT}" --eercf-data-root "${EERCF_DATA_ROOT}" \
                --video-cache-root "${EERCF_CACHE_ROOT}" --datasets "${eercf_dt}" \
                --settings "${setting}" --rerantopk "${op}" --output-op "${op}" \
                --init-model "${eercf_init}"
            ;;
        hnsw|ivf|ivfpq|opq)
            # op = candidate budget K (num_candidates). ef_search/nprobe are PAIRED to K
            # (ANN_*_BY_K maps in _env.sh) so search effort grows with the budget: the
            # sweep traces a recall-vs-latency curve (Stage-1 effort rises with K) while
            # K sizes the full ANN retrieval output. HNSW ef_search=K; IVF-family nprobe
            # is 4/8/16/32; IVFPQ/OPQ additionally use PQ m=16, nbits=8.
            # Per-op candidate JSON (effectiveness) + ANN recall-latency captured here
            # (recall-latency.sh skips ann; this is the single owner of ANN timing).
            # A.1/A.2 can be gated independently (ANN_RUN_EFFECTIVENESS / ANN_RUN_LATENCY)
            # to split effectiveness and latency across separate GPUs.
            local ann_knob
            if [ "${baseline}" = "hnsw" ]; then ann_knob=$(ann_knob_for "${ANN_HNSW_EF_BY_K}" "${op}"); else ann_knob=$(ann_knob_for "${ANN_IVF_NPROBE_BY_K}" "${op}"); fi
            local cand_out="${CAND_BASE_ROOT}/${baseline}/${ds_lower}/${ds_lower}_ann_${baseline}_${op}_candidates_t${setting}.json"
            local lat_out="${RECALL_LATENCY_ROOT}/${baseline}/${ds_lower}/${ds_lower}_t${setting}_${op}_latency.json"
            mkdir -p "$(dirname "${cand_out}")" "$(dirname "${lat_out}")"
            # A.1 effectiveness (warm-cache search_batched).
            if [ "${ANN_RUN_EFFECTIVENESS:-1}" -ne 1 ]; then echo "  skip effectiveness (gated off): ${baseline} ${ds} t${setting} op=${op}"
            elif [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${cand_out}" ]; then
                echo "  skip-existing effectiveness: ${cand_out}"
            elif [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY effectiveness: ${baseline} ${ds} t${setting} op=${op}"
            else
                local knob_args="NUM_CANDIDATES=${op}"
                if [ "${baseline}" = "hnsw" ]; then knob_args="${knob_args} HNSW_EF_SEARCH=${ann_knob}"; else knob_args="${knob_args} IVF_NPROBE=${ann_knob} PQ_M=${ANN_PQ_M} PQ_NBITS=${ANN_PQ_NBITS}"; fi
                DEVICE="${DEVICE}" INDICES="${baseline}" DATASETS="${ds_lower}" SETTINGS="${setting}" \
                    CAND_ROOT="${CAND_BASE_ROOT}" CACHE_PARENT="${CACHE_PARENT}" PANDA_CKPT="${PANDA_CKPT}" \
                    OUT_ROOT="${RUNTIME_ROOT}/ann_baseline" OUTPUT_JSON="${cand_out}" \
                    LOG_ROOT="${RECALL_STAGE_ROOT}/_logs/ann_export" MANIFESTS_DIR="${SENTINEL_DIR}" \
                    env ${knob_args} bash "${FUNC_DIR}/baselines/ann_stage1_export.sh"
            fi
            # A.2 latency (cold-start search_per_query on the 200-query subset).
            if [ "${ANN_RUN_LATENCY:-1}" -ne 1 ]; then echo "  skip latency (gated off): ${baseline} ${ds} t${setting} op=${op}"
            elif [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${lat_out}" ]; then echo "  skip-existing latency: ${lat_out}"
            elif [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY latency: ${baseline} ${ds} t${setting} op=${op}"
            else
                local ann_lat_knob=""
                if [ "${baseline}" = "hnsw" ]; then ann_lat_knob="HNSW_EF_SEARCH=${ann_knob}"; else ann_lat_knob="IVF_NPROBE=${ann_knob} PQ_M=${ANN_PQ_M} PQ_NBITS=${ANN_PQ_NBITS}"; fi
                # `env` so the expanded ${ann_lat_knob} token is parsed as an assignment
                # (bash recognizes inline assignment prefixes before expansion, not after).
                env BASELINE="${baseline}" DATASET="${ds}" SETTING="${setting}" OP_VALUE="${op}" \
                    ${ann_lat_knob} DEVICE="${DEVICE}" SEED="${SEED}" \
                    bash "${FUNC_DIR}/baselines/recall_latency_cell.sh" \
                    || echo "WARN: ann recall-latency ${baseline}/${ds}/t${setting}/op${op} rc=$?"
            fi
            ;;
        *) echo "Unknown baseline: ${baseline}" >&2; return 2 ;;
    esac
}

# ---- Rerank-Stage: X-Pool rerank (effectiveness; R@K) ----
cell_rerank_stage () {
    local baseline=$1 ds=$2 setting=$3 op=$4
    local ds_lower="${ds,,}"
    # indist: per-dataset X-Pool reranker + in-distribution feature cache.
    use_dataset_xpool "${ds_lower}"
    local report_dir="${RERANK_STAGE_ROOT}/${baseline}/${ds_lower}/setting${setting}/${op}"
    mkdir -p "${report_dir}"
    local out_json="${report_dir}/xpool_eval.json"
    case "${baseline}" in
        hnsw|ivf|ivfpq|opq) echo "  skip ANN rerank-stage (native full ANN retrieval): ${baseline}"; return 0 ;;
    esac
    if [ "${SKIP_EXISTING}" -eq 1 ]; then
        case "${baseline}" in
            *)        [ -f "${out_json}" ]    && { echo "  skip-existing rerank: ${out_json}"; return 0; } ;;
        esac
    fi
    if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY rerank: ${baseline} ${ds} t${setting} op=${op}"; return 0; fi
    case "${baseline}" in
        hnsw|ivf|ivfpq|opq)
            return 0  # native full ANN retrieval: R@K comes from the ANN candidate JSON.
            ;;
        eercf)
            return 0  # native sim-matrix baseline: no X-Pool rerank (R@1 from its own candidates).
            ;;
        *)
            local cand_path=""
            case "${baseline}" in
                grdr_ref)   cand_path="${CAND_GRDR_ROOT}/${ds_lower}/${ds_lower}_t${setting}_${op}_candidates.json" ;;
                tiger|avg)  cand_path="${CAND_BASE_ROOT}/${baseline}/${ds_lower}/${ds_lower}_${baseline}_${op}_candidates_t${setting}.json" ;;
                t2vindexer) cand_path="${CAND_BASE_ROOT}/t2vindexer/${ds_lower}/${ds_lower}_t${setting}_${op}_candidates.json" ;;
            esac
            BASELINE="${baseline}" DATASET="${ds}" SETTING="${setting}" OP="${op}" \
                CAND_PATH="${cand_path}" OUT_DIR="${report_dir}" \
                PANDA_CKPT="${PANDA_CKPT}" CACHE_PARENT="${CACHE_PARENT}" \
                DEVICE="${DEVICE}" SEED="${SEED}" \
                bash "${FUNC_DIR}/track2_rerank_local.sh"
            ;;
    esac
}

# ---- Recall-Latency: stage-1 retrieval latency (efficiency) ----
cell_recall_latency () {
    local baseline=$1 ds=$2 setting=$3 op=$4
    local ds_lower="${ds,,}"
    # ANN stage-1 latency is owned by Recall-Stage A.2; skip here.
    case "${baseline}" in hnsw|ivf|ivfpq|opq) echo "  skip (recall-stage owns ANN latency): ${baseline}"; return 0 ;; esac
    local lat_out="${RECALL_LATENCY_ROOT}/${baseline}/${ds_lower}/${ds_lower}_t${setting}_${op}_latency.json"
    mkdir -p "$(dirname "${lat_out}")"
    if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${lat_out}" ]; then echo "  skip-existing: ${lat_out}"; return 0; fi
    if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY: recall-latency ${baseline} ${ds} t${setting} op=${op}"; return 0; fi
    if [ "${baseline}" = "grdr_ref" ]; then
        BASELINE="${baseline}" DATASET="${ds}" SETTING="${setting}" OP_VALUE="${op}" \
            DEVICE="${DEVICE}" SEED="${SEED}" \
            bash "${FUNC_DIR}/GRDR/grdr_recall_latency.sh"
    else
        BASELINE="${baseline}" DATASET="${ds}" SETTING="${setting}" OP_VALUE="${op}" \
            DEVICE="${DEVICE}" SEED="${SEED}" \
            bash "${FUNC_DIR}/baselines/recall_latency_cell.sh"
    fi
}

# ---- Rerank-Latency: X-Pool stage-2 latency (efficiency) ----
cell_rerank_latency () {
    local baseline=$1 ds=$2 setting=$3 op=$4
    local ds_lower="${ds,,}"
    # indist: per-dataset X-Pool reranker + in-distribution feature cache (stage-2 latency).
    use_dataset_xpool "${ds_lower}"
    case "${baseline}" in hnsw|ivf|ivfpq|opq)
        if [ "${ALLOW_ANN_RERANK_LATENCY:-0}" != "1" ]; then echo "  skip ANN rerank-latency: ${baseline}"; return 0; fi
        ;;
    esac
    local lat_in="${RECALL_LATENCY_ROOT}/${baseline}/${ds_lower}/${ds_lower}_t${setting}_${op}_latency.json"
    local stage2_report="${RERANK_LATENCY_ROOT}/${baseline}/${ds_lower}/setting${setting}/${op}/perquery_summary.json"
    if [ "${SKIP_EXISTING}" -eq 1 ] && [ -f "${stage2_report}" ]; then echo "  skip-existing stage2: ${stage2_report}"; return 0; fi
    if [ "${DRY_RUN}" -eq 1 ]; then echo "  DRY: rerank-latency ${baseline} ${ds} t${setting} op=${op}"; return 0; fi
    if [ ! -f "${lat_in}" ]; then echo "WARN: rerank-latency skipped (no recall-latency JSON ${lat_in})"; return 0; fi
    mkdir -p "$(dirname "${stage2_report}")"
    # Stage the recall-latency JSON under the stable filename the stage2 reader expects.
    local stage_dir="${RUNTIME_ROOT}/stage2_stage/${baseline}/${ds_lower}/op${op}/candidates_b/${baseline}/${ds_lower}"
    mkdir -p "${stage_dir}"
    cp -f "${lat_in}" "${stage_dir}/${ds_lower}_t${setting}_latency.json"
    local stage2_results_root="${RUNTIME_ROOT}/stage2_stage/${baseline}/${ds_lower}/op${op}/results_b"
    mkdir -p "${stage2_results_root}"
    ( \
        source "${CONDA_SH}" && conda activate "${CONDA_ENV_XPOOL}" && \
        BASELINES="${baseline}" DATASETS="${ds}" SETTINGS="${setting}" \
            DEVICE="${DEVICE}" SEED="${SEED}" WALL_CAP_S="${WALL_CAP_S}" PANDA_CKPT="${PANDA_CKPT}" \
            MANIFEST_DIR="${MANIFEST_DIR}" \
            CAND_B_ROOT="${RUNTIME_ROOT}/stage2_stage/${baseline}/${ds_lower}/op${op}/candidates_b" \
            RESULTS_B_ROOT="${stage2_results_root}" \
            bash "${FUNC_DIR}/baselines/stage2_latency.sh" \
    ) || echo "WARN: rerank-latency ${baseline}/${ds}/t${setting}/op${op} rc=$?"
    local produced="${stage2_results_root}/${baseline}/${ds_lower}/setting${setting}/perquery_summary.json"
    if [ -f "${produced}" ]; then
        cp -f "${produced}" "${stage2_report}"
        [ -f "${produced%.json}.csv" ] && cp -f "${produced%.json}.csv" "${stage2_report%.json}.csv"
    fi
}

# ---- Shared loop runner: <cell_fn> <sentinel_name> ----
run_stage_loop () {
    local cell_fn=$1 sentinel_name=$2
    echo "===== $(date -u +%FT%TZ) ${sentinel_name}: BASELINES='${BASELINES}' DATASETS='${DATASETS}' SETTINGS='${SETTINGS}' OPS='${OPERATING_POINTS}' ====="
    for baseline in ${BASELINES}; do
        local ops="${OPERATING_POINTS}"
        [ -z "${ops}" ] && ops=$(default_ops_for "${baseline}")
        for ds in ${DATASETS}; do
            for setting in ${SETTINGS}; do
                for op in ${ops}; do
                    echo "----- ${sentinel_name} cell: ${baseline} ${ds} t${setting} op=${op} -----"
                    "${cell_fn}" "${baseline}" "${ds}" "${setting}" "${op}" \
                        || echo "WARN: ${sentinel_name} ${baseline}/${ds}/t${setting}/op${op} rc=$?"
                done
            done
        done
    done
    if [ "${DRY_RUN}" -eq 1 ]; then
        echo "${sentinel_name} dry-run: sentinel not written"
        return 0
    fi
    mkdir -p "${SENTINEL_DIR}"
    local sentinel="${SENTINEL_DIR}/${sentinel_name}.done"
    { echo "status=ok"; echo "completed_at=$(date -u +%FT%TZ)"; echo "baselines=${BASELINES}"; echo "datasets=${DATASETS}"; echo "settings=${SETTINGS}"; } > "${sentinel}"
    echo "${sentinel_name} sentinel: ${sentinel}"
}

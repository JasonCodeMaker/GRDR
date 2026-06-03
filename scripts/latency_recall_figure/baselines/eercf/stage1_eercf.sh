#!/usr/bin/env bash
# P4 Track-2 Stage-1 -- Run EERCF zero-shot evaluation (4 datasets x 2 settings)
# under the P3d-Panda 2.15M ckpt, producing 8 sim-matrices that G4
# (import_eercf_matrix.py) slices into package-uniform candidate JSONs for the
# existing P4_track2_rerank_local_tiger_avg.sh (BASELINES=eercf).
#
# Setting 1: --do_eval only           -> writes <out>/sim_matrix.npy
#            pool = test-only, no cache read
# Setting 2: --do_eval --expanded_pool -> writes <out>/expanded_pool_sim_matrix.npy
#            pool = test + cached_train; --cached_features_path required
#
# Frame/caption resolution: package-local symlink tree under
#   ${PKG_ROOT}/_frame_roots/{ActivityNet,DiDeMo,LSMDC}/
# so a single --data_path resolves both. DiDeMo+LSMDC frame symlinks are
# load-bearing from P3.5 (do not recreate); ActivityNet symlinks and the
# caption-tree symlinks are added here (idempotent).
#
# Overrides via env:
#   DEVICE=1                  CUDA_VISIBLE_DEVICES
#   DATASETS="msrvtt activity didemo lsmdc"   space-separated
#   SETTINGS="1 2"
#   INIT_MODEL=<path>         P3d-Panda 2.15M ckpt (default below)
#   RERANTOPK=50              matches P3d training; output candidate top-100 is
#                             selected from the full sim-matrix row by G4 and
#                             is NOT bounded by rerantopk
#   BATCH_VAL=32 NUM_WORKERS=8
#   MAX_WORDS=32              MAX_FRAMES auto-set to 16 for activity, 12 otherwise
#   MASTER_PORT=29501
#   CONDA_ENV=semanticID
# Sentinels (rc captured per-cell; written even on failure):
#   $PKG_ROOT/manifests/P4_eercf_stage1_<dsl>_setting<n>.done
#   $PKG_ROOT/manifests/P4_eercf_stage1.done   (final overall rc)
set -u

DEVICE=${DEVICE:-1}
DATASETS=${DATASETS:-"msrvtt activity didemo lsmdc"}
SETTINGS=${SETTINGS:-"1 2"}
RERANTOPK=${RERANTOPK:-50}
BATCH_VAL=${BATCH_VAL:-32}
NUM_WORKERS=${NUM_WORKERS:-8}
MAX_WORDS=${MAX_WORDS:-32}
MASTER_PORT=${MASTER_PORT:-29501}

CONDA_ENV=${CONDA_ENV:-semanticID}
REPO=${REPO:-/home/uqzzha35/Project/SemanticID/EERCF}
PKG_ROOT=${PKG_ROOT:-/home/uqzzha35/Project/SemanticID/GRDR/output/evaluation_results/figures}
INIT_MODEL=${INIT_MODEL:-${PKG_ROOT}/output/eercf/panda_2150k_s42/pytorch_model.bin.best.0}

CACHE_ROOT=${CACHE_ROOT:-${PKG_ROOT}/cached_video_features_p3d}
MATRIX_ROOT=${MATRIX_ROOT:-${PKG_ROOT}/matrices/eercf}
LINK_ROOT=${LINK_ROOT:-${PKG_ROOT}/_frame_roots}
LOG_DIR=${LOG_DIR:-${PKG_ROOT}/logs}
MANIFEST_DIR=${MANIFEST_DIR:-${PKG_ROOT}/manifests}
COMBINED_LOG="${LOG_DIR}/p4_eercf_stage1.console.log"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}" "${MATRIX_ROOT}" \
         "${LINK_ROOT}/ActivityNet" "${LINK_ROOT}/DiDeMo" "${LINK_ROOT}/LSMDC"

# Package-local symlink tree so a single --data_path resolves both frame dirs
# (hardcoded subpaths in EERCF dataloaders) and caption JSONs.
# DiDeMo + LSMDC frame links exist from P3.5; ln -sfn is idempotent if the
# target matches. ActivityNet + caption-tree links are added here.
ln -sfn /data2/uqzzha35/VideoRetrieval/ActivityNet/Activity_Frames_224x224 "${LINK_ROOT}/ActivityNet/Activity_Frames"
ln -sfn /home/uqzzha35/Project/SemanticID/EERCF/data/ACTNET/video_retreival_caption "${LINK_ROOT}/ActivityNet/video_retreival_caption"
ln -sfn /data2/uqzzha35/VideoRetrieval/DiDeMo/train_frame_224x224 "${LINK_ROOT}/DiDeMo/train_frame"
ln -sfn /data2/uqzzha35/VideoRetrieval/DiDeMo/test_frame_224x224 "${LINK_ROOT}/DiDeMo/test_frame"
ln -sfn /home/uqzzha35/Project/SemanticID/EERCF/data/DIDEMO/video_retreival_caption "${LINK_ROOT}/DiDeMo/video_retreival_caption"
ln -sfn /data2/uqzzha35/VideoRetrieval/LSMDC/LSMDC_Frames_224x224 "${LINK_ROOT}/LSMDC/LSMDC_Frames_256"
ln -sfn /home/uqzzha35/Project/SemanticID/EERCF/data/LSMDC/video_retreival_caption "${LINK_ROOT}/LSMDC/video_retreival_caption"

# Per-dataset paths.
MSRVTT_FEATURES_PATH=/data2/uqzzha35/VideoRetrieval/msrvtt_data/MSRVTT_Frames
MSRVTT_TRAIN_CSV=/home/uqzzha35/Project/SemanticID/EERCF/data/MSRVTT/raw/MSRVTT_train.9k.csv
MSRVTT_TEST_CSV=/home/uqzzha35/Project/SemanticID/EERCF/data/MSRVTT/raw/MSRVTT_JSFUSION_test.csv
MSRVTT_DATA_JSON=/home/uqzzha35/Project/SemanticID/EERCF/data/MSRVTT/raw/MSRVTT_data.json

source /data2/uqzzha35/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

cd "${REPO}"

echo "===== $(date -u +%FT%TZ) P4 EERCF Stage-1 START (GPU=${DEVICE} ckpt=${INIT_MODEL} datasets='${DATASETS}' settings='${SETTINGS}') =====" | tee -a "${COMBINED_LOG}"

OVERALL_RC=0

for DS in ${DATASETS}; do
    case "${DS}" in
        msrvtt)
            DSL=msrvtt
            CACHE_SUB=MSRVTT/msrvtt
            MAX_FRAMES=12
            COMMON_EXTRA="--features_path ${MSRVTT_FEATURES_PATH} --train_csv ${MSRVTT_TRAIN_CSV} --val_csv ${MSRVTT_TEST_CSV} --test_csv ${MSRVTT_TEST_CSV} --multi_data_path ${MSRVTT_DATA_JSON} --data_path ${MSRVTT_DATA_JSON} --expand_msrvtt_sentences"
            ;;
        activity)
            DSL=activity
            CACHE_SUB=ACTNET/activity
            # Indist mode uses an ACTNET-specific EERCF ckpt trained at 24 frames;
            # the zeroshot Panda ckpt uses 16. Source-of-truth is the ckpt's
            # frame_logit_weight shape — keep this branch in sync with it.
            MAX_FRAMES=$([ "${EVAL_MODE:-zeroshot}" = "indist" ] && echo 24 || echo 16)
            COMMON_EXTRA="--data_path ${LINK_ROOT}/ActivityNet"
            ;;
        didemo)
            DSL=didemo
            CACHE_SUB=DIDEMO/didemo
            MAX_FRAMES=12
            COMMON_EXTRA="--data_path ${LINK_ROOT}/DiDeMo"
            ;;
        lsmdc)
            DSL=lsmdc
            CACHE_SUB=LSMDC/lsmdc
            MAX_FRAMES=12
            COMMON_EXTRA="--data_path ${LINK_ROOT}/LSMDC"
            ;;
        *)
            echo "[P4-eercf] unknown dataset '${DS}' (allowed: msrvtt activity didemo lsmdc) -- skipping" | tee -a "${COMBINED_LOG}"
            continue
            ;;
    esac

    for SETTING in ${SETTINGS}; do
        OUT_DIR="${MATRIX_ROOT}/${DSL}/setting${SETTING}"
        mkdir -p "${OUT_DIR}"
        CELL_LOG="${LOG_DIR}/p4_eercf_stage1_${DSL}_setting${SETTING}.console.log"
        SENTINEL="${MANIFEST_DIR}/P4_eercf_stage1_${DSL}_setting${SETTING}.done"

        POOL_FLAGS=""
        if [[ "${SETTING}" == "2" ]]; then
            POOL_FLAGS="--expanded_pool --cached_features_path ${CACHE_ROOT}/${CACHE_SUB}"
        fi

        HEADER="===== $(date -u +%FT%TZ) P4 eercf cell=${DSL} setting=${SETTING} GPU=${DEVICE} out=${OUT_DIR} ====="
        echo "${HEADER}" | tee -a "${COMBINED_LOG}" "${CELL_LOG}" >/dev/null

        # Skip-rebuild: the sim matrix is identical across rerantopk values (only the JSON
        # exporter downstream uses rerantopk). If the matrix file exists, skip the 1-2 hour
        # main_eercf.py call. Re-build is triggered by deleting the matrix file.
        EXPECTED_MATRIX="${OUT_DIR}/expanded_pool_sim_matrix.npy"
        [[ "${SETTING}" == "1" ]] && EXPECTED_MATRIX="${OUT_DIR}/sim_matrix.npy"
        if [ -f "${EXPECTED_MATRIX}" ] && [ "${EERCF_FORCE_REBUILD:-0}" != "1" ]; then
            echo "  [skip-rebuild] sim matrix exists: ${EXPECTED_MATRIX}" | tee -a "${COMBINED_LOG}" "${CELL_LOG}" >/dev/null
            continue
        fi

        set +u
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${DEVICE} python -m torch.distributed.run --nproc_per_node=1 --master_port=${MASTER_PORT} main_eercf.py \
            --do_eval \
            ${POOL_FLAGS} \
            --datatype ${DSL} \
            --init_model "${INIT_MODEL}" \
            --rerantopk ${RERANTOPK} \
            --num_thread_reader ${NUM_WORKERS} \
            --batch_size_val ${BATCH_VAL} \
            --max_words ${MAX_WORDS} \
            --max_frames ${MAX_FRAMES} \
            --feature_framerate 1 \
            --slice_framepos 2 \
            --freeze_layer_num 0 \
            --loose_type \
            --linear_patch 2d \
            --sim_header seqTransf \
            --pretrained_clip_name ViT-B/32 \
            --seed 42 \
            --output_dir "${OUT_DIR}" \
            ${COMMON_EXTRA} \
            2>&1 | tee -a "${CELL_LOG}"
        RC=${PIPESTATUS[0]}
        set -u

        echo "${RC}" > "${SENTINEL}"
        echo "===== $(date -u +%FT%TZ) P4 eercf cell=${DSL} setting=${SETTING} EXIT rc=${RC} sentinel=${SENTINEL} =====" | tee -a "${COMBINED_LOG}" "${CELL_LOG}" >/dev/null

        if [[ ${RC} -ne 0 ]]; then
            OVERALL_RC=${RC}
        fi
    done
done

FINAL_SENTINEL="${MANIFEST_DIR}/P4_eercf_stage1.done"
echo "${OVERALL_RC}" > "${FINAL_SENTINEL}"
echo "===== $(date -u +%FT%TZ) P4 EERCF Stage-1 DONE rc=${OVERALL_RC} sentinel=${FINAL_SENTINEL} =====" | tee -a "${COMBINED_LOG}"

exit ${OVERALL_RC}

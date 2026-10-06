#!/usr/bin/env bash
# Phase 1 check: does DeepStream decode video and run a detector on the T1000?
# Run inside the edge container:  bash edge/check_deepstream.sh
set -u
DS=/opt/nvidia/deepstream/deepstream
echo "== GPU"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "== DeepStream version"; deepstream-app --version-all 2>/dev/null | head -8 || echo "deepstream-app not found"
echo "== Key GStreamer plugins"
for p in nvv4l2decoder nvstreammux nvinfer nvtracker; do
  gst-inspect-1.0 "$p" >/dev/null 2>&1 && echo "  $p: OK" || echo "  $p: MISSING"
done

SAMPLE=$(ls $DS/samples/streams/sample_1080p_h264.mp4 2>/dev/null || ls $DS/samples/streams/*.mp4 2>/dev/null | head -1)
echo "== Hardware decode test on: $SAMPLE"
if [ -n "$SAMPLE" ]; then
  START=$(date +%s%N)
  if gst-launch-1.0 -q filesrc location="$SAMPLE" ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink sync=false >/dev/null 2>&1; then
    MS=$(( ($(date +%s%N) - START) / 1000000 )); echo "  decode: OK in ${MS} ms"
  else echo "  decode: FAILED"; fi
fi

echo "== Detector test: 2 streams -> decode -> batch -> detector (no display)"
# Uses the sample detector directly with gst-launch (the same building blocks our own pipeline will use).
# batch-size=2 reuses the TensorRT engine already built for the T1000 (..._b2_gpu0_fp16.engine).
INFER_DIR=$DS/samples/configs/deepstream-app
INFER_CFG=$INFER_DIR/config_infer_primary.txt
ls "$DS/samples/models/Primary_Detector/"*.engine 2>/dev/null | sed 's/^/  engine: /'
LOG=/workspace/edge/logs/ds_check_$(date +%Y%m%d_%H%M%S).log
mkdir -p /workspace/edge/logs
echo "  full log: ${LOG#/workspace/}"
echo "  running for up to 3 minutes; fps lines appear every few seconds"
cd "$INFER_DIR" && timeout 180 gst-launch-1.0 -v -e \
  nvstreammux name=m batch-size=2 width=1920 height=1080 batched-push-timeout=40000 ! \
  nvinfer config-file-path="$INFER_CFG" batch-size=2 ! \
  fpsdisplaysink name=fps text-overlay=false video-sink=fakesink sync=false signal-fps-measurements=true \
  filesrc location="$SAMPLE" ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_0 \
  filesrc location="$SAMPLE" ! qtdemux ! h264parse ! nvv4l2decoder ! m.sink_1 \
  > "$LOG" 2>&1
RC=$?
grep -oE "rendered: [0-9]+, dropped: [0-9]+, current: [0-9.]+, average: [0-9.]+" "$LOG" | tail -5 | sed 's/^/  batches /'
grep -E "ERROR|Error|error" "$LOG" | grep -v "error-post" | head -10 | sed 's/^/  /'
echo "  gst-launch exit code: $RC (0 = OK, 124 = still running after 3 min)"
echo "  note: each batch holds 2 frames (one per stream), so frames per second per stream = batch fps"
echo "== Done. Paste everything above back to Claude."

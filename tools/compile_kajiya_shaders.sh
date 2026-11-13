#!/usr/bin/env bash
# DEV: compiles the Kajiya HLSL shaders to SPIR-V with dxc (Vulkan SDK). Run from the repository root.
set -u
TMPDIR=${TMPDIR:-/tmp}
SRC=src/main/resources/kajiya/shaders
OUT=src/main/resources/kajiya/spv
mkdir -p "$OUT"
ok=0; fail=0
LIST="$(pwd)/tools/kajiya_shaders.txt"
cd "$SRC"
for f in $(cat "$LIST"); do
  case $f in
    *.rgen.hlsl|*.rchit.hlsl|*.rmiss.hlsl) prof=lib_6_4; entry="";;
    *_vs.hlsl) prof=vs_6_4; entry="-E main";;
    *_ps.hlsl) prof=ps_6_4; entry="-E main";;
    *) prof=cs_6_4; entry="-E main";;
  esac
  name=$(echo "$f" | tr '/' '_')
  if dxc -spirv -fspv-target-env=vulkan1.2 -WX -Ges -HV 2021 -T $prof $entry -Fo "../spv/$name.spv" "$f" > "$TMPDIR/dxc.log" 2>&1; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); echo "FAIL $f"; head -5 "$TMPDIR/dxc.log"
  fi
done
echo "compiled=$ok failed=$fail"

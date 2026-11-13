#!/bin/bash
# Re-evaluate every dumped frame of one state with MACE-LES, full text precision.
SID=$1
D=/publicfs10/fs10-m9/home/m9s005258/qy/cdft/scan/traj/$SID/run
OUT=${ENERGY_OUT:-/publicfs10/fs10-m9/home/m9s005258/qy/cdft/energy_out}
mkdir -p "$OUT"
CSV=$OUT/$SID.energy.csv
# 2001 data rows + 1 header
[ -s "$CSV" ] && [ "$(wc -l < $CSV)" -ge 2002 ] && exit 0
TMP=${SLURM_TMPDIR:-/tmp}/ev_$SID
mkdir -p "$TMP"
zcat $D/traj.dump.gz > $TMP/traj.dump
PS=$(grep "^pair_style" $D/in.les_md); PC=$(grep "^pair_coeff" $D/in.les_md)
DATA=$(grep "^read_data" $D/in.les_md | awk '{print $2}')
cat > $TMP/in.e <<EOT
units metal
atom_style molecular/kk
atom_modify map yes
boundary p p p
read_data $DATA
$PS
$PC
neighbor 2.0 bin
neigh_modify every 10 delay 0 check yes
run_style verlet/kk
thermo 1
thermo_style custom step atoms pe press
thermo_modify format float %.16g
rerun $TMP/traj.dump dump x y z
EOT
/publicfs10/fs10-m9/home/m9s005258/qy/build-cpu-mpi/lmp -k on t 1 -in $TMP/in.e -log $TMP/run.log > /dev/null 2>&1
printf 'step,pe_eV,press_bar\n' > "$CSV.tmp"
awk '/^ *[0-9]+ +[0-9]+ +-?[0-9.eE+-]+/ && $2>0 {print $1","$3","$4}' $TMP/run.log | awk '!seen[$1]++' >> "$CSV.tmp"
mv "$CSV.tmp" "$CSV"
echo "$SID $(($(wc -l < $CSV)-1)) $(date +%H:%M:%S) $(hostname)" >> $OUT/progress.txt
rm -rf "$TMP"
